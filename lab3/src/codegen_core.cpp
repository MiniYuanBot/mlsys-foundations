// Provided: codegen helpers, the elementwise kernel driver and the GEMM template.
#include "tc/codegen_detail.hpp"

#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace tc::detail {

Ctx make_ctx(const Graph& g, const Kernel& k) {
    Ctx c{g, k, {}, {}};
    for (size_t i = 0; i < k.reads.size(); ++i) c.buf[k.reads[i]] = "in" + std::to_string(i);
    c.inside.insert(k.nodes.begin(), k.nodes.end());
    return c;
}

std::string literal(double v) {
    if (std::isinf(v)) return v > 0 ? "INFINITY" : "(-INFINITY)";
    std::ostringstream os;
    os << std::setprecision(9) << std::showpoint << static_cast<float>(v) << "f";
    std::string s = os.str();
    return s[0] == '-' ? "(" + s + ")" : s;
}

std::string operand(const Ctx& c, int id) {
    if (c.inside.count(id)) return "v" + std::to_string(id);
    const Node& n = c.g.node(id);
    if (n.op == Op::Const) return literal(n.value);
    auto it = c.buf.find(id);
    if (it == c.buf.end())
        throw std::logic_error("codegen: operand n" + std::to_string(id) + " is not available");
    return emit_load(it->second, n.shape);
}

void emit_ssa(const Ctx& c, const std::vector<int>& nodes, std::ostringstream& os,
              const std::string& indent) {
    for (int id : nodes) {
        const Node& n = c.g.node(id);
        if (!is_elementwise(n.op))
            throw std::logic_error(std::string("emit_ssa: not elementwise: ") + op_name(n.op));
        std::vector<std::string> args;
        for (int s : n.srcs) args.push_back(operand(c, s));
        os << indent << "const float v" << id << " = " << emit_op(n.op, args) << ";\n";
    }
}

std::string signature(const Ctx& c, const std::string& dims) {
    std::ostringstream os;
    os << "extern \"C\" __global__ void tc_kernel(";
    for (size_t i = 0; i < c.k.reads.size(); ++i) os << "const float* __restrict__ in" << i << ", ";
    os << "float* __restrict__ out, " << dims << ")";
    return os.str();
}

// Parameter list / argument list of the inputs, for the GEMM helper functions.
static std::string in_params(const Ctx& c) {
    std::string s;
    for (size_t i = 0; i < c.k.reads.size(); ++i)
        s += "const float* __restrict__ in" + std::to_string(i) + ", ";
    return s;
}
static std::string in_args(const Ctx& c) {
    std::string s;
    for (size_t i = 0; i < c.k.reads.size(); ++i) s += "in" + std::to_string(i) + ", ";
    return s;
}

const char* kPrelude = R"(
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
__device__ __forceinline__ float tc_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float tc_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Block-wide reductions: every thread of the block returns the result.
// `smem` needs 32 floats. Safe to call several times in a row.
__device__ float tc_block_sum(float v, float* smem) {
    v = tc_warp_sum(v);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = (blockDim.x + 31) / 32;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = lane < nwarps ? smem[lane] : 0.f;
    v = tc_warp_sum(v);
    __syncthreads();
    return v;
}
__device__ float tc_block_max(float v, float* smem) {
    v = tc_warp_max(v);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = (blockDim.x + 31) / 32;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = lane < nwarps ? smem[lane] : -INFINITY;
    v = tc_warp_max(v);
    __syncthreads();
    return v;
}
)";

// Elementwise: one thread per output element of the [R, C] grid, grid-stride.
std::string codegen_elementwise(const Ctx& c) {
    std::ostringstream os;
    os << kPrelude << signature(c, "long R, long C") << " {\n"
       << "  for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < R * C; t += "
          "(long)gridDim.x * blockDim.x) {\n"
       << "    const long i = t / C, j = t % C;\n";
    emit_ssa(c, c.k.nodes, os, "    ");
    os << "    out[t] = v" << c.k.output << ";\n  }\n}\n";
    return os.str();
}

// GEMM template: 64 x 64 output tile per block, 256 threads, 4 x 4 outputs
// per thread, BK = 16. Bounds-checked, any M, N, K. Your hooks decide what
// "loading A", "loading B" and "storing C" mean.
std::string codegen_matmul(const Ctx& c) {
    const MatmulHooks h = matmul_hooks(c);
    const std::string P = in_params(c), A = in_args(c);
    std::ostringstream os;
    os << kPrelude << "__device__ __forceinline__ float tc_load_a(" << P << "long i, long j) {\n"
       << h.load_a << "}\n"
       << "__device__ __forceinline__ float tc_load_b(" << P << "long i, long j) {\n"
       << h.load_b << "}\n"
       << "__device__ __forceinline__ float tc_epilogue(float acc, " << P << "long i, long j) {\n"
       << h.epilogue << "}\n"
       << signature(c, "long M, long N, long K") << R"( {
  constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
  __shared__ float As[BK][BM + 4];
  __shared__ float Bs[BK][BN + 4];
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  const long m0 = (long)blockIdx.y * BM, n0 = (long)blockIdx.x * BN;
  float acc[TM][TN] = {};
  for (long k0 = 0; k0 < K; k0 += BK) {
    for (int e = threadIdx.x; e < BM * BK; e += 256) {
      const int r = e / BK, kk = e % BK;
      const long i = m0 + r, j = k0 + kk;
      As[kk][r] = (i < M && j < K) ? tc_load_a()"
       << A << R"(i, j) : 0.f;
    }
    for (int e = threadIdx.x; e < BK * BN; e += 256) {
      const int kk = e / BN, cc = e % BN;
      const long i = k0 + kk, j = n0 + cc;
      Bs[kk][cc] = (i < K && j < N) ? tc_load_b()"
       << A << R"(i, j) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
#pragma unroll
      for (int x = 0; x < TM; ++x) a[x] = As[kk][ty * TM + x];
#pragma unroll
      for (int y = 0; y < TN; ++y) b[y] = Bs[kk][tx * TN + y];
#pragma unroll
      for (int x = 0; x < TM; ++x)
#pragma unroll
        for (int y = 0; y < TN; ++y) acc[x][y] += a[x] * b[y];
    }
    __syncthreads();
  }
#pragma unroll
  for (int x = 0; x < TM; ++x)
#pragma unroll
    for (int y = 0; y < TN; ++y) {
      const long i = m0 + ty * TM + x, j = n0 + tx * TN + y;
      if (i < M && j < N) out[i * N + j] = tc_epilogue(acc[x][y], )"
       << A << R"(i, j);
    }
}
)";
    return os.str();
}

} // namespace tc::detail

namespace tc {

std::string codegen(const Graph& g, const Kernel& k) {
    const detail::Ctx c = detail::make_ctx(g, k);
    switch (k.kind) {
    case KernelKind::Elementwise:
        return detail::codegen_elementwise(c);
    case KernelKind::Row:
        return detail::codegen_row(c);
    case KernelKind::MatMul:
        return detail::codegen_matmul(c);
    }
    throw std::logic_error("codegen: bad kernel kind");
}

} // namespace tc
