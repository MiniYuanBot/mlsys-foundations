// Part A3, B2, B3: generate CUDA C++ source.
//
//   build/lab3/test_a A3      build/lab3/test_b B2      build/lab3/test_b B3
//   build/lab3/tc_dump softmax     <- prints the plan and generated source
//
// Read codegen_core.cpp first: it shows the elementwise driver (which uses
// your A3 functions) and the GEMM template (which uses your B3 hooks).
// Generated code is compiled by NVRTC at runtime; compile errors are thrown
// with the full source attached, so read them like normal compiler errors.
#include "tc/codegen_detail.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace tc {

[[maybe_unused]] static std::logic_error todo(const char* what) {
    return std::logic_error(std::string("TODO ") + what + " is not implemented yet");
}

// TODO(A3a) one C expression per elementwise op. Wrap binary results in
// parentheses so they compose safely:
//   Neg (-a)  Exp expf(a)  Tanh tanhf(a)  Relu fmaxf(a, 0.0f)  Rsqrt rsqrtf(a)
//   Add (a + b)  Sub (a - b)  Mul (a * b)  Div (a / b)  Max fmaxf(a, b)
std::string emit_op(Op op, const std::vector<std::string>& a) {
    (void)op; (void)a;
    throw todo("A3a emit_op");
}

// TODO(A3b) load element (i, j) of a materialised operand with shape s,
// where i, j are long variables of the generated loop. A dimension of size
// 1 is broadcast (index 0):
//   [R, C] -> "buf[i * C + j]"     (C written as a number, e.g. "in0[i * 64 + j]")
//   [1, C] -> "buf[j]"     [R, 1] -> "buf[i]"     [1, 1] -> "buf[0]"
std::string emit_load(const std::string& buf, Shape s) {
    (void)buf; (void)s;
    throw todo("A3b emit_load");
}

namespace detail {

// A node is "full" if it varies along the row (cols > 1): it must be computed
// inside a column loop. Otherwise it is a per-row scalar (shape [R,1] or [1,1]).
[[maybe_unused]] static bool is_full(const Ctx& c, int id) { return c.g.node(id).shape.cols > 1; }

// Provided: the full nodes of this kernel that `target` needs, in topological
// order (stops at per-row scalars, which are already in registers, and at
// materialised operands).
[[maybe_unused]] static std::vector<int> full_closure(const Ctx& c, int target) {
    std::vector<int> out;
    std::set<int> seen;
    std::function<void(int)> go = [&](int id) {
        if (!c.inside.count(id) || !is_full(c, id) || !seen.insert(id).second) return;
        for (int s : c.g.node(id).srcs) go(s);
        out.push_back(id);
    };
    go(target);
    return out;
}

// TODO(B2) Row kernel: one thread block per row i, 256 threads, columns
// j = threadIdx.x, threadIdx.x + blockDim.x, ... < C. Generate:
//
//   <kPrelude>
//   <signature(c, "long R, long C")> {
//     const long i = blockIdx.x;
//     __shared__ float tc_smem[32];
//     <emit every per-row scalar node that is ready (all in-kernel srcs emitted)>
//     for each reduction r in k.nodes (topological order):
//       float acc<r> = 0.0f;            (RowMax: -INFINITY)
//       for (long j = threadIdx.x; j < C; j += blockDim.x) {
//         <emit_ssa(full_closure(c, src of r))>
//         acc<r> += v<src>;             (RowMax: acc<r> = fmaxf(acc<r>, v<src>);)
//       }
//       const float v<r> = tc_block_sum(acc<r>, tc_smem);   (tc_block_max)
//       <emit every per-row scalar node that is now ready>
//     if the output is full:
//       for (long j = ...) { <emit_ssa(full_closure(c, output))>  out[i * C + j] = v<out>; }
//     else:
//       if (threadIdx.x == 0) out[i] = v<out>;
//   }
//
// "Ready": a per-row scalar, non-reduction node whose srcs inside the kernel
// have all been emitted. Emit it with emit_ssa (it uses only i, never j).
// Full values are RECOMPUTED in every loop that needs them: softmax's x - m
// is evaluated twice. That costs a few FLOPs but saves writing [R, C]
// temporaries to memory, the whole point of fusion.
std::string codegen_row(const Ctx& c) {
    (void)c;
    throw todo("B2 codegen_row");
}

// TODO(B3) hooks for the GEMM template in codegen_core.cpp. The kernel holds
// one MatMul node mm = MatMul(a, b) plus elementwise nodes that are either
//   PROLOGUE: inside the kernel and (transitively) feeding a  -> fused into the A load
//   EPILOGUE: everything else inside except mm                -> applied to the result
// b is always a materialised operand (the partitioner guarantees it).
//
//   load_a:   (i, j) = (row of A, column of A)
//             emit_ssa(prologue nodes, topological order) then "return <operand(c, a)>;"
//             If a is materialised there are no prologue nodes: just return the load.
//   load_b:   "return <operand(c, b)>;"   (i, j) = (row of B, column of B)
//   epilogue: (i, j) = output position; the GEMM result arrives as `acc`:
//             "const float v<mm> = acc;" then emit_ssa(epilogue nodes) then
//             "return v<output>;"
// Bodies are plain statement lists; indent with two spaces.
MatmulHooks matmul_hooks(const Ctx& c) {
    (void)c;
    throw todo("B3 matmul_hooks");
}

}  // namespace detail
}  // namespace tc
