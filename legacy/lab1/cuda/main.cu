#include "gemm.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {
struct Shape {
  const char* name;
  int m, n, k;
};
const std::vector<Shape> correctness_shapes = {
    {"scalar", 1, 1, 1}, {"rectangular", 7, 11, 5},
    {"one_tile", 16, 16, 16}, {"all_edges", 31, 37, 19},
    {"skinny", 64, 65, 33}, {"tall", 129, 17, 65}};
const std::vector<Shape> performance_shapes = {
    {"small_square", 256, 256, 256}, {"large_square", 1024, 1024, 1024},
    {"irregular", 257, 509, 131}, {"wide_small_k", 512, 2048, 64},
    {"tall_small_k", 2048, 256, 128}};
enum class Implementation { Cublas, Naive, Student };
const char* name(Implementation impl) {
  if (impl == Implementation::Cublas) return "cublas_fp32";
  if (impl == Implementation::Naive) return "naive_fp32";
  return "student_tiled_fp32";
}

struct DeviceBuffer {
  float* ptr = nullptr;
  explicit DeviceBuffer(size_t elements) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), elements * sizeof(float)));
  }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Context {
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  Context() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
      CUBLAS_CHECK(cublasCreate(&cublas));
      CUBLAS_CHECK(cublasSetStream(cublas, stream));
      // FP32 arithmetic across baselines. TF32/FP16 is a separate extension.
      CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH));
      CUBLAS_CHECK(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST));
    } catch (...) {
      if (cublas) cublasDestroy(cublas);
      cudaStreamDestroy(stream);
      throw;
    }
  }
  ~Context() {
    // Destructors cannot throw during unwinding; work errors are checked at
    // launches, synchronization and copies before cleanup.
    if (cublas) cublasDestroy(cublas);
    if (stream) cudaStreamDestroy(stream);
  }
};
struct Event {
  cudaEvent_t event = nullptr;
  Event() { CUDA_CHECK(cudaEventCreate(&event)); }
  ~Event() { if (event) cudaEventDestroy(event); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

void fill(std::vector<float>& values, uint32_t seed) {
  // Explicit integer generator gives identical inputs across implementations
  // and runs; values include signs and are not all ones or identities.
  for (float& value : values) {
    seed = seed * 1664525u + 1013904223u;
    value = static_cast<float>(static_cast<int>((seed >> 16) % 257u) - 128) / 128.0f;
  }
}

double cpu_element(const std::vector<float>& a, const std::vector<float>& b,
                   const Shape& s, size_t index) {
  const size_t row = index / static_cast<size_t>(s.n);
  const size_t col = index % static_cast<size_t>(s.n);
  double sum = 0;
  for (int p = 0; p < s.k; ++p) {
    sum += static_cast<double>(a[row * s.k + p]) * b[static_cast<size_t>(p) * s.n + col];
  }
  return sum;
}
std::vector<float> cpu_oracle(const std::vector<float>& a,
                              const std::vector<float>& b, const Shape& s) {
  std::vector<float> expected(static_cast<size_t>(s.m) * s.n);
  for (size_t index = 0; index < expected.size(); ++index) {
    expected[index] = static_cast<float>(cpu_element(a, b, s, index));
  }
  return expected;
}
void check_value(float actual, double expected, size_t index, const std::string& label) {
  constexpr double atol = 2e-4, rtol = 3e-4;
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(static_cast<double>(actual) - expected) > atol + rtol * std::abs(expected)) {
    throw std::runtime_error(label + " mismatch at " + std::to_string(index) +
                             ": got=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected));
  }
}
void check_all(const std::vector<float>& actual, const std::vector<float>& expected,
               const std::string& label) {
  if (actual.size() != expected.size()) throw std::runtime_error("output size mismatch");
  for (size_t index = 0; index < actual.size(); ++index) {
    check_value(actual[index], expected[index], index, label);
  }
}
void check_cpu_samples(const std::vector<float>& actual, const std::vector<float>& a,
                       const std::vector<float>& b, const Shape& s) {
  // Large benchmark shapes: independent CPU checks at both edges and 128
  // evenly spread positions; full comparison against cuBLAS follows below.
  const size_t total = actual.size();
  for (size_t i = 0; i < 128; ++i) {
    const size_t index = i * (total - 1) / 127;
    check_value(actual[index], cpu_element(a, b, s, index), index, "benchmark CPU sample");
  }
}

void launch(Context& ctx, Implementation impl, const DeviceBuffer& a,
             const DeviceBuffer& b, DeviceBuffer& c, const Shape& s) {
  if (impl == Implementation::Cublas) {
    launch_cublas(ctx.cublas, a.ptr, b.ptr, c.ptr, s.m, s.n, s.k);
  } else if (impl == Implementation::Naive) {
    launch_naive(a.ptr, b.ptr, c.ptr, s.m, s.n, s.k, ctx.stream);
  } else {
    launch_student(a.ptr, b.ptr, c.ptr, s.m, s.n, s.k, ctx.stream);
  }
}
void copy_inputs(Context& ctx, DeviceBuffer& da, DeviceBuffer& db,
                  const std::vector<float>& a, const std::vector<float>& b) {
  CUDA_CHECK(cudaMemcpyAsync(da.ptr, a.data(), a.size() * sizeof(float),
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(db.ptr, b.data(), b.size() * sizeof(float),
                             cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}
std::vector<float> execute_checked(Context& ctx, Implementation impl,
                                    const DeviceBuffer& da, const DeviceBuffer& db,
                                    DeviceBuffer& dc, const Shape& s) {
  const size_t total = static_cast<size_t>(s.m) * s.n;
  // Catch missing stores, including the untouched empty student kernel.
  std::vector<float> result(total, std::numeric_limits<float>::quiet_NaN());
  CUDA_CHECK(cudaMemcpyAsync(dc.ptr, result.data(), total * sizeof(float),
                             cudaMemcpyHostToDevice, ctx.stream));
  launch(ctx, impl, da, db, dc, s);
  CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  CUDA_CHECK(cudaMemcpy(result.data(), dc.ptr, total * sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}

void check_case(Context& ctx, const std::vector<Implementation>& implementations,
                 const Shape& s, const std::vector<float>& a,
                 const std::vector<float>& b, const std::vector<float>& expected,
                 uint32_t seed) {
  DeviceBuffer da(a.size()), db(b.size()), dc(expected.size());
  copy_inputs(ctx, da, db, a, b);
  for (Implementation impl : implementations) {
    const auto actual = execute_checked(ctx, impl, da, db, dc, s);
    check_all(actual, expected, std::string(s.name) + "/" + name(impl));
    std::cout << "# correctness_pass," << s.name << ',' << s.m << ',' << s.n << ','
              << s.k << ",seed=" << seed << ',' << name(impl) << '\n';
  }
}
void run_correctness(Context& ctx, const std::vector<Implementation>& implementations) {
  // Non-square fixture with manually computed answers; checks the oracle too.
  const Shape fixture{"fixed_dev", 2, 3, 4};
  const std::vector<float> a{1, 2, 3, 4, -1, 0, 2, -2};
  const std::vector<float> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<float> expected{70, 80, 90, -7, -8, -9};
  check_all(cpu_oracle(a, b, fixture), expected, "CPU oracle fixture");
  check_case(ctx, implementations, fixture, a, b, expected, 0);
  for (const Shape& s : correctness_shapes) {
    for (uint32_t seed : {1u, 17u}) {
      std::vector<float> va(static_cast<size_t>(s.m) * s.k);
      std::vector<float> vb(static_cast<size_t>(s.k) * s.n);
      fill(va, seed); fill(vb, seed + 101u);
      check_case(ctx, implementations, s, va, vb, cpu_oracle(va, vb, s), seed);
    }
  }
  const Shape zeros{"zero_a", 9, 17, 13};
  std::vector<float> za(static_cast<size_t>(zeros.m) * zeros.k, 0);
  std::vector<float> zb(static_cast<size_t>(zeros.k) * zeros.n);
  fill(zb, 31u);
  check_case(ctx, implementations, zeros, za, zb, cpu_oracle(za, zb, zeros), 31u);
}

float time_one(Context& ctx, Implementation impl, const DeviceBuffer& da,
                 const DeviceBuffer& db, DeviceBuffer& dc, const Shape& s, int iterations) {
  Event start, stop;
  CUDA_CHECK(cudaEventRecord(start.event, ctx.stream));
  for (int i = 0; i < iterations; ++i) launch(ctx, impl, da, db, dc, s);
  CUDA_CHECK(cudaEventRecord(stop.event, ctx.stream));
  CUDA_CHECK(cudaEventSynchronize(stop.event));
  float elapsed = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.event, stop.event));
  if (!(elapsed > 0) || !std::isfinite(elapsed)) throw std::runtime_error("invalid event timing");
  return elapsed / iterations;
}
void run_benchmark(Context& ctx, const std::vector<Implementation>& implementations, int iterations) {
  std::cout << "shape,m,n,k,seed,implementation,round,iterations,mean_ms,tflops\n";
  for (const Shape& s : performance_shapes) {
    const uint32_t seed = 20261005u;
    std::vector<float> a(static_cast<size_t>(s.m) * s.k), b(static_cast<size_t>(s.k) * s.n);
    fill(a, seed); fill(b, seed + 101u);
    DeviceBuffer da(a.size()), db(b.size()), dc(static_cast<size_t>(s.m) * s.n);
    copy_inputs(ctx, da, db, a, b);
    const auto strong_reference = execute_checked(ctx, Implementation::Cublas, da, db, dc, s);
    check_cpu_samples(strong_reference, a, b, s);
    for (Implementation impl : implementations) {
      const auto actual = execute_checked(ctx, impl, da, db, dc, s);
      check_all(actual, strong_reference, std::string(s.name) + "/" + name(impl));
      check_cpu_samples(actual, a, b, s);
      // Module loading, allocator initialization and first launches happen before
      // timing. Each implementation receives the same number of warmup launches.
      for (int warmup = 0; warmup < 5; ++warmup) launch(ctx, impl, da, db, dc, s);
      CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }
    for (size_t round = 0; round < 3; ++round) {
      // Rotate order to reduce systematic first/last thermal or clock bias.
      for (size_t offset = 0; offset < implementations.size(); ++offset) {
        const auto impl = implementations[(offset + round) % implementations.size()];
        const float ms = time_one(ctx, impl, da, db, dc, s, iterations);
        const double tflops = 2.0 * s.m * s.n * s.k / (static_cast<double>(ms) * 1e9);
        std::cout << s.name << ',' << s.m << ',' << s.n << ',' << s.k << ',' << seed
                  << ',' << name(impl) << ',' << round + 1 << ',' << iterations << ','
                  << std::setprecision(8) << ms << ',' << tflops << '\n';
      }
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    bool test = false, benchmark = false, reference_only = false;
    int iterations = 20;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--test") test = true;
      else if (arg == "--benchmark") benchmark = true;
      else if (arg == "--reference-only") reference_only = true;
      else if (arg == "--iterations" && i + 1 < argc) {
        const std::string value(argv[++i]);
        size_t consumed = 0;
        iterations = std::stoi(value, &consumed);
        if (consumed != value.size() || iterations < 1 || iterations > 10000)
          throw std::runtime_error("--iterations must be an integer in [1,10000]");
      } else if (arg == "--help") {
        std::cout << "gemm_lab --test|--benchmark [--reference-only] [--iterations 20]\n"
                     "--benchmark always runs correctness gates first.\n"
                     "--reference-only excludes student work and cannot certify lab completion.\n";
        return 0;
      } else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if (!test && !benchmark) throw std::runtime_error("use --test or --benchmark; see --help");
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 1) throw std::runtime_error("ENVIRONMENT_BLOCKED: no CUDA GPU; this is not a pass");
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    Context ctx;
    int runtime = 0, driver = 0, blas = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUBLAS_CHECK(cublasGetVersion(ctx.cublas, &blas));
    std::cout << "# scope=" << (reference_only ? "REFERENCE_ONLY" : "STUDENT_AND_REFERENCES")
              << ",gpu=" << props.name << ",cc=" << props.major << '.' << props.minor
              << ",device_memory_bytes=" << props.totalGlobalMem << '\n'
              << "# cuda_runtime=" << runtime << ",cuda_driver=" << driver << ",cublas=" << blas
              << ",compiled_cuda=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '\n'
              << "# precision=FP32_pedantic,warmup=5,rounds=3,atol=0.0002,rtol=0.0003\n";
    std::vector<Implementation> implementations{Implementation::Cublas, Implementation::Naive};
    if (!reference_only) implementations.push_back(Implementation::Student);
    run_correctness(ctx, implementations);
    if (benchmark) run_benchmark(ctx, implementations, iterations);
    std::cout << (reference_only ? "# REFERENCE_CHECKS_PASSED; STUDENT_NOT_EVALUATED\n"
                                 : "# STUDENT_CORRECTNESS_PASSED\n");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
