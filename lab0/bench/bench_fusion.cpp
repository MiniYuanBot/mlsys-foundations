// Fusion benchmark on two workloads over large vectors (default 2^24 floats = 64 MiB each).
//
//   affine  y = relu(x * w + b) * 0.5 + x       cheap ops   -> memory-bound
//   gelu    y = 0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x^3)))
//                                               tanh-heavy  -> compute-bound
// Variants:
//   unfused      one kernel per op (eager PyTorch style), scalar interpreter
//   fused        C2 schedule + C3 compile, scalar interpreter
//   tiled        C2 schedule + C3 compile, C4 tiled interpreter
//   handwritten  the same formula as a plain C++ loop (what a code generator emits)
//   copy         y = x, a memory-bandwidth yardstick
//
// usage: bench_fusion [--n 16777216] [--repeats 3]
#include "mt/lazy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>

using namespace mt;
using namespace mt::lazy;
using Clock = std::chrono::steady_clock;

static double time_ms(const std::function<void()>& fn) {
    const auto t0 = Clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static Tensor random_tensor(int64_t n, unsigned seed) {
    std::vector<float> v(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-3, 3);
    for (auto& x : v) x = u(rng);
    return Tensor::from_vector({n}, std::move(v));
}

// What a code generator would emit for each workload: one plain loop.
static void affine_loop(int64_t n, const float* __restrict x, const float* __restrict w,
                        const float* __restrict b, float* __restrict out) {
    for (int64_t i = 0; i < n; ++i) {
        const float t = x[i] * w[i] + b[i];
        out[i] = (t > 0.f ? t : 0.f) * 0.5f + x[i];
    }
}

static void gelu_loop(int64_t n, const float* __restrict x, float* __restrict out) {
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        out[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

static void run_workload(const char* workload, const LazyTensor& y,
                         const std::function<void(float*)>& handwritten, int64_t n,
                         int64_t hand_bytes, int repeats) {
    std::vector<float> want(n), scratch(n);
    handwritten(want.data());
    struct Variant {
        const char* name;
        std::function<Tensor(ExecStats*)> run;
    };
    Variant variants[] = {
        {"unfused", [&](ExecStats* s) { return realize_unfused(y, s); }},
        {"fused", [&](ExecStats* s) { return realize_fused(y, Interp::Scalar, s); }},
        {"tiled", [&](ExecStats* s) { return realize_fused(y, Interp::Tiled, s); }},
    };
    for (auto& v : variants) {
        ExecStats st;
        Tensor out;
        try {
            out = v.run(&st);
        } catch (const std::logic_error& e) {
            std::fprintf(stderr, "# skip %s/%s: %s\n", workload, v.name, e.what());
            continue;
        }
        double err = 0;
        for (int64_t i = 0; i < n; ++i)
            err = std::max(err, double(std::fabs(out.data()[i] - want[i])));
        const int64_t moved = st.bytes_read + st.bytes_written;
        for (int r = 1; r <= repeats; ++r) {
            const double ms = time_ms([&] { out = v.run(nullptr); });
            std::printf("%s,%s,%d,%.3f,%d,%lld,%.2f,%.2e\n", workload, v.name, r, ms, st.kernels,
                        (long long)moved, moved / (ms * 1e6), err);
            std::fflush(stdout);
        }
    }
    for (int r = 1; r <= repeats; ++r) {
        const double ms = time_ms([&] { handwritten(scratch.data()); });
        std::printf("%s,handwritten,%d,%.3f,1,%lld,%.2f,0\n", workload, r, ms,
                    (long long)hand_bytes, hand_bytes / (ms * 1e6));
    }
}

int main(int argc, char** argv) {
    int64_t n = 1 << 24;
    int repeats = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n" && i + 1 < argc)
            n = std::stoll(argv[++i]);
        else if (a == "--repeats" && i + 1 < argc)
            repeats = std::stoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: bench_fusion [--n N] [--repeats R]\n");
            return 2;
        }
    }
    const int64_t F = sizeof(float);
    Tensor xt = random_tensor(n, 1), wt = random_tensor(n, 2), bt = random_tensor(n, 3);
    const float *xp = xt.data(), *wp = wt.data(), *bp = bt.data();
    auto x = LazyTensor::input(xt), w = LazyTensor::input(wt), b = LazyTensor::input(bt);

    std::printf("workload,variant,repeat,ms,kernels,bytes_moved,effective_GBps,max_abs_err\n");

    auto affine = (x * w + b).relu() * 0.5f + x;
    run_workload(
        "affine", affine, [&](float* out) { affine_loop(n, xp, wp, bp, out); }, n, 4 * n * F,
        repeats);

    auto inner = 0.7978845608f * (x + 0.044715f * (x * x * x));
    auto gelu = 0.5f * x * (1.0f + inner.tanh());
    run_workload("gelu", gelu, [&](float* out) { gelu_loop(n, xp, out); }, n, 2 * n * F, repeats);

    std::vector<float> dst(n);
    for (int r = 1; r <= repeats; ++r) {
        const double ms = time_ms([&] {
            for (int64_t i = 0; i < n; ++i) dst[i] = xp[i];
        });
        std::printf("bandwidth,copy,%d,%.3f,1,%lld,%.2f,0\n", r, ms, (long long)(2 * n * F),
                    2.0 * n * F / (ms * 1e6));
    }
    return 0;
}
