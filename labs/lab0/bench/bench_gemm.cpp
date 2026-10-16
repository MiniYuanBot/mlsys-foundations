// GEMM benchmark: prints CSV to stdout.
//
//   bench_gemm                      all implemented kernels, sizes 128..1024
//   bench_gemm --sizes 256,512,2048 --threads 4 --repeats 5
//
// Every timed kernel is first checked against the naive kernel on the same
// inputs; unimplemented kernels (still throwing TODO) are skipped.
// Square sizes only: GFLOP/s = 2*N^3 / time.
#include "mt/gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace mt;
using Clock = std::chrono::steady_clock;

static std::vector<int> parse_list(const std::string &s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stoi(item));
    return out;
}

int main(int argc, char **argv) {
    std::vector<int> sizes{128, 256, 512, 1024};
    int repeats = 3, threads = 0, naive_max = 1024;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--sizes")
            sizes = parse_list(next());
        else if (a == "--repeats")
            repeats = std::stoi(next());
        else if (a == "--threads")
            threads = std::stoi(next());
        else if (a == "--naive-max")
            naive_max = std::stoi(next());
        else {
            std::fprintf(
                stderr,
                "usage: bench_gemm [--sizes a,b,c] [--repeats r] [--threads t] [--naive-max n]\n");
            return 2;
        }
    }
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    std::fprintf(stderr, "# hardware threads: %d, MR=%d NR=%d MC=%d KC=%d NC=%d\n", hw, MR, NR, MC,
                 KC, NC);
    std::printf("kernel,threads,n,repeat,ms,gflops\n");

    const GemmKernel kernels[] = {GemmKernel::Naive, GemmKernel::Reordered, GemmKernel::Blocked,
                                  GemmKernel::Packed, GemmKernel::Parallel};
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(-1, 1);
    volatile float sink = 0;

    for (int n : sizes) {
        std::vector<float> A(static_cast<size_t>(n) * n), B(A.size()), C(A.size()), ref(A.size());
        for (auto &x : A) x = u(rng);
        for (auto &x : B) x = u(rng);
        // Oracle: your B1 kernel if it works (fast), otherwise the naive one.
        try {
            gemm_reordered(n, n, n, A.data(), n, B.data(), n, ref.data(), n);
        } catch (const std::logic_error &) {
            gemm_naive(n, n, n, A.data(), n, B.data(), n, ref.data(), n);
        }
        for (GemmKernel k : kernels) {
            if (k == GemmKernel::Naive && n > naive_max) continue;
            auto run = [&]() {
                if (k == GemmKernel::Parallel)
                    gemm_parallel(n, n, n, A.data(), n, B.data(), n, C.data(), n, threads);
                else
                    gemm(k, n, n, n, A.data(), n, B.data(), n, C.data(), n);
            };
            try {
                run(); // warm-up + correctness
            } catch (const std::logic_error &e) {
                std::fprintf(stderr, "# skip %s: %s\n", kernel_name(k), e.what());
                continue;
            }
            double worst = 0;
            for (size_t i = 0; i < C.size(); ++i)
                worst = std::max(worst, std::fabs(double(C[i]) - ref[i]));
            if (!(worst < 1e-2)) {
                std::fprintf(stderr, "# %s is WRONG at n=%d (max err %g); not timing it\n",
                             kernel_name(k), n, worst);
                continue;
            }
            for (int r = 1; r <= repeats; ++r) {
                // Repeat small sizes so each measurement lasts >= ~50 ms.
                int iters = 1;
                double ms = 0;
                while (true) {
                    const auto t0 = Clock::now();
                    for (int it = 0; it < iters; ++it) run();
                    ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    if (ms >= 50 || iters >= 1 << 16) break;
                    iters *= 2;
                }
                sink = C[0];
                ms /= iters;
                const double gflops = 2.0 * n * n * n / (ms * 1e6);
                const int t = k == GemmKernel::Parallel ? (threads > 0 ? threads : hw) : 1;
                std::printf("%s,%d,%d,%d,%.4f,%.2f\n", kernel_name(k), t, n, r, ms, gflops);
                std::fflush(stdout);
            }
        }
    }
    (void)sink;
    return 0;
}
