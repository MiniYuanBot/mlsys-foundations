// GEMM benchmark: every implemented kernel vs cuBLAS on square matrices.
//
//   bench_gemm                               all kernels, n = 1024, 2048, 4096
//   bench_gemm --only sgemm --sizes 4096 --repeats 5
//   bench_gemm --only hgemm
//
// Each kernel is checked against cuBLAS on the same inputs before it is timed;
// a wrong kernel is reported and skipped. TFLOP/s = 2 n^3 / time.
#include "l1/kernels.cuh"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

using namespace l1;

static std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stoi(item));
    return out;
}

template <class T, class Fn>
static void run_family(const char* dtype, const std::vector<std::pair<const char*, Fn>>& kernels,
                       Fn ref, int n, int repeats, int naive_max) {
    const auto Af = make_input(static_cast<size_t>(n) * n, 1),
               Bf = make_input(static_cast<size_t>(n) * n, 2);
    DeviceBuffer<T> A(Af.size()), B(Bf.size());
    if constexpr (std::is_same_v<T, float>) {
        A.upload(Af);
        B.upload(Bf);
    } else {
        A.upload(to_half(Af));
        B.upload(to_half(Bf));
    }
    DeviceBuffer<float> C(Af.size()), R(Af.size());
    ref(n, n, n, A.get(), B.get(), R.get());
    const auto want = R.download();
    float ref_ms = time_ms([&] { ref(n, n, n, A.get(), B.get(), R.get()); });

    auto emit = [&](const char* name, int r, float ms) {
        const double tflops = 2.0 * n * n * n / (ms * 1e9);
        std::printf("%s,%s,%d,%d,%.4f,%.2f,%.1f\n", name, dtype, n, r, ms, tflops,
                    100.0 * ref_ms / ms);
        std::fflush(stdout);
    };
    for (int r = 1; r <= repeats; ++r) {
        if (r > 1) ref_ms = time_ms([&] { ref(n, n, n, A.get(), B.get(), R.get()); });
        emit("cublas", r, ref_ms);
    }
    for (const auto& entry : kernels) {
        const char* name = entry.first;
        const Fn fn = entry.second;
        if (std::string(name) == "naive" && n > naive_max) continue;
        C.poison();
        try {
            fn(n, n, n, A.get(), B.get(), C.get());
            CUDA_CHECK(cudaDeviceSynchronize());
        } catch (const NotImplemented& e) {
            std::fprintf(stderr, "# skip %s: %s\n", name, e.what());
            continue;
        }
        const auto got = C.download();
        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = std::fabs(double(got[i]) - want[i]);
            worst = std::isfinite(got[i]) ? std::max(worst, d) : INFINITY;
        }
        if (!(worst <= 1e-2 * std::sqrt(n))) {
            std::fprintf(stderr, "# %s is WRONG at n=%d (max |err| = %g); not timing it\n", name, n,
                         worst);
            continue;
        }
        for (int r = 1; r <= repeats; ++r)
            emit(name, r, time_ms([&] { fn(n, n, n, A.get(), B.get(), C.get()); }));
    }
}

int main(int argc, char** argv) {
    std::vector<int> sizes{1024, 2048, 4096};
    std::string only;
    int repeats = 3, naive_max = 2048;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--sizes")
            sizes = parse_list(next());
        else if (a == "--only")
            only = next();
        else if (a == "--repeats")
            repeats = std::stoi(next());
        else if (a == "--naive-max")
            naive_max = std::stoi(next());
        else {
            std::fprintf(stderr, "usage: bench_gemm [--only sgemm|hgemm] [--sizes a,b] [--repeats "
                                 "r] [--naive-max n]\n");
            return 2;
        }
    }
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::fprintf(stderr, "# %s, sm_%d%d, %d SMs\n", p.name, p.major, p.minor,
                 p.multiProcessorCount);
    std::printf("kernel,dtype,n,repeat,ms,tflops,pct_of_cublas\n");

    using S = void (*)(int, int, int, const float*, const float*, float*);
    using H = void (*)(int, int, int, const __half*, const __half*, float*);
    const std::vector<std::pair<const char*, S>> sgemms{{"naive", sgemm_naive},
                                                        {"coalesced", sgemm_coalesced},
                                                        {"smem", sgemm_smem},
                                                        {"blocktile_1d", sgemm_blocktile_1d},
                                                        {"blocktile_2d", sgemm_blocktile_2d}};
    const std::vector<std::pair<const char*, H>> hgemms{{"wmma", hgemm_wmma}, {"mma", hgemm_mma}};
    for (int n : sizes) {
        if (only.empty() || only == "sgemm")
            run_family<float, S>("fp32", sgemms, sgemm_cublas, n, repeats, naive_max);
        if (only.empty() || only == "hgemm")
            run_family<__half, H>("fp16", hgemms, hgemm_cublas, n, repeats, naive_max);
    }
    return 0;
}
