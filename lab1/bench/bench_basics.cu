// Memory-bound kernels: report effective bandwidth and compare it with a
// device-to-device cudaMemcpy, the practical ceiling for streaming kernels.
//
//   bench_basics [--n 67108864] [--side 8192] > results/lab1/basics.csv
//
// bytes = bytes the kernel must read + write at minimum (e.g. 12n for c = a + b).
#include "l1/kernels.cuh"

#include <cstdio>
#include <functional>
#include <string>

using namespace l1;

int main(int argc, char** argv) {
    int n = 1 << 26, side = 8192;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n" && i + 1 < argc)
            n = std::stoi(argv[++i]);
        else if (a == "--side" && i + 1 < argc)
            side = std::stoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: bench_basics [--n N] [--side S]\n");
            return 2;
        }
    }
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::fprintf(stderr, "# %s, %d SMs\n", p.name, p.multiProcessorCount);

    DeviceBuffer<float> a(n), b(n), c(n), partial(reduce_scratch_size()), out(1);
    a.upload(make_input(n, 1));
    b.upload(make_input(n, 2));
    const double fbytes = sizeof(float);

    std::printf("op,variant,size,bytes,ms,GBps,pct_of_memcpy\n");
    double memcpy_gbps = 0;
    auto report = [&](const char* op, const char* variant, long long size, double bytes,
                      const std::function<void()>& fn) {
        float ms;
        try {
            ms = time_ms(fn);
        } catch (const NotImplemented& e) {
            std::fprintf(stderr, "# skip %s/%s: %s\n", op, variant, e.what());
            return;
        }
        const double gbps = bytes / (ms * 1e6);
        if (memcpy_gbps == 0) memcpy_gbps = gbps;
        std::printf("%s,%s,%lld,%.0f,%.4f,%.1f,%.1f\n", op, variant, size, bytes, ms, gbps,
                    100 * gbps / memcpy_gbps);
        std::fflush(stdout);
    };

    report("copy", "cudaMemcpy", n, 2 * n * fbytes, [&] {
        CUDA_CHECK(cudaMemcpyAsync(c.get(), a.get(), n * fbytes, cudaMemcpyDeviceToDevice));
    });
    report("vector_add", "grid_stride", n, 3 * n * fbytes,
           [&] { vector_add(a.get(), b.get(), c.get(), n); });
    report("reduce_sum", "shuffle", n, n * fbytes,
           [&] { reduce_sum(a.get(), n, partial.get(), out.get()); });

    const long long elems = static_cast<long long>(side) * side;
    if (elems <= n) {
        report("transpose", "naive", elems, 2 * elems * fbytes,
               [&] { transpose_naive(a.get(), c.get(), side, side); });
        report("transpose", "tiled", elems, 2 * elems * fbytes,
               [&] { transpose_tiled(a.get(), c.get(), side, side); });
        report("transpose", "padded", elems, 2 * elems * fbytes,
               [&] { transpose_padded(a.get(), c.get(), side, side); });
    }
    return 0;
}
