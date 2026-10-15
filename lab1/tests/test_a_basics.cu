// Part A tests.  test_a_basics A2   runs only the reduction tests.
#include "gpu_check.cuh"

#include <numeric>

using namespace l1;
using namespace l1test;

TEST(A0_device_is_visible) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    CHECK(count > 0);
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("         device 0: %s, sm_%d%d, %d SMs, %.1f GiB\n", p.name, p.major, p.minor,
                p.multiProcessorCount, p.totalGlobalMem / double(1 << 30));
    CHECK(p.major * 10 + p.minor >= 80); // Part C needs sm_80+ (Ampere or newer)
}

TEST(A1_vector_add) {
    for (int n : {1, 31, 1000, (1 << 20) + 7, 1 << 24}) {
        const auto a = make_input(n, 1), b = make_input(n, 2);
        DeviceBuffer<float> da(n), db(n), dc(n);
        da.upload(a);
        db.upload(b);
        dc.poison();
        vector_add(da.get(), db.get(), dc.get(), n);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto c = dc.download();
        for (int i = 0; i < n; ++i)
            if (c[i] != a[i] + b[i]) {
                std::ostringstream os;
                os << "n=" << n << " c[" << i << "] = " << c[i] << ", expected " << a[i] + b[i];
                throw check::Failure(os.str());
            }
    }
}

TEST(A2_reduce_sum) {
    DeviceBuffer<float> partial(reduce_scratch_size()), out(1);
    for (int n : {1, 32, 33, 1000, 256 * 1024 + 5, 1 << 24}) {
        const auto x = make_input(n, 3);
        DeviceBuffer<float> dx(n);
        dx.upload(x);
        out.poison();
        reduce_sum(dx.get(), n, partial.get(), out.get());
        CUDA_CHECK(cudaDeviceSynchronize());
        double want = 0, mag = 0;
        for (float v : x) {
            want += v;
            mag += std::fabs(v);
        }
        const float got = out.download()[0];
        if (!(std::fabs(got - want) <= 1e-6 * mag + 1e-6)) {
            std::ostringstream os;
            os << "n=" << n << ": got " << got << ", expected " << want;
            throw check::Failure(os.str());
        }
    }
}

TEST(A2_reduce_sum_all_ones_exact) {
    // Every partial sum of ones is an exact integer: any lost or doubled element shows.
    const int n = (1 << 22) + 123;
    DeviceBuffer<float> dx(n), partial(reduce_scratch_size()), out(1);
    dx.upload(std::vector<float>(n, 1.0f));
    reduce_sum(dx.get(), n, partial.get(), out.get());
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_EQ(out.download()[0], static_cast<float>(n));
}

static void check_transpose(void (*fn)(const float *, float *, int, int), int M, int N) {
    const auto in = make_input(static_cast<size_t>(M) * N, 4);
    DeviceBuffer<float> din(in.size()), dout(in.size());
    din.upload(in);
    dout.poison();
    fn(din.get(), dout.get(), M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto out = dout.download();
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            if (out[static_cast<size_t>(c) * M + r] != in[static_cast<size_t>(r) * N + c]) {
                std::ostringstream os;
                os << "M=" << M << " N=" << N << ": out(" << c << "," << r << ") wrong";
                throw check::Failure(os.str());
            }
}

TEST(A3_transpose_naive_reference) {
    const int shapes[][2] = {{1, 1}, {33, 31}, {1000, 777}};
    for (const auto &s : shapes) check_transpose(transpose_naive, s[0], s[1]);
}
TEST(A3a_transpose_tiled) {
    const int shapes[][2] = {{1, 1}, {32, 32}, {33, 31}, {1000, 777}, {2048, 64}};
    for (const auto &s : shapes) check_transpose(transpose_tiled, s[0], s[1]);
}
TEST(A3b_transpose_padded) {
    const int shapes[][2] = {{1, 1}, {32, 32}, {33, 31}, {1000, 777}, {64, 2048}};
    for (const auto &s : shapes) check_transpose(transpose_padded, s[0], s[1]);
}

int main(int argc, char **argv) {
    return run_tests(argc, argv);
}
