// Part C tests: Tensor Core kernels against a CPU double reference and cuBLAS.
#include "gpu_check.cuh"

using namespace l1;
using namespace l1test;

TEST(C0_cublas_hgemm_reference) {
    run_hgemm_case("cublas_hgemm", hgemm_cublas, 64, 64, 16, 1);
    run_hgemm_case("cublas_hgemm", hgemm_cublas, 128, 192, 96, 2);
}

TEST(C1_wmma) {
    run_hgemm_case("wmma", hgemm_wmma, 64, 64, 16, 11);   // one block, one k-step
    run_hgemm_case("wmma", hgemm_wmma, 128, 192, 48, 12); // several blocks and k-steps
    run_hgemm_case("wmma", hgemm_wmma, 64, 128, 512, 13);
    compare_with("wmma", hgemm_wmma, hgemm_cublas, 1024);
}

TEST(C1_wmma_rejects_bad_shape) {
    DeviceBuffer<__half> h(128 * 128);
    DeviceBuffer<float> f(128 * 128);
    CHECK_THROWS(hgemm_wmma(96, 64, 16, h.get(), h.get(), f.get()), std::invalid_argument);
    CHECK_THROWS(hgemm_wmma(64, 64, 8, h.get(), h.get(), f.get()), std::invalid_argument);
}

TEST(C2_mma_single_tile) {
    // K = 32: exactly one k-tile, so the pipeline prologue/epilogue edge cases matter.
    run_hgemm_case("mma", hgemm_mma, 128, 128, 32, 21);
}

TEST(C2_mma) {
    run_hgemm_case("mma", hgemm_mma, 128, 128, 64, 22);   // KT = 2 < stages
    run_hgemm_case("mma", hgemm_mma, 256, 384, 96, 23);   // KT = 3 = stages
    run_hgemm_case("mma", hgemm_mma, 128, 256, 1024, 24); // long pipeline
    compare_with("mma", hgemm_mma, hgemm_cublas, 1024);
    compare_with("mma", hgemm_mma, hgemm_cublas, 4096);
}

TEST(C2_mma_rejects_bad_shape) {
    DeviceBuffer<__half> h(256 * 256);
    DeviceBuffer<float> f(256 * 256);
    CHECK_THROWS(hgemm_mma(64, 128, 32, h.get(), h.get(), f.get()), std::invalid_argument);
    CHECK_THROWS(hgemm_mma(128, 128, 16, h.get(), h.get(), f.get()), std::invalid_argument);
}

int main(int argc, char **argv) {
    return run_tests(argc, argv);
}
