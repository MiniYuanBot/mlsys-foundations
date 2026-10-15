// Part B tests: each kernel against a CPU double reference, then against cuBLAS at n = 1024.
#include "gpu_check.cuh"

using namespace l1;
using namespace l1test;

// Any shape, including 1 x 1 and edges that are not multiples of 32.
static void sweep_any_shape(const char *name, SgemmFn fn) {
    unsigned seed = 1;
    const int shapes[][3] = {{1, 1, 1},    {7, 11, 5},   {32, 32, 32},
                             {33, 31, 65}, {64, 65, 33}, {129, 17, 200}};
    for (const auto &s : shapes) run_sgemm_case(name, fn, s[0], s[1], s[2], seed++);
}

TEST(B0_naive_reference) {
    sweep_any_shape("naive", sgemm_naive);
}
TEST(B0_cublas_reference) {
    sweep_any_shape("cublas", sgemm_cublas);
}

TEST(B1_coalesced) {
    sweep_any_shape("coalesced", sgemm_coalesced);
    compare_with("coalesced", sgemm_coalesced, sgemm_cublas, 1024);
}

TEST(B2_smem) {
    sweep_any_shape("smem", sgemm_smem);
    compare_with("smem", sgemm_smem, sgemm_cublas, 1024);
}

TEST(B3_blocktile_1d) {
    run_sgemm_case("blocktile_1d", sgemm_blocktile_1d, 64, 64, 8, 11);
    run_sgemm_case("blocktile_1d", sgemm_blocktile_1d, 128, 192, 24,
                   12); // several blocks, several k-steps
    run_sgemm_case("blocktile_1d", sgemm_blocktile_1d, 192, 64, 136, 13);
    compare_with("blocktile_1d", sgemm_blocktile_1d, sgemm_cublas, 1024);
}

TEST(B3_blocktile_1d_rejects_bad_shape) {
    DeviceBuffer<float> buf(65 * 65);
    CHECK_THROWS(sgemm_blocktile_1d(65, 64, 8, buf.get(), buf.get(), buf.get()),
                 std::invalid_argument);
    CHECK_THROWS(sgemm_blocktile_1d(64, 64, 12, buf.get(), buf.get(), buf.get()),
                 std::invalid_argument);
}

TEST(B4_blocktile_2d) {
    run_sgemm_case("blocktile_2d", sgemm_blocktile_2d, 128, 128, 8, 21);
    run_sgemm_case("blocktile_2d", sgemm_blocktile_2d, 256, 128, 64, 22);
    run_sgemm_case("blocktile_2d", sgemm_blocktile_2d, 128, 384, 520, 23);
    compare_with("blocktile_2d", sgemm_blocktile_2d, sgemm_cublas, 1024);
    compare_with("blocktile_2d", sgemm_blocktile_2d, sgemm_cublas, 2048);
}

TEST(B4_blocktile_2d_rejects_bad_shape) {
    DeviceBuffer<float> buf(256 * 256);
    CHECK_THROWS(sgemm_blocktile_2d(64, 128, 8, buf.get(), buf.get(), buf.get()),
                 std::invalid_argument);
    CHECK_THROWS(sgemm_blocktile_2d(128, 128, 4, buf.get(), buf.get(), buf.get()),
                 std::invalid_argument);
}

int main(int argc, char **argv) {
    return run_tests(argc, argv);
}
