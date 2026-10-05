// Part B tests: every kernel against a double-precision reference.
#include "check.hpp"
#include "mt/gemm.hpp"
#include "mt/tensor.hpp"

#include <limits>
#include <random>
#include <vector>

using namespace mt;

namespace {

struct Problem {
    int M, N, K, lda, ldb, ldc;
    std::vector<float> A, B;
};

Problem make(int M, int N, int K, int pad, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1, 1);
    Problem p{M, N, K, K + pad, N + pad, N + pad, {}, {}};
    p.A.resize(static_cast<size_t>(M) * p.lda);
    p.B.resize(static_cast<size_t>(K) * p.ldb);
    for (auto& x : p.A) x = u(rng);
    for (auto& x : p.B) x = u(rng);
    return p;
}

std::vector<double> reference(const Problem& p) {
    std::vector<double> c(static_cast<size_t>(p.M) * p.N, 0.0);
    for (int i = 0; i < p.M; ++i)
        for (int k = 0; k < p.K; ++k)
            for (int j = 0; j < p.N; ++j)
                c[i * p.N + j] += static_cast<double>(p.A[i * p.lda + k]) * p.B[k * p.ldb + j];
    return c;
}

// Runs one kernel; C starts as NaN so any element that is not overwritten fails.
void expect_correct(GemmKernel kernel, const Problem& p) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> C(static_cast<size_t>(std::max(p.M, 1)) * p.ldc, nan);
    const auto A0 = p.A, B0 = p.B;
    gemm(kernel, p.M, p.N, p.K, p.A.data(), p.lda, p.B.data(), p.ldb, C.data(), p.ldc);
    const auto ref = reference(p);
    for (int i = 0; i < p.M; ++i) {
        for (int j = 0; j < p.N; ++j) {
            const double want = ref[i * p.N + j], got = C[i * p.ldc + j];
            if (!(std::fabs(got - want) <= 1e-4 * (1.0 + std::sqrt(static_cast<double>(p.K))))) {
                std::ostringstream os;
                os << kernel_name(kernel) << " M=" << p.M << " N=" << p.N << " K=" << p.K << " C(" << i << ","
                   << j << ") = " << got << ", expected " << want;
                throw check::Failure(os.str());
            }
        }
        // padding columns of C must be left alone
        for (int j = p.N; j < p.ldc; ++j)
            if (!std::isnan(C[i * p.ldc + j])) throw check::Failure("kernel wrote into the padding of C");
    }
    CHECK(p.A == A0);
    CHECK(p.B == B0);
}

void sweep(GemmKernel kernel) {
    unsigned seed = 1;
    // Small, odd, and not multiples of MR/NR/MC/KC: edge handling matters.
    for (int M : {1, 5, 6, 7, 13, 97})
        for (int N : {1, 15, 16, 17, 33})
            for (int K : {1, 3, 17, 64})
                expect_correct(kernel, make(M, N, K, 0, seed++));
    expect_correct(kernel, make(31, 29, 23, 3, seed++));      // padded leading dims
    expect_correct(kernel, make(130, 70, 300, 1, seed++));    // crosses MC and KC blocks
    expect_correct(kernel, make(7, 5, 0, 0, seed++));         // K = 0 -> C must be all zeros
}

}  // namespace

TEST(B0_naive_reference_sanity) { sweep(GemmKernel::Naive); }
TEST(B1_reordered) { sweep(GemmKernel::Reordered); }
TEST(B2_blocked) { sweep(GemmKernel::Blocked); }

TEST(B3a_pack_A_layout) {
    // 7 x 2 block, MR = 6: two panels, the second has 1 real row + 5 zero rows.
    std::vector<float> A(7 * 3);
    for (int i = 0; i < 7; ++i)
        for (int k = 0; k < 3; ++k) A[i * 3 + k] = 10.0f * i + k;  // lda = 3, use first 2 cols
    std::vector<float> Ap(2 * MR * 2, -1.0f);
    detail::pack_A(7, 2, A.data(), 3, Ap.data());
    for (int p = 0; p < 2; ++p)
        for (int k = 0; k < 2; ++k)
            for (int r = 0; r < MR; ++r) {
                const int row = p * MR + r;
                const float want = row < 7 ? 10.0f * row + k : 0.0f;
                CHECK_EQ(Ap[p * 2 * MR + k * MR + r], want);
            }
}

TEST(B3b_pack_B_layout) {
    const int kc = 3, nc = NR + 2;
    std::vector<float> B(kc * (nc + 1));
    for (int k = 0; k < kc; ++k)
        for (int c = 0; c < nc + 1; ++c) B[k * (nc + 1) + c] = 100.0f * k + c;
    std::vector<float> Bp(2 * NR * kc, -1.0f);
    detail::pack_B(kc, nc, B.data(), nc + 1, Bp.data());
    for (int q = 0; q < 2; ++q)
        for (int k = 0; k < kc; ++k)
            for (int c = 0; c < NR; ++c) {
                const int col = q * NR + c;
                const float want = col < nc ? 100.0f * k + col : 0.0f;
                CHECK_EQ(Bp[q * kc * NR + k * NR + c], want);
            }
}

TEST(B3c_micro_kernel_accumulates_and_respects_edges) {
    const int kc = 4;
    std::vector<float> Ap(kc * MR), Bp(kc * NR);
    for (int k = 0; k < kc; ++k) {
        for (int r = 0; r < MR; ++r) Ap[k * MR + r] = static_cast<float>(r + 1);
        for (int c = 0; c < NR; ++c) Bp[k * NR + c] = static_cast<float>(c);
    }
    const int ldc = NR + 1;
    std::vector<float> C(MR * ldc, 1.0f);
    detail::micro_kernel(kc, Ap.data(), Bp.data(), C.data(), ldc, MR - 1, NR - 2);
    for (int r = 0; r < MR; ++r)
        for (int c = 0; c < ldc; ++c) {
            const bool written = r < MR - 1 && c < NR - 2;
            const float want = written ? 1.0f + kc * (r + 1.0f) * c : 1.0f;
            CHECK_EQ(C[r * ldc + c], want);
        }
}

TEST(B3_packed) { sweep(GemmKernel::Packed); }

TEST(B4_parallel) {
    sweep(GemmKernel::Parallel);
    // explicit thread counts, including more threads than row tiles
    for (int threads : {1, 2, 3, 64}) {
        Problem p = make(50, 40, 30, 2, 99 + threads);
        std::vector<float> C(p.M * p.ldc);
        gemm_parallel(p.M, p.N, p.K, p.A.data(), p.lda, p.B.data(), p.ldb, C.data(), p.ldc, threads);
        const auto ref = reference(p);
        for (int i = 0; i < p.M; ++i)
            for (int j = 0; j < p.N; ++j) CHECK_NEAR(C[i * p.ldc + j], ref[i * p.N + j], 1e-3);
    }
}

TEST(B_capstone_tensor_matmul_on_views) {
    // mt::matmul makes operands contiguous, then calls the Best kernel.
    Tensor a = Tensor::arange(6).view({2, 3});
    Tensor b = Tensor::arange(6).view({2, 3}).transpose(0, 1);  // 3x2 view
    Tensor c = matmul(a, b);
    CHECK_EQ(c.to_vector(), (std::vector<float>{5, 14, 14, 50}));
}

int main(int argc, char** argv) { return run_tests(argc, argv); }
