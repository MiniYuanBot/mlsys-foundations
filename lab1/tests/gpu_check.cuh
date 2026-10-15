// Test helpers shared by the lab1 test programs.
#pragma once

#include "check.hpp"
#include "l1/kernels.cuh"

#include <cmath>
#include <sstream>

namespace l1test {

using l1::DeviceBuffer;

// CPU reference C = A @ B in double precision.
inline std::vector<double> gemm_ref(int M, int N, int K, const std::vector<float>& A,
                                    const std::vector<float>& B) {
    std::vector<double> C(static_cast<size_t>(M) * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(i) * K + k];
            for (int j = 0; j < N; ++j)
                C[static_cast<size_t>(i) * N + j] += a * B[static_cast<size_t>(k) * N + j];
        }
    return C;
}

// Every element finite and within tolerance; reports the first mismatch.
template <class Ref>
void expect_matrix_close(const char* what, int M, int N, const std::vector<float>& got,
                         const Ref& want, double atol, double rtol) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            const size_t idx = static_cast<size_t>(i) * N + j;
            const double g = got[idx], w = want[idx];
            if (!std::isfinite(g) || std::fabs(g - w) > atol + rtol * std::fabs(w)) {
                std::ostringstream os;
                os << what << " M=" << M << " N=" << N << ": C(" << i << "," << j << ") = " << g
                   << ", expected " << w
                   << (std::isnan(g) ? "  (NaN: this element was never written)" : "");
                throw check::Failure(os.str());
            }
        }
}

using SgemmFn = void (*)(int, int, int, const float*, const float*, float*);
using HgemmFn = void (*)(int, int, int, const __half*, const __half*, float*);

inline void run_sgemm_case(const char* name, SgemmFn fn, int M, int N, int K, unsigned seed) {
    const auto A = l1::make_input(static_cast<size_t>(M) * K, seed);
    const auto B = l1::make_input(static_cast<size_t>(K) * N, seed + 1000);
    DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(static_cast<size_t>(M) * N);
    dA.upload(A);
    dB.upload(B);
    dC.poison();
    fn(M, N, K, dA.get(), dB.get(), dC.get());
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_matrix_close(name, M, N, dC.download(), gemm_ref(M, N, K, A, B),
                        1e-4 * std::sqrt(K) + 1e-5, 1e-5);
    CHECK(dA.download() == A); // inputs must not be modified
    CHECK(dB.download() == B);
}

inline void run_hgemm_case(const char* name, HgemmFn fn, int M, int N, int K, unsigned seed) {
    const auto Af = l1::make_input(static_cast<size_t>(M) * K, seed);
    const auto Bf = l1::make_input(static_cast<size_t>(K) * N, seed + 1000);
    DeviceBuffer<__half> dA(Af.size()), dB(Bf.size());
    DeviceBuffer<float> dC(static_cast<size_t>(M) * N);
    dA.upload(l1::to_half(Af));
    dB.upload(l1::to_half(Bf));
    dC.poison();
    fn(M, N, K, dA.get(), dB.get(), dC.get());
    CUDA_CHECK(cudaDeviceSynchronize());
    // Inputs and products are exact in fp16; only fp32 accumulation rounds.
    expect_matrix_close(name, M, N, dC.download(), gemm_ref(M, N, K, Af, Bf),
                        1e-4 * std::sqrt(K) + 1e-5, 1e-5);
}

// Large shapes: compare against cuBLAS instead of the (slow) CPU reference.
inline void compare_with(const char* name, SgemmFn fn, SgemmFn ref, int n) {
    const auto A = l1::make_input(static_cast<size_t>(n) * n, 7),
               B = l1::make_input(static_cast<size_t>(n) * n, 8);
    DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(A.size()), dR(A.size());
    dA.upload(A);
    dB.upload(B);
    dC.poison();
    fn(n, n, n, dA.get(), dB.get(), dC.get());
    ref(n, n, n, dA.get(), dB.get(), dR.get());
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_matrix_close(name, n, n, dC.download(), dR.download(), 1e-3 * std::sqrt(n), 1e-4);
}

inline void compare_with(const char* name, HgemmFn fn, HgemmFn ref, int n) {
    const auto A = l1::to_half(l1::make_input(static_cast<size_t>(n) * n, 7));
    const auto B = l1::to_half(l1::make_input(static_cast<size_t>(n) * n, 8));
    DeviceBuffer<__half> dA(A.size()), dB(B.size());
    DeviceBuffer<float> dC(A.size()), dR(A.size());
    dA.upload(A);
    dB.upload(B);
    dC.poison();
    fn(n, n, n, dA.get(), dB.get(), dC.get());
    ref(n, n, n, dA.get(), dB.get(), dR.get());
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_matrix_close(name, n, n, dC.download(), dR.download(), 1e-3 * std::sqrt(n), 1e-4);
}

} // namespace l1test
