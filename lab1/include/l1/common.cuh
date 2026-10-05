// Shared helpers for lab1: error checking, RAII device buffers, timing,
// deterministic inputs. Read once; you should not need to change it.
#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace l1 {

inline void check_cuda(cudaError_t code, const char* expr, const char* file, int line) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expr + ": " +
                                 cudaGetErrorString(code));
}
inline void check_cublas(cublasStatus_t code, const char* expr, const char* file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expr +
                                 ": cuBLAS status " + std::to_string(static_cast<int>(code)));
}
#define CUDA_CHECK(expr) ::l1::check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) ::l1::check_cublas((expr), #expr, __FILE__, __LINE__)
// Call right after a <<<launch>>>: catches bad launch configurations.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Thrown by every launcher you have not implemented yet.
struct NotImplemented : std::logic_error {
    using std::logic_error::logic_error;
};

// Owning device allocation. Non-copyable, movable.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : n_(n) { CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T))); }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) { o.ptr_ = nullptr; o.n_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        std::swap(ptr_, o.ptr_);
        std::swap(n_, o.n_);
        return *this;
    }
    T* get() const { return ptr_; }
    size_t size() const { return n_; }

    void upload(const std::vector<T>& host) {
        if (host.size() != n_) throw std::invalid_argument("upload: size mismatch");
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), n_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(n_);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
    // Fill every byte with 0xFF: for float that is a NaN, so any element a
    // kernel forgets to write shows up as NaN in the result.
    void poison() { CUDA_CHECK(cudaMemset(ptr_, 0xFF, n_ * sizeof(T))); }

private:
    T* ptr_ = nullptr;
    size_t n_ = 0;
};

// Deterministic values k/128 with k in [-128, 128]. They are exactly
// representable in both float and half, and so are their pairwise products,
// which keeps reference comparisons tight.
inline std::vector<float> make_input(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    for (auto& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>((seed >> 16) % 257u) - 128) / 128.0f;
    }
    return v;
}

inline std::vector<__half> to_half(const std::vector<float>& v) {
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    return h;
}

// Times `fn` with CUDA events. Runs `warmup` launches, then repeats the
// launch until at least `min_ms` has elapsed, and returns ms per launch.
template <class F>
float time_ms(F&& fn, int warmup = 3, float min_ms = 50.f) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    int iters = 1;
    float ms = 0.f;
    while (true) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms >= min_ms || iters >= (1 << 14)) break;
        iters *= 2;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace l1
