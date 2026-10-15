// Provided: the MLP, synthetic data, Adam, and the communication thread.
#include "coll/train.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace coll {

namespace {
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9e3779b97f4a7c15ull | 1) {}
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uniform() {
        return static_cast<float>((next() >> 40) / double(1ull << 24)) * 2.f - 1.f;
    } // [-1, 1)
};

// C[M,N] (+)= op(A)[M,K] @ B[K,N], row-major, op = transpose if ta. i-k-j loops
// so the inner loop vectorises (lab0 Part B1).
void gemm(bool ta, int M, int N, int K, const float* A, const float* B, float* C, bool accumulate) {
    if (!accumulate) std::fill(C, C + static_cast<size_t>(M) * N, 0.f);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            const float a =
                ta ? A[static_cast<size_t>(k) * M + i] : A[static_cast<size_t>(i) * K + k];
            if (a == 0.f) continue;
            float* c = C + static_cast<size_t>(i) * N;
            const float* b = B + static_cast<size_t>(k) * N;
            for (int j = 0; j < N; ++j) c[j] += a * b[j];
        }
}

// C[M,N] = A[M,K] @ B[N,K]^T: rows of A dotted with rows of B, both contiguous.
void gemm_nt(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            const float* a = A + static_cast<size_t>(i) * K;
            const float* b = B + static_cast<size_t>(j) * K;
            float s = 0.f;
#pragma omp simd reduction(+ : s)
            for (int k = 0; k < K; ++k) s += a[k] * b[k];
            C[static_cast<size_t>(i) * N + j] = s;
        }
}
} // namespace

MLP::MLP(std::vector<int> dims, uint64_t seed) : dims_(std::move(dims)) {
    const int L = num_layers();
    offset_.resize(L);
    size_.resize(L);
    size_t off = 0;
    for (int l = L - 1; l >= 0; --l) { // backward order
        offset_[l] = off;
        size_[l] = static_cast<size_t>(dims_[l + 1]) * dims_[l] + dims_[l + 1];
        off += size_[l];
    }
    params.assign(off, 0.f);
    grads.assign(off, 0.f);
    Rng rng(seed);
    for (int l = 0; l < L; ++l) {
        const float scale = std::sqrt(2.f / dims_[l]);
        float* w = params.data() + offset_[l];
        for (size_t i = 0; i < static_cast<size_t>(dims_[l + 1]) * dims_[l]; ++i)
            w[i] = rng.uniform() * scale;
    }
}

float MLP::forward(const float* x, const float* y, int B) {
    const int L = num_layers();
    B_ = B;
    act_.assign(L + 1, {});
    act_[0].assign(x, x + static_cast<size_t>(B) * dims_[0]);
    for (int l = 0; l < L; ++l) {
        const int in = dims_[l], out = dims_[l + 1];
        const float* W = params.data() + offset_[l]; // [out, in]
        const float* b = W + static_cast<size_t>(out) * in;
        act_[l + 1].resize(static_cast<size_t>(B) * out);
        gemm_nt(B, out, in, act_[l].data(), W, act_[l + 1].data());
        for (int i = 0; i < B; ++i)
            for (int j = 0; j < out; ++j) {
                float& v = act_[l + 1][static_cast<size_t>(i) * out + j];
                v += b[j];
                if (l + 1 < L) v = std::max(v, 0.f);
            }
    }
    y_.assign(y, y + static_cast<size_t>(B) * dims_[L]);
    double loss = 0;
    for (size_t i = 0; i < y_.size(); ++i)
        loss += 0.5 * double(act_[L][i] - y_[i]) * (act_[L][i] - y_[i]);
    return static_cast<float>(loss / B);
}

void MLP::backward(const std::function<void(int)>& on_ready) {
    const int L = num_layers(), B = B_;
    std::vector<float> delta(act_[L].size());
    for (size_t i = 0; i < delta.size(); ++i) delta[i] = (act_[L][i] - y_[i]) / B; // d loss / d out
    for (int l = L - 1; l >= 0; --l) {
        const int in = dims_[l], out = dims_[l + 1];
        float* gW = grads.data() + offset_[l];
        float* gb = gW + static_cast<size_t>(out) * in;
        gemm(true, out, in, B, delta.data(), act_[l].data(), gW, false); // delta^T @ act
        for (int j = 0; j < out; ++j) gb[j] = 0.f;
        for (int i = 0; i < B; ++i)
            for (int j = 0; j < out; ++j) gb[j] += delta[static_cast<size_t>(i) * out + j];
        if (l > 0) {
            const float* W = params.data() + offset_[l];
            std::vector<float> prev(static_cast<size_t>(B) * in);
            gemm(false, B, in, out, delta.data(), W, prev.data(), false); // delta @ W
            for (size_t i = 0; i < prev.size(); ++i)
                if (act_[l][i] <= 0.f) prev[i] = 0.f; // ReLU'
            delta.swap(prev);
        }
        if (on_ready) on_ready(l);
    }
}

void make_batch(int in, int out, size_t first_sample, int B, std::vector<float>& x,
                std::vector<float>& y) {
    static std::vector<float> A; // teacher [in, out], fixed
    if (A.size() != static_cast<size_t>(in) * out) {
        Rng r(12345);
        A.resize(static_cast<size_t>(in) * out);
        for (auto& a : A) a = r.uniform() / std::sqrt(static_cast<float>(in));
    }
    x.resize(static_cast<size_t>(B) * in);
    y.assign(static_cast<size_t>(B) * out, 0.f);
    for (int i = 0; i < B; ++i) {
        Rng r(1000003ull * (first_sample + i) + 7);
        float* xi = x.data() + static_cast<size_t>(i) * in;
        for (int k = 0; k < in; ++k) xi[k] = r.uniform();
        for (int k = 0; k < in; ++k)
            for (int j = 0; j < out; ++j)
                y[static_cast<size_t>(i) * out + j] += xi[k] * A[static_cast<size_t>(k) * out + j];
        for (int j = 0; j < out; ++j)
            y[static_cast<size_t>(i) * out + j] = std::tanh(y[static_cast<size_t>(i) * out + j]);
    }
}

void adam_update(float* p, const float* g, float* m, float* v, size_t n, int t, const Adam& h) {
    const float c1 = 1.f - std::pow(h.b1, static_cast<float>(t)),
                c2 = 1.f - std::pow(h.b2, static_cast<float>(t));
    for (size_t i = 0; i < n; ++i) {
        m[i] = h.b1 * m[i] + (1 - h.b1) * g[i];
        v[i] = h.b2 * v[i] + (1 - h.b2) * g[i] * g[i];
        p[i] -= h.lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + h.eps);
    }
}

CommThread::CommThread(const Comm& c, AllreduceFn fn) : c_(c), fn_(fn) {
    thread_ = std::thread([this] { loop(); });
}

CommThread::~CommThread() {
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
}

void CommThread::submit(float* data, size_t n) {
    {
        std::lock_guard<std::mutex> g(mu_);
        queue_.emplace_back(data, n);
        ++pending_;
        ++submitted_;
    }
    cv_.notify_all();
}

void CommThread::wait_all() {
    std::unique_lock<std::mutex> g(mu_);
    cv_.wait(g, [this] { return pending_ == 0; });
}

void CommThread::loop() {
    while (true) {
        std::pair<float*, size_t> job;
        {
            std::unique_lock<std::mutex> g(mu_);
            cv_.wait(g, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;
            job = queue_.front();
            queue_.pop_front();
        }
        const double t0 = MPI_Wtime();
        fn_(job.first, job.second, c_);
        const double dt = MPI_Wtime() - t0;
        {
            std::lock_guard<std::mutex> g(mu_);
            busy_ += dt;
            --pending_;
        }
        cv_.notify_all();
    }
}

} // namespace coll
