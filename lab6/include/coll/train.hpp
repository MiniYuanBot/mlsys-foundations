// Data-parallel training of a small MLP, for Part C. Provided except the
// classes marked "student".
//
// All parameters live in ONE flat float buffer, and gradients in another with
// the same layout, ordered the way BACKWARD produces them: the last layer
// first. So the gradients that are ready first form a contiguous prefix,
// which is what makes bucketing easy (PyTorch DDP orders its buckets the same way).
//
//   flat = [ W_{L-1}, b_{L-1}, W_{L-2}, b_{L-2}, ..., W_0, b_0 ]
#pragma once

#include "coll/coll.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace coll {

// ------------------------------------------------------------------ model ----
class MLP {
public:
    // dims = {in, hidden..., out}. Hidden layers use ReLU; the last layer is linear.
    MLP(std::vector<int> dims, uint64_t seed);

    int num_layers() const { return static_cast<int>(dims_.size()) - 1; }
    size_t num_params() const { return params.size(); }
    // Offset and size (W and b together) of layer l inside the flat buffers.
    size_t offset(int l) const { return offset_[l]; }
    size_t size(int l) const { return size_[l]; }

    // Mean-squared-error loss of a batch [B, in] against targets [B, out].
    float forward(const float* x, const float* y, int B);
    // Gradients of the last forward() into `grads` (overwritten). After layer
    // l's gradient is complete, calls on_ready(l); layers finish in the order
    // L-1, L-2, ..., 0.
    void backward(const std::function<void(int)>& on_ready = {});

    std::vector<float> params, grads;

private:
    std::vector<int> dims_;
    std::vector<size_t> offset_, size_;
    std::vector<std::vector<float>> act_; // activations of the last forward
    std::vector<float> y_;
    int B_ = 0;
};

// Deterministic synthetic regression data: sample s (a global index) is the
// same on every rank and every run. y = tanh(x A) for a fixed random A.
void make_batch(int in, int out, size_t first_sample, int B, std::vector<float>& x,
                std::vector<float>& y);

// ---------------------------------------------------------------- optimizers ----
struct Adam {
    float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
};
// Provided: Adam update of p[0..n) using g, m, v (all length n), step t >= 1.
void adam_update(float* p, const float* g, float* m, float* v, size_t n, int t, const Adam& h);
inline void sgd_update(float* p, const float* g, size_t n, float lr) {
    for (size_t i = 0; i < n; ++i) p[i] -= lr * g[i];
}

// ------------------------------------------------------------ comm thread ----
// A background thread that runs an allreduce on each submitted range, in
// order. The training thread keeps computing meanwhile (this is the overlap).
// Only this thread calls MPI while training is running.
class CommThread {
public:
    CommThread(const Comm& c, AllreduceFn fn);
    ~CommThread();
    void submit(float* data, size_t n); // enqueue an in-place allreduce
    void wait_all();                    // block until every submitted range is done
    int submitted() const { return submitted_; }
    double busy_seconds() const { return busy_; }

private:
    void loop();
    Comm c_;
    AllreduceFn fn_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::pair<float*, size_t>> queue_;
    int pending_ = 0, submitted_ = 0;
    bool stop_ = false;
    double busy_ = 0;
    // Declared LAST and started in the constructor body: members are
    // initialised in declaration order, and the thread must not run loop()
    // before mu_, cv_ and stop_ exist (this was a real intermittent crash).
    std::thread thread_;
};

// ------------------------------------------------------- student: C1 ----
// DDP-style gradient bucketing (see train_student.cpp).
class Bucketer {
public:
    Bucketer(MLP& model, size_t bucket_bytes, CommThread& comm, int world_size);
    void on_layer_ready(int layer); // called by MLP::backward
    void finish();                  // after backward: flush, wait, average
    int buckets() const { return buckets_; }

private:
    MLP& m_;
    size_t cap_floats_;
    CommThread& comm_;
    int P_;
    size_t start_ = 0,
           pending_ = 0; // [start_, start_ + pending_) of grads is ready but not submitted
    int expect_ = -1;    // next layer we expect to become ready
    int buckets_ = 0;
};

// ------------------------------------------------------- student: C2 ----
// ZeRO stage 1: each rank keeps Adam's m and v only for ITS chunk of the parameters.
class Zero1Adam {
public:
    Zero1Adam(size_t num_params, const Comm& c, Adam h);
    // grads: this rank's local gradient (overwritten). params: updated in place on every rank.
    void step(std::vector<float>& params, std::vector<float>& grads);
    size_t state_floats() const { return m_.size() + v_.size(); }

private:
    Comm c_;
    Adam h_;
    size_t begin_, end_;
    std::vector<float> m_, v_;
    int t_ = 0;
};

} // namespace coll
