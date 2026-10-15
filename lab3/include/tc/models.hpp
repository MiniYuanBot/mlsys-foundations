// Small graphs used by the tests and benchmarks (provided).
#pragma once

#include "tc/ir.hpp"

namespace tc::models {

// y = softmax(x) over rows.
inline Graph softmax(int64_t R, int64_t C) {
    Graph g;
    int x = g.input("x", {R, C});
    int e = g.exp(g.sub(x, g.row_max(x)));
    g.set_output(g.div(e, g.row_sum(e)));
    return g;
}

// y = (x - mean) / sqrt(var + eps) * gamma + beta, over rows.
inline Graph layernorm(int64_t R, int64_t C, double eps = 1e-5) {
    Graph g;
    int x = g.input("x", {R, C});
    int gamma = g.input("gamma", {1, C});
    int beta = g.input("beta", {1, C});
    int inv_c = g.constant(1.0 / static_cast<double>(C));
    int d = g.sub(x, g.mul(g.row_sum(x), inv_c));
    int var = g.mul(g.row_sum(g.mul(d, d)), inv_c);
    int y = g.mul(g.mul(d, g.rsqrt(g.add(var, g.constant(eps)))), gamma);
    g.set_output(g.add(y, beta));
    return g;
}

// y = x * rsqrt(mean(x^2) + eps) * gamma
inline Graph rmsnorm(int64_t R, int64_t C, double eps = 1e-6) {
    Graph g;
    int x = g.input("x", {R, C});
    int gamma = g.input("gamma", {1, C});
    int r = g.rsqrt(g.add(g.mul(g.row_sum(g.mul(x, x)), g.constant(1.0 / static_cast<double>(C))),
                          g.constant(eps)));
    g.set_output(g.mul(g.mul(x, r), gamma));
    return g;
}

// y = gelu_tanh(x @ W + b)   (an MLP up-projection)
inline Graph matmul_bias_gelu(int64_t M, int64_t K, int64_t N) {
    Graph g;
    int x = g.input("x", {M, K});
    int w = g.input("W", {K, N});
    int b = g.input("b", {1, N});
    int h = g.add(g.matmul(x, w), b);
    int h3 = g.mul(g.mul(h, h), h);
    int inner = g.mul(g.add(h, g.mul(h3, g.constant(0.044715))), g.constant(0.7978845608028654));
    g.set_output(g.mul(g.mul(h, g.constant(0.5)), g.add(g.tanh(inner), g.constant(1.0))));
    return g;
}

// y = rmsnorm(x) @ W + residual. Written the way a model file would write it:
// normalise first, then project. Part C finds a better form.
inline Graph rmsnorm_matmul(int64_t M, int64_t K, int64_t N, double eps = 1e-6) {
    Graph g;
    int x = g.input("x", {M, K});
    int gamma = g.input("gamma", {1, K});
    int w = g.input("W", {K, N});
    int res = g.input("res", {M, N});
    int r = g.rsqrt(g.add(g.mul(g.row_sum(g.mul(x, x)), g.constant(1.0 / static_cast<double>(K))),
                          g.constant(eps)));
    int xn = g.mul(g.mul(x, r), gamma);
    g.set_output(g.add(g.matmul(xn, w), res));
    return g;
}

// y = softmax(x) @ V : the shape of attention, with x the [M, K] score matrix.
// A rule-based fuser must write e = exp(x - m) to memory, because both the row
// sum and the matmul need it. Part C rediscovers the FlashAttention trick.
inline Graph softmax_matmul(int64_t M, int64_t K, int64_t N) {
    Graph g;
    int x = g.input("x", {M, K});
    int v = g.input("V", {K, N});
    int e = g.exp(g.sub(x, g.row_max(x)));
    g.set_output(g.matmul(g.div(e, g.row_sum(e)), v));
    return g;
}

} // namespace tc::models
