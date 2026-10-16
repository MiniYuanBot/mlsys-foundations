// Part C tests: schedules are checked exactly, values against eager mt ops.
#include "check.hpp"
#include "mt/lazy.hpp"

#include <random>

using namespace mt;
using namespace mt::lazy;

namespace {
Tensor random(const Shape &s, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-2, 2);
    std::vector<float> v(numel_of(s));
    for (auto &x : v) x = u(rng);
    return Tensor::from_vector(s, v);
}
void expect_close(const Tensor &got, const Tensor &want, double tol = 1e-5) {
    CHECK_EQ(got.shape(), want.shape());
    const auto g = got.to_vector(), w = want.to_vector();
    for (size_t i = 0; i < g.size(); ++i) CHECK_NEAR(g[i], w[i], tol * (1 + std::fabs(w[i])));
}
int position(const std::vector<NodePtr> &order, const NodePtr &n) {
    for (size_t i = 0; i < order.size(); ++i)
        if (order[i] == n) return static_cast<int>(i);
    return -1;
}
constexpr int64_t F = sizeof(float);
} // namespace

// y = relu(x * w + b)
TEST(C1_toposort_order_and_uniqueness) {
    auto x = LazyTensor::input(random({8}, 1)), w = LazyTensor::input(random({8}, 2));
    auto d = x * w;
    auto y = (d + d).relu(); // d is used twice but must appear once
    auto order = toposort(y.node());
    CHECK_EQ(order.size(), 5u);
    CHECK(order.back() == y.node());
    CHECK(position(order, x.node()) < position(order, d.node()));
    CHECK(position(order, w.node()) < position(order, d.node()));
    CHECK(position(order, d.node()) < position(order, y.node()));
    // deterministic depth-first post-order: x, w, d, (d+d), relu
    CHECK(order[0] == x.node());
    CHECK(order[1] == w.node());
}

TEST(C1_unfused_schedule_counts) {
    const int64_t N = 1000;
    auto x = LazyTensor::input(random({N}, 1)), w = LazyTensor::input(random({N}, 2)),
         b = LazyTensor::input(random({N}, 3));
    auto y = (x * w + b).relu();
    auto st = analyze(schedule_unfused(y.node()), y.node());
    CHECK_EQ(st.kernels, 3);
    CHECK_EQ(st.bytes_written, 3 * N * F);
    CHECK_EQ(st.bytes_read, 5 * N * F); // mul: x,w   add: t1,b   relu: t2
    CHECK_EQ(st.intermediate_bytes, 2 * N * F);
}

TEST(C2_fused_chain_is_one_kernel) {
    const int64_t N = 1000;
    auto x = LazyTensor::input(random({N}, 1)), w = LazyTensor::input(random({N}, 2)),
         b = LazyTensor::input(random({N}, 3));
    auto y = (x * w + b).relu();
    auto ks = schedule_fused(y.node());
    CHECK_EQ(ks.size(), 1u);
    CHECK_EQ(ks[0].nodes.size(), 3u);
    CHECK(ks[0].nodes.back() == y.node());
    auto st = analyze(ks, y.node());
    CHECK_EQ(st.bytes_written, N * F);
    CHECK_EQ(st.bytes_read, 3 * N * F);
    CHECK_EQ(st.intermediate_bytes, 0);
}

TEST(C2_shared_node_is_materialised) {
    auto x = LazyTensor::input(random({64}, 4)), w = LazyTensor::input(random({64}, 5));
    auto d = x * w;
    auto y = (d + LazyTensor::constant(1.f)) * (d * LazyTensor::constant(2.f));
    auto ks = schedule_fused(y.node());
    CHECK_EQ(ks.size(), 2u);
    CHECK(ks[0].output == d.node());
    CHECK(ks[1].output == y.node());
    CHECK_EQ(ks[1].nodes.size(), 3u); // add, mul-by-2, final mul
}

TEST(C2_square_counts_two_uses) {
    auto x = LazyTensor::input(random({64}, 6));
    auto t = x.exp();
    auto y = t * t;
    CHECK_EQ(schedule_fused(y.node()).size(), 2u);
}

TEST(C2_reduction_fuses_its_producer) {
    auto x = LazyTensor::input(random({4, 32}, 7)), w = LazyTensor::input(random({4, 32}, 8));
    auto s = (x * w).exp().sum_last();
    auto ks = schedule_fused(s.node());
    CHECK_EQ(ks.size(), 1u);
    CHECK_EQ(ks[0].nodes.size(), 3u);
    // relu after the reduction needs a second kernel (reduction output is materialised)
    auto r = s.relu();
    CHECK_EQ(schedule_fused(r.node()).size(), 2u);
    CHECK_EQ(schedule_unfused(r.node()).size(), 4u);
}

TEST(C3_compile_postfix) {
    auto x = LazyTensor::input(random({8}, 1)), w = LazyTensor::input(random({8}, 2)),
         b = LazyTensor::input(random({8}, 3));
    auto y = (x * w + b).relu();
    Program p = compile(schedule_fused(y.node())[0]);
    CHECK_EQ(p.buffers.size(), 3u);
    CHECK(p.buffers[0] == x.node() && p.buffers[1] == w.node() && p.buffers[2] == b.node());
    CHECK_EQ(p.code.size(), 6u);
    CHECK(p.code[0].kind == InstrKind::Load && p.code[0].buffer == 0);
    CHECK(p.code[1].kind == InstrKind::Load && p.code[1].buffer == 1);
    CHECK(p.code[2].kind == InstrKind::Binary && p.code[2].op == Op::Mul);
    CHECK(p.code[3].kind == InstrKind::Load && p.code[3].buffer == 2);
    CHECK(p.code[4].kind == InstrKind::Binary && p.code[4].op == Op::Add);
    CHECK(p.code[5].kind == InstrKind::Unary && p.code[5].op == Op::Relu);
    CHECK(!p.reduce);
}

TEST(C3_compile_reuses_buffer_and_inlines_const) {
    auto x = LazyTensor::input(random({8}, 1));
    auto y = (x * x + 3.0f * x.exp()).tanh();
    Program p = compile(schedule_fused(y.node())[0]);
    CHECK_EQ(p.buffers.size(), 1u);
    int consts = 0;
    for (auto &ins : p.code) consts += ins.kind == InstrKind::Const;
    CHECK_EQ(consts, 1);
}

TEST(C3_compile_reduction) {
    auto x = LazyTensor::input(random({3, 10}, 9));
    auto s = x.exp().sum_last();
    Program p = compile(schedule_fused(s.node())[0]);
    CHECK(p.reduce);
    CHECK_EQ(p.reduce_len, 10);
    CHECK_EQ(p.code.size(), 2u); // Load, Exp
}

TEST(C3_fused_equals_unfused_equals_eager) {
    Tensor xt = random({5, 37}, 11), wt = random({5, 37}, 12);
    auto x = LazyTensor::input(xt), w = LazyTensor::input(wt);
    auto h = (x * w + LazyTensor::constant(0.5f));
    auto y = (h.tanh() * h - x.neg().exp() / (w * w + 1.0f)).relu();
    Tensor eager = map((xt * wt) + Tensor::full({}, 0.5f), [](float v) { return std::tanh(v); });
    eager = eager * ((xt * wt) + Tensor::full({}, 0.5f));
    Tensor rhs = map(xt, [](float v) { return std::exp(-v); }) / (wt * wt + Tensor::full({}, 1.0f));
    eager = map(eager - rhs, [](float v) { return v > 0 ? v : 0.f; });
    expect_close(realize_unfused(y), eager);
    expect_close(realize_fused(y), eager);
}

TEST(C3_reduction_values) {
    Tensor xt = random({6, 33}, 13);
    auto s = (LazyTensor::input(xt).exp().sum_last() + 1.0f).relu();
    Tensor want = map(sum(map(xt, [](float v) { return std::exp(v); }), -1) + Tensor::full({}, 1.f),
                      [](float v) { return v > 0 ? v : 0.f; });
    expect_close(realize_fused(s), want);
    expect_close(realize_unfused(s), want);
}

TEST(C4_tiled_matches_scalar) {
    // sizes around the tile boundary, plus a reduction longer than one tile
    for (int64_t n : {1, 7, kTile - 1, kTile, kTile + 1, 3 * kTile + 5}) {
        Tensor xt = random({n}, 20 + static_cast<unsigned>(n)), wt = random({n}, 30);
        auto x = LazyTensor::input(xt), w = LazyTensor::input(wt);
        auto y = ((x - w) * (x + 2.0f) / (w.exp() + 1.0f)).neg().relu().tanh();
        expect_close(realize_fused(y, Interp::Tiled), realize_fused(y, Interp::Scalar), 1e-6);
    }
    Tensor xt = random({3, 2 * kTile + 3}, 40);
    auto s = (LazyTensor::input(xt) * 0.5f).exp().sum_last();
    expect_close(realize_fused(s, Interp::Tiled), realize_fused(s, Interp::Scalar), 1e-5);
}

int main(int argc, char **argv) {
    return run_tests(argc, argv);
}
