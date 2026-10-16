// Part A tests: toposort, interpreter, elementwise codegen.   test_a A2
#include "helpers.hpp"
#include "tc/models.hpp"

using namespace tc;
using namespace tc::test;

TEST(A1_toposort_postorder_and_unique) {
    Graph g;
    int x = g.input("x", {2, 3}), y = g.input("y", {2, 3});
    int d = g.mul(x, y);
    int o = g.add(d, d);
    int r = g.relu(o);
    g.set_output(r);
    CHECK_EQ(toposort(g), (std::vector<int>{x, y, d, o, r}));
}

TEST(A1_toposort_skips_unreachable) {
    Graph g;
    int x = g.input("x", {2, 2});
    g.exp(x); // dead
    g.set_output(g.neg(x));
    CHECK_EQ(toposort(g).size(), 2u);
}

TEST(A2_elementwise_and_broadcast) {
    Graph g;
    int x = g.input("x", {2, 3}), row = g.input("row", {1, 3}), col = g.input("col", {2, 1});
    g.set_output(g.sub(g.mul(x, row), g.max(col, g.constant(0.5))));
    Feeds f{{"x", {{2, 3}, {1, 2, 3, 4, 5, 6}}},
            {"row", {{1, 3}, {10, 20, 30}}},
            {"col", {{2, 1}, {0, 1}}}};
    CHECK_EQ(evaluate(g, f).data, (std::vector<double>{9.5, 39.5, 89.5, 39, 99, 179}));
}

TEST(A2_unary_reductions_matmul) {
    Graph g;
    int a = g.input("a", {2, 2}), b = g.input("b", {2, 3});
    int mm = g.matmul(a, b);
    g.set_output(g.add(g.row_sum(mm), g.row_max(g.neg(mm))));
    Feeds f{{"a", {{2, 2}, {1, 2, 3, 4}}}, {"b", {{2, 3}, {1, 0, 2, 0, 1, 1}}}};
    // mm = [[1,2,4],[3,4,10]]; rowsum = [7,17]; rowmax(-mm) = [-1,-3]
    CHECK_EQ(evaluate(g, f).data, (std::vector<double>{6, 14}));
    Graph h;
    int z = h.input("z", {1, 3});
    h.set_output(h.add(h.add(h.exp(z), h.tanh(z)),
                       h.add(h.relu(z), h.rsqrt(h.add(h.mul(z, z), h.constant(1))))));
    Feeds fz{{"z", {{1, 3}, {-1, 0, 2}}}};
    const auto v = evaluate(h, fz).data;
    CHECK_NEAR(v[0], std::exp(-1.0) + std::tanh(-1.0) + 0 + 1 / std::sqrt(2.0), 1e-12);
    CHECK_NEAR(v[2], std::exp(2.0) + std::tanh(2.0) + 2 + 1 / std::sqrt(5.0), 1e-12);
}

TEST(A2_softmax_rows_sum_to_one) {
    Graph g = models::softmax(5, 7);
    const HostTensor y = evaluate(g, random_feeds(g));
    for (int r = 0; r < 5; ++r) {
        double s = 0;
        for (int c = 0; c < 7; ++c) s += y.at(r, c);
        CHECK_NEAR(s, 1.0, 1e-12);
    }
}

TEST(A2_rejects_bad_feed) {
    Graph g;
    g.set_output(g.exp(g.input("x", {2, 2})));
    CHECK_THROWS(evaluate(g, Feeds{{"x", {{3, 2}, std::vector<double>(6)}}}),
                 std::invalid_argument);
    CHECK_THROWS(evaluate(g, Feeds{}), std::invalid_argument);
}

TEST(A3a_emit_op) {
    CHECK_EQ(emit_op(Op::Add, {"a", "b"}), std::string("(a + b)"));
    CHECK_EQ(emit_op(Op::Exp, {"x"}), std::string("expf(x)"));
    CHECK_EQ(emit_op(Op::Relu, {"x"}), std::string("fmaxf(x, 0.0f)"));
    CHECK_EQ(emit_op(Op::Max, {"a", "b"}), std::string("fmaxf(a, b)"));
}

TEST(A3b_emit_load) {
    CHECK_EQ(emit_load("in0", {4, 64}), std::string("in0[i * 64 + j]"));
    CHECK_EQ(emit_load("in1", {1, 64}), std::string("in1[j]"));
    CHECK_EQ(emit_load("in2", {4, 1}), std::string("in2[i]"));
    CHECK_EQ(emit_load("in3", {1, 1}), std::string("in3[0]"));
}

TEST(A3_unfused_plans_run_on_gpu) {
    // Every node is its own kernel, so this exercises the elementwise driver
    // with every op and every broadcast pattern (no Row/MatMul kernels).
    Runtime rt;
    Graph g;
    int x = g.input("x", {37, 50}), row = g.input("row", {1, 50}), col = g.input("col", {37, 1});
    int t = g.mul(g.add(x, row), g.exp(col));
    t = g.div(g.sub(g.tanh(t), g.neg(row)), g.add(g.relu(col), g.constant(2.0)));
    g.set_output(g.max(t, g.rsqrt(g.add(g.mul(x, x), g.constant(1.0)))));
    Plan p = plan_unfused(g);
    for (auto &k : p.kernels) CHECK(k.kind == KernelKind::Elementwise);
    expect_plan_matches(rt, g, p, "elementwise ops");
}

int main(int argc, char **argv) {
    return run_tests(argc, argv);
}
