// Part C tests: rewrites, verification, cost model, search.   test_c C2
#include "helpers.hpp"
#include "tc/models.hpp"

#include <algorithm>

using namespace tc;
using namespace tc::test;

static bool contains(const std::vector<Graph> &gs, const std::string &canon) {
    return std::any_of(gs.begin(), gs.end(),
                       [&](const Graph &g) { return g.canonical() == canon; });
}

TEST(C0_canonical_distinguishes_sharing_from_copies) {
    Graph shared, copies;
    int x = shared.input("x", {4, 4});
    int e = shared.exp(x);
    shared.set_output(shared.add(e, e));
    int y = copies.input("x", {4, 4});
    copies.set_output(copies.add(copies.exp(y), copies.exp(y)));
    CHECK(shared.canonical() != copies.canonical());
    Graph renumbered;
    renumbered.constant(3.0); // shifts every id by one
    int z = renumbered.input("x", {4, 4});
    int e2 = renumbered.exp(z);
    renumbered.set_output(renumbered.add(e2, e2));
    CHECK_EQ(shared.canonical(), renumbered.canonical());
}

TEST(C1_R1_row_scale_moves_out_of_matmul) {
    for (bool use_div : {false, true}) {
        Graph g;
        int a = g.input("a", {8, 4}), r = g.input("r", {8, 1}), b = g.input("b", {4, 3});
        g.set_output(g.matmul(use_div ? g.div(a, r) : g.mul(a, r), b));
        Graph want;
        int a2 = want.input("a", {8, 4}), r2 = want.input("r", {8, 1}),
            b2 = want.input("b", {4, 3});
        const int mm = want.matmul(a2, b2);
        want.set_output(use_div ? want.div(mm, r2) : want.mul(mm, r2));
        CHECK(contains(rewrite_once(g), want.canonical()));
    }
}

TEST(C1_R2_recompute_instead_of_share) {
    Graph g;
    int x = g.input("x", {8, 4});
    int e = g.exp(x);
    g.set_output(g.add(g.row_sum(e), g.row_max(e)));
    const auto cands = rewrite_once(g);
    CHECK_EQ(cands.size(), 2u); // one per consumer edge of e
    for (const Graph &c : cands) {
        int exps = 0;
        for (int id : toposort(c)) exps += c.node(id).op == Op::Exp;
        CHECK_EQ(exps, 2); // e now exists twice
    }
}

TEST(C1_R3_speculative_column_scale_is_proposed_when_shapes_allow) {
    Graph g;
    int a = g.input("a", {8, 4}), s = g.input("s", {1, 4}), b = g.input("b", {4, 4}); // K == N
    g.set_output(g.matmul(g.mul(a, s), b));
    Graph bad;
    int a2 = bad.input("a", {8, 4}), s2 = bad.input("s", {1, 4}), b2 = bad.input("b", {4, 4});
    bad.set_output(bad.mul(bad.matmul(a2, b2), s2));
    CHECK(contains(rewrite_once(g), bad.canonical()));
}

TEST(C2_accepts_valid_rewrites) {
    Graph g = models::softmax_matmul(12, 9, 5);
    const auto cands = rewrite_once(g);
    CHECK(!cands.empty());
    for (const Graph &c : cands) CHECK(probably_equivalent(g, c));
    CHECK(probably_equivalent(g, g));
}

TEST(C2_rejects_invalid_rewrite) {
    Graph g, bad;
    int a = g.input("a", {8, 4}), s = g.input("s", {1, 4}), b = g.input("b", {4, 4});
    g.set_output(g.matmul(g.mul(a, s), b));
    int a2 = bad.input("a", {8, 4}), s2 = bad.input("s", {1, 4}), b2 = bad.input("b", {4, 4});
    bad.set_output(bad.mul(bad.matmul(a2, b2), s2));
    CHECK(!probably_equivalent(g, bad));
    Graph other;
    other.set_output(other.exp(other.input("y", {8, 4}))); // different inputs
    CHECK(!probably_equivalent(g, other));
}

TEST(C3a_cost_counts_bytes_and_flops) {
    const int64_t M = 1024, K = 512, N = 256;
    Graph g = models::rmsnorm_matmul(M, K, N);
    const Device dev{100, 10, 0};
    const Cost unfused = estimate(g, plan_unfused(g), dev), fused = estimate(g, plan_fused(g), dev);
    // fused = row kernel (rowsum(x*x): reads x, writes [M,1])
    //       + matmul kernel (prologue x*r*gamma, epilogue +res: reads x, rowsum, gamma, W, res; writes y)
    const double row_bytes = 4.0 * (M * K + M);
    const double mm_bytes = 4.0 * (M * K + M + K + K * N + M * N + M * N),
                 mm_flops = 2.0 * M * N * K;
    CHECK_NEAR(fused.bytes, row_bytes + mm_bytes, 1);
    CHECK_NEAR(fused.flops, mm_flops, 1);
    CHECK(unfused.bytes > 2 * fused.bytes);
    // per kernel: max(bytes / bandwidth, flops / rate); 100 GB/s = 1e8 B/ms, 10 TFLOP/s = 1e10 FLOP/ms
    CHECK_NEAR(fused.ms, row_bytes / 1e8 + std::max(mm_bytes / 1e8, mm_flops / 1e10), 1e-9);
    const Cost with_launch = estimate(g, plan_fused(g), Device{100, 10, 5});
    CHECK_NEAR(with_launch.ms - fused.ms, 2 * 5e-3, 1e-9); // 2 kernels x 5 us
}

TEST(C3b_search_rediscovers_recompute) {
    const int64_t M = 2048, K = 2048, N = 64;
    Graph g = models::softmax_matmul(M, K, N);
    const Device dev;
    // Rule-based fusion has to materialise e = exp(x - m): written once, read twice.
    CHECK(estimate(g, plan_fused(g), dev).bytes > 4.0 * 4 * M * K);
    const auto cands = search(g, dev, 3);
    CHECK(cands.size() >= 3);
    for (size_t i = 1; i < cands.size(); ++i) CHECK(cands[i - 1].cost.ms <= cands[i].cost.ms);
    const Candidate &best = cands.front();
    // The winner never writes an [M, K] tensor; x is read three times instead.
    for (const auto &k : best.plan.kernels) CHECK(best.graph.node(k.output).shape.numel() < M * K);
    CHECK(best.cost.bytes < 4.0 * 3.1 * M * K);
    for (const auto &c : cands) CHECK(probably_equivalent(g, c.graph));
}

TEST(C_best_candidate_runs_on_gpu) {
    Runtime rt;
    Graph g = models::softmax_matmul(200, 300, 70);
    const auto best = search(g, Device{}, 3).front();
    expect_plan_matches(rt, best.graph, best.plan, "best candidate", 1e-4);
}

int main(int argc, char **argv) {
    return run_tests(argc, argv);
}
