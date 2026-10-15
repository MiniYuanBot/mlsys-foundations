// Part B tests: partitioning decisions and fused code on the GPU.   test_b B2
#include "helpers.hpp"
#include "tc/models.hpp"

#include <tuple>

using namespace tc;
using namespace tc::test;

TEST(B1_elementwise_chain_is_one_kernel) {
    Graph g;
    int x = g.input("x", {64, 64}), b = g.input("b", {1, 64});
    g.set_output(g.relu(g.add(g.mul(x, g.constant(2.0)), b)));
    const Plan p = plan_fused(g);
    CHECK_EQ(p.kernels.size(), 1u);
    CHECK(p.kernels[0].kind == KernelKind::Elementwise);
}

TEST(B1_softmax_layernorm_rmsnorm_are_one_row_kernel) {
    for (const Graph& g :
         {models::softmax(8, 100), models::layernorm(8, 100), models::rmsnorm(8, 100)}) {
        const Plan p = plan_fused(g);
        CHECK_EQ(p.kernels.size(), 1u);
        CHECK(p.kernels[0].kind == KernelKind::Row);
    }
}

TEST(B1_matmul_absorbs_epilogue) {
    Graph g = models::matmul_bias_gelu(64, 32, 48);
    const Plan p = plan_fused(g);
    CHECK_EQ(p.kernels.size(), 1u);
    CHECK(p.kernels[0].kind == KernelKind::MatMul);
    size_t compute_nodes = 0;
    for (int id : toposort(g))
        compute_nodes += g.node(id).op != Op::Input && g.node(id).op != Op::Const;
    CHECK_EQ(plan_unfused(g).kernels.size(), compute_nodes);
}

TEST(B1_matmul_absorbs_elementwise_prologue_on_A_only) {
    Graph g;
    int x = g.input("x", {64, 32}), s = g.input("s", {1, 32}), w = g.input("W", {32, 16});
    int a = g.relu(g.mul(x, s));
    g.set_output(g.matmul(a, g.mul(w, g.constant(2.0)))); // prologue on A ok, on B not
    const Plan p = plan_fused(g);
    CHECK_EQ(p.kernels.size(), 2u);
    CHECK(p.kernels[0].kind == KernelKind::Elementwise); // W * 2 is materialised
    CHECK(p.kernels[1].kind == KernelKind::MatMul);
    CHECK_EQ(p.kernels[1].nodes.size(), 3u); // mul, relu, matmul
}

TEST(B1_reduction_cannot_join_matmul) {
    Graph g = models::rmsnorm_matmul(64, 32, 16);
    const Plan p = plan_fused(g);
    CHECK_EQ(p.kernels.size(), 2u);
    CHECK(p.kernels[0].kind == KernelKind::Row);
    CHECK(p.kernels[1].kind == KernelKind::MatMul);
}

TEST(B1_node_feeding_two_kernels_is_materialised) {
    Graph g;
    int x = g.input("x", {16, 16}), w = g.input("W", {16, 16});
    int t = g.exp(x); // feeds the matmul's A AND the epilogue: both sides
    g.set_output(g.add(g.matmul(t, w), t));
    const Plan p = plan_fused(g);
    CHECK_EQ(p.kernels.size(), 2u);
    CHECK_EQ(p.kernels[0].output, t);
}

TEST(B1_reduction_width_must_match) {
    Graph g;
    int x = g.input("x", {8, 32}), y = g.input("y", {8, 16});
    g.set_output(g.mul(y, g.row_sum(x))); // a 16-column loop cannot also reduce 32 columns
    CHECK_EQ(plan_fused(g).kernels.size(), 2u);
}

TEST(B2_row_kernels_on_gpu) {
    Runtime rt;
    for (int64_t C : {1, 7, 256, 1000, 4097}) {
        Graph s = models::softmax(33, C), l = models::layernorm(33, C), r = models::rmsnorm(33, C);
        expect_plan_matches(rt, s, plan_fused(s), "softmax C=" + std::to_string(C));
        expect_plan_matches(rt, l, plan_fused(l), "layernorm C=" + std::to_string(C), 1e-3);
        expect_plan_matches(rt, r, plan_fused(r), "rmsnorm C=" + std::to_string(C));
    }
}

TEST(B2_row_kernel_with_scalar_output) {
    Runtime rt;
    Graph g;
    int x = g.input("x", {40, 300});
    g.set_output(g.rsqrt(g.add(g.row_sum(g.mul(x, x)), g.constant(1.0)))); // [40, 1]
    const Plan p = plan_fused(g);
    CHECK_EQ(p.kernels.size(), 1u);
    expect_plan_matches(rt, g, p, "row kernel with [R,1] output");
}

TEST(B3_matmul_epilogue_on_gpu) {
    Runtime rt;
    const std::tuple<int64_t, int64_t, int64_t> shapes[] = {
        {64, 32, 64}, {1, 5, 3}, {100, 70, 130}, {256, 512, 192}};
    for (const auto& [M, K, N] : shapes) {
        Graph g = models::matmul_bias_gelu(M, K, N);
        expect_plan_matches(rt, g, plan_fused(g), "matmul_bias_gelu", 1e-4);
    }
}

TEST(B3_matmul_prologue_on_gpu) {
    Runtime rt;
    Graph g;
    int x = g.input("x", {77, 45}), s = g.input("s", {1, 45}), r = g.input("r", {77, 1}),
        w = g.input("W", {45, 33});
    g.set_output(g.mul(g.matmul(g.mul(g.relu(g.mul(x, s)), r), w), g.constant(0.5)));
    const Plan p = plan_fused(g);
    CHECK_EQ(p.kernels.size(), 1u);
    expect_plan_matches(rt, g, p, "matmul prologue", 1e-4);
}

TEST(B_fused_equals_unfused_on_rmsnorm_matmul) {
    Runtime rt;
    Graph g = models::rmsnorm_matmul(128, 96, 80);
    expect_plan_matches(rt, g, plan_unfused(g), "unfused");
    expect_plan_matches(rt, g, plan_fused(g), "fused");
}

int main(int argc, char** argv) {
    return run_tests(argc, argv);
}
