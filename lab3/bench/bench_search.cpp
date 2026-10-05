// Part C: does the superoptimizer's choice actually win, and does the cost
// model rank the candidates the same way the GPU does?
//
//   build/lab3/bench_search [M K N] > results/lab3/search.csv
//
// For every equivalent candidate found by search(): predicted ms (cost model)
// and measured ms (median of 20 runs), plus the rank under each.
#include "tc/compiler.hpp"
#include "tc/models.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>

using namespace tc;

int main(int argc, char** argv) {
    const int64_t M = argc > 3 ? std::atoll(argv[1]) : 8192, K = argc > 3 ? std::atoll(argv[2]) : 8192,
                  N = argc > 3 ? std::atoll(argv[3]) : 64;
    Graph g = models::softmax_matmul(M, K, N);  // attention-shaped: 8192 queries, 8192 keys, head dim 64
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u(-1, 1);
    Feeds f;
    for (int id : g.inputs()) {
        const Node& n = g.node(id);
        HostTensor h{n.shape, std::vector<double>(n.shape.numel())};
        for (auto& x : h.data) x = u(rng);
        f[n.name] = h;
    }
    Runtime rt;
    const Device dev;
    std::vector<Candidate> cands;
    try {
        cands = search(g, dev, 3);
    } catch (const std::logic_error& e) {
        std::fprintf(stderr, "# %s\n", e.what());
        return 1;
    }
    struct Row {
        double pred, meas;
        size_t kernels;
        double mb;
        std::string canon;
    };
    std::vector<Row> rows;
    for (const auto& c : cands)
        rows.push_back({c.cost.ms, rt.time_ms(c.graph, c.plan, f, 20), c.plan.kernels.size(), c.cost.bytes / 1e6,
                        c.graph.canonical()});
    const double unfused_ms = rt.time_ms(g, plan_unfused(g), f, 20);
    std::vector<size_t> by_meas(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) by_meas[i] = i;
    std::sort(by_meas.begin(), by_meas.end(), [&](size_t a, size_t b) { return rows[a].meas < rows[b].meas; });
    std::vector<size_t> meas_rank(rows.size());
    for (size_t r = 0; r < by_meas.size(); ++r) meas_rank[by_meas[r]] = r;
    std::printf("# M=%lld K=%lld N=%lld, unfused original: %.4f ms\n", (long long)M, (long long)K, (long long)N,
                unfused_ms);
    std::printf("pred_rank,meas_rank,pred_ms,meas_ms,kernels,MB_model,graph\n");
    for (size_t i = 0; i < rows.size(); ++i)
        std::printf("%zu,%zu,%.4f,%.4f,%zu,%.1f,\"%s\"\n", i, meas_rank[i], rows[i].pred, rows[i].meas, rows[i].kernels,
                    rows[i].mb, rows[i].canon.c_str());
    return 0;
}
