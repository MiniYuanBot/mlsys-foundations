// Fused vs unfused plans for row-wise ops and an MLP epilogue.
//
//   build/lab3/bench_fusion > results/lab3/fusion.csv
//
// GB/s counts the minimum traffic of the whole computation (inputs read once,
// output written once), so unfused and fused are compared on the same scale.
#include "tc/compiler.hpp"
#include "tc/models.hpp"

#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>

using namespace tc;

static Feeds feeds_for(const Graph& g) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> u(-1, 1);
    Feeds f;
    for (int id : g.inputs()) {
        const Node& n = g.node(id);
        HostTensor h{n.shape, std::vector<double>(n.shape.numel())};
        for (auto& x : h.data) x = u(rng);
        f[n.name] = h;
    }
    return f;
}

static double min_bytes(const Graph& g) {
    double b = 4.0 * g.node(g.output()).shape.numel();
    for (int id : g.inputs()) b += 4.0 * g.node(id).shape.numel();
    return b;
}

int main() {
    Runtime rt;
    std::printf("model,shape,plan,kernels,ms,GBps_min_traffic,tflops\n");
    struct Case {
        const char* name;
        std::string shape;
        Graph g;
        double flops;
    };
    std::vector<Case> cases = {
        {"softmax", "8192x4096", models::softmax(8192, 4096), 0},
        {"layernorm", "8192x4096", models::layernorm(8192, 4096), 0},
        {"rmsnorm", "8192x4096", models::rmsnorm(8192, 4096), 0},
        {"matmul_bias_gelu", "4096x1024x4096", models::matmul_bias_gelu(4096, 1024, 4096),
         2.0 * 4096 * 1024 * 4096},
    };
    for (auto& c : cases) {
        const Feeds f = feeds_for(c.g);
        for (const char* which : {"unfused", "fused"}) {
            try {
                const Plan p = std::string(which) == "fused" ? plan_fused(c.g) : plan_unfused(c.g);
                const double ms = rt.time_ms(c.g, p, f);
                std::printf("%s,%s,%s,%zu,%.4f,%.1f,%.2f\n", c.name, c.shape.c_str(), which,
                            p.kernels.size(), ms, min_bytes(c.g) / (ms * 1e6),
                            c.flops / (ms * 1e9));
                std::fflush(stdout);
            } catch (const std::logic_error& e) {
                std::fprintf(stderr, "# skip %s/%s: %s\n", c.name, which, e.what());
            }
        }
    }
    return 0;
}
