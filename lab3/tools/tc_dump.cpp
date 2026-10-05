// Print a model's graph, its unfused and fused plans, and the generated CUDA.
//
//   build/lab3/tc_dump softmax | layernorm | rmsnorm | gelu | rmsnorm_matmul [--search]
//
// --search also runs Part C and prints every equivalent candidate with its cost.
#include "tc/compiler.hpp"
#include "tc/models.hpp"

#include <cstdio>
#include <iostream>
#include <string>

using namespace tc;

int main(int argc, char** argv) {
    const std::string name = argc > 1 ? argv[1] : "softmax";
    const bool do_search = argc > 2 && std::string(argv[2]) == "--search";
    Graph g;
    if (name == "softmax") g = models::softmax(4, 128);
    else if (name == "layernorm") g = models::layernorm(4, 128);
    else if (name == "rmsnorm") g = models::rmsnorm(4, 128);
    else if (name == "gelu") g = models::matmul_bias_gelu(64, 64, 64);
    else if (name == "rmsnorm_matmul") g = models::rmsnorm_matmul(64, 64, 32);
    else if (name == "softmax_matmul") g = models::softmax_matmul(64, 64, 32);
    else {
        std::fprintf(stderr, "usage: tc_dump softmax|layernorm|rmsnorm|gelu|rmsnorm_matmul|softmax_matmul [--search]\n");
        return 2;
    }
    try {
        std::cout << "=== graph\n" << g.to_string();
        std::cout << "\n=== unfused plan\n" << plan_unfused(g).to_string(g);
        const Plan p = plan_fused(g);
        std::cout << "\n=== fused plan\n" << p.to_string(g);
        for (size_t i = 0; i < p.kernels.size(); ++i)
            std::cout << "\n=== kernel " << i << " source\n" << codegen(g, p.kernels[i]);
        if (do_search) {
            std::cout << "\n=== search (cost model, cheapest first)\n";
            for (const auto& c : search(g, Device{}, 3)) {
                std::cout << "--- " << c.cost.ms << " ms est, " << c.cost.bytes / 1e6 << " MB, " << c.plan.kernels.size()
                          << " kernels: " << c.graph.canonical() << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
