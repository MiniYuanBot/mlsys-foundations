// tinycomp pipeline:
//
//   Graph --(A1 toposort)--> order
//         --(A2 evaluate)--> CPU reference values (fp64)        [the oracle]
//         --(B1 partition)--> Plan: list of Kernels             [fusion decisions]
//         --(A3/B2/B3 codegen)--> CUDA C++ source per kernel
//         --(Runtime: NVRTC + driver API)--> run on the GPU
//   Part C searches over equivalent Graphs (rewrites) for the cheapest Plan.
#pragma once

#include "tc/ir.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tc {

// ------------------------------------------------------------- Part A ----
// A1: every node reachable from g.output(), each after all of its srcs,
// each exactly once (depth-first post-order, srcs left to right).
std::vector<int> toposort(const Graph &g);

// Row-major host tensor.
struct HostTensor {
    Shape shape;
    std::vector<double> data;
    double at(int64_t r, int64_t c) const { return data[r * shape.cols + c]; }
};
using Feeds = std::map<std::string, HostTensor>; // input name -> value

// A2: evaluate the graph on the CPU in double precision.
HostTensor evaluate(const Graph &g, const Feeds &feeds);

// --------------------------------------------------------- Part B: plans ----
enum class KernelKind {
    Elementwise, // only elementwise nodes; one thread per output element
    Row,         // contains >= 1 reduction; one thread block per row
    MatMul,      // one MatMul + elementwise prologue on A + elementwise epilogue
};
const char *kind_name(KernelKind k);

struct Kernel {
    KernelKind kind = KernelKind::Elementwise;
    int output = -1;        // the node this kernel materialises
    std::vector<int> nodes; // nodes computed inside, topological order, ends with output
    std::vector<int> reads; // materialised operands it loads (Inputs or other kernels' outputs)
};

struct Plan {
    std::vector<Kernel> kernels; // execution order
    std::string to_string(const Graph &g) const;
};

// Provided: one kernel per non-leaf node (what eager PyTorch does).
Plan plan_unfused(const Graph &g);
// B1: the fusing partitioner (see fusion_student.cpp for the rules).
Plan plan_fused(const Graph &g);
// Provided: fills Kernel::reads and checks the plan is well formed.
void finalize_plan(const Graph &g, Plan &plan);

// ------------------------------------------------------- Part A/B: codegen ----
// Generated kernels are `extern "C" __global__ void tc_kernel(...)` with
// parameters (const float* in0, ..., float* out, ...dims). See codegen_core.cpp.
std::string codegen(const Graph &g, const Kernel &k);
// A3: one C expression for an elementwise op applied to operand expressions,
// e.g. emit_op(Op::Add, {"v1", "v2"}) == "(v1 + v2)".
std::string emit_op(Op op, const std::vector<std::string> &args);
// A3: C expression that loads element (i, j) of a materialised operand with
// shape `s` stored in buffer `buf`, broadcasting to the kernel's [R, C] grid.
// e.g. a [1, C] row vector: "in1[j]";  [R, C]: "in0[(long)i * 64 + j]".
std::string emit_load(const std::string &buf, Shape s);

// ----------------------------------------------------- runtime (provided) ----
class Runtime {
public:
    Runtime();
    ~Runtime();
    // Compile (cached by source text), allocate, run the plan, return the output.
    HostTensor run(const Graph &g, const Plan &plan, const Feeds &feeds);
    // Median GPU time (ms) of the whole plan over `repeats` runs (inputs uploaded once).
    double time_ms(const Graph &g, const Plan &plan, const Feeds &feeds, int repeats = 10);
    int compiled_kernels() const;
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// --------------------------------------------------------- Part C: search ----
// C1: graphs equivalent to g obtained by ONE rule application anywhere.
std::vector<Graph> rewrite_once(const Graph &g);
// C2: probabilistic equivalence test on `trials` random inputs.
bool probably_equivalent(const Graph &a, const Graph &b, int trials = 3, unsigned seed = 1);
// C3: analytical cost of a plan.
struct Device {
    double bandwidth_gbps = 200; // achievable DRAM bandwidth (lab1: cudaMemcpy)
    double flops_tflops =
        5; // achievable FP32 rate of the tinycomp GEMM template (bench_fusion: ~5.3 on a 4060 Laptop)
    double launch_us = 4; // per-kernel overhead
};
struct Cost {
    double bytes = 0, flops = 0, ms = 0;
};
Cost estimate(const Graph &g, const Plan &plan, const Device &dev);
// C3: explore rewrites breadth-first up to `depth`, keep equivalent graphs,
// return them sorted by estimated cost of plan_fused (cheapest first).
struct Candidate {
    Graph graph;
    Plan plan;
    Cost cost;
};
std::vector<Candidate> search(const Graph &g, const Device &dev, int depth = 3);

} // namespace tc
