// Part C: mini-Mirage, a tiny superoptimizer.
//
//   build/lab3/test_c            build/lab3/bench_search
//
// Rule-based fusion (Part B) only fuses what the graph already allows. A
// superoptimizer also changes the graph: it generates algebraically
// rewritten graphs, VERIFIES each one is equivalent, and picks the cheapest.
// Mirage (OSDI'25) does this over multi-level "uGraphs" and verifies with
// random tests over finite fields; we do the same idea at toy scale:
//
//   C1 rewrite_once      propose candidates (some rules are only sometimes valid!)
//   C2 probably_equivalent   test candidates on random inputs
//   C3 estimate + search     analytical cost model, breadth-first search
//
// Read lecture L3 section 5 first.
#include "tc/compiler.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <random>
#include <set>
#include <stdexcept>

namespace tc {

[[maybe_unused]] static std::logic_error todo(const char* what) {
    return std::logic_error(std::string("TODO ") + what + " is not implemented yet");
}

// Provided: copy `g`, replacing node `target` by whatever `make` builds.
// `make(ng, m)` receives the new graph and a map old id -> new id that already
// covers every node `target` depends on; it returns the id of the replacement.
// Nodes downstream of target are rebuilt on top of the replacement. Nodes no
// longer reachable from the output are simply left behind (harmless).
static Graph rebuild(const Graph& g, int target, const std::function<int(Graph&, const std::vector<int>&)>& make) {
    Graph ng;
    std::vector<int> m(g.size(), -1);
    for (int id : toposort(g)) {
        const Node& n = g.node(id);
        if (id == target) {
            m[id] = make(ng, m);
            continue;
        }
        if (n.op == Op::Input) m[id] = ng.input(n.name, n.shape);
        else if (n.op == Op::Const) m[id] = ng.constant(n.value);
        else {
            std::vector<int> srcs;
            for (int s : n.srcs) srcs.push_back(m[s]);
            m[id] = ng.add_node(n.op, srcs);
        }
    }
    ng.set_output(m[g.output()]);
    return ng;
}

// Provided: copy `g`, but give consumer node `c` its OWN copy of operand
// number `k` (a fresh node with the same op and operands). This undoes common
// subexpression elimination for one edge: the value will be recomputed.
static Graph clone_operand(const Graph& g, int c, int k) {
    const int victim = g.node(c).srcs.at(k);
    Graph ng;
    std::vector<int> m(g.size(), -1);
    for (int id : toposort(g)) {
        const Node& n = g.node(id);
        if (n.op == Op::Input) m[id] = ng.input(n.name, n.shape);
        else if (n.op == Op::Const) m[id] = ng.constant(n.value);
        else {
            std::vector<int> srcs;
            for (int s : n.srcs) srcs.push_back(m[s]);
            if (id == c) {
                std::vector<int> vs;
                for (int s : g.node(victim).srcs) vs.push_back(m[s]);
                srcs[k] = ng.add_node(g.node(victim).op, vs);
            }
            m[id] = ng.add_node(n.op, srcs);
        }
    }
    ng.set_output(m[g.output()]);
    return ng;
}

// TODO(C1) one rewrite step. For every node t reachable from the output, try
// each rule; every successful match yields ONE new graph.
//
//  R1 row-scale commutes with matmul (use rebuild):
//       MatMul(Mul(a, r), B) -> Mul(MatMul(a, B), r)      (either operand order of the Mul)
//       MatMul(Div(a, r), B) -> Div(MatMul(a, B), r)
//     when r has shape [M, 1] and a has cols > 1: scaling row i of A scales row i of A @ B.
//  R2 recompute instead of share (use clone_operand):
//     for every elementwise node n with two or more consumer edges, and every
//     such edge (consumer c, operand index k): give c its own copy of n.
//     On its own R2 only adds work; combined with fusion it can remove a
//     materialised tensor. This is the heart of FlashAttention.
//  R3 SPECULATIVE column-scale "commute" (use rebuild):
//       MatMul(Mul(a, s), B) -> Mul(MatMul(a, B), s)  for s of shape [1, K], only when K == N
//     It is WRONG; a real superoptimizer's generator also proposes bad
//     candidates and relies on the verifier (C2) to reject them.
std::vector<Graph> rewrite_once(const Graph& g) {
    (void)g; (void)rebuild; (void)clone_operand;
    throw todo("C1 rewrite_once");
}

// TODO(C2) probabilistic equivalence. For each trial:
//   - make random feeds for every Input of `a` (uniform in [-1, 1], from
//     std::mt19937(seed + trial)); `b` must have exactly the same inputs
//     (name and shape), otherwise return false
//   - evaluate both (Part A2) and compare: shapes equal and every element
//     |x - y| <= 1e-9 * (1 + max |x|)
// Return true only if every trial matches. Why random inputs are convincing:
// two different polynomials of degree d agree on a random point with
// probability <= d / |field| (Schwartz-Zippel); Mirage uses exactly this,
// over a finite field so that there is no rounding at all.
bool probably_equivalent(const Graph& a, const Graph& b, int trials, unsigned seed) {
    (void)a; (void)b; (void)trials; (void)seed;
    throw todo("C2 probably_equivalent");
}

// TODO(C3a) cost model, per kernel k of the plan:
//   bytes = 4 * (sum of numel over k.reads + numel of k.output)
//   flops = 2 * M * N * K for its MatMul node (0 if none)
//   time  = max(bytes / bandwidth, flops / flop rate) + launch overhead
// Sum over kernels. Units: dev.bandwidth_gbps is GB/s (1e9 B/s),
// dev.flops_tflops is TFLOP/s, dev.launch_us is microseconds; Cost.ms in ms.
// This ignores re-reads inside a kernel (a GEMM reads A once per N-tile; a Row
// kernel re-reads its row once per reduction) and caching. The bench compares
// the model with reality: notice where it is right and where it is wrong.
Cost estimate(const Graph& g, const Plan& plan, const Device& dev) {
    (void)g; (void)plan; (void)dev;
    throw todo("C3a estimate");
}

// TODO(C3b) breadth-first search over rewrites:
//   frontier = {g}; seen = {g.canonical()}; keep = [g]
//   repeat `depth` times: for each graph f in frontier, for each c in rewrite_once(f):
//       skip if c.canonical() was seen; mark it seen;
//       keep it (and put it in the next frontier) only if probably_equivalent(g, c)
//   For every kept graph build plan_fused + estimate, return sorted by cost.ms.
std::vector<Candidate> search(const Graph& g, const Device& dev, int depth) {
    (void)g; (void)dev; (void)depth;
    throw todo("C3b search");
}

}  // namespace tc
