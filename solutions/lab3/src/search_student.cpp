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
static Graph rebuild(const Graph& g, int target,
                     const std::function<int(Graph&, const std::vector<int>&)>& make) {
    Graph ng;
    std::vector<int> m(g.size(), -1);
    for (int id : toposort(g)) {
        const Node& n = g.node(id);
        if (id == target) {
            m[id] = make(ng, m);
            continue;
        }
        if (n.op == Op::Input)
            m[id] = ng.input(n.name, n.shape);
        else if (n.op == Op::Const)
            m[id] = ng.constant(n.value);
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
        if (n.op == Op::Input)
            m[id] = ng.input(n.name, n.shape);
        else if (n.op == Op::Const)
            m[id] = ng.constant(n.value);
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
    std::vector<Graph> out;
    const auto cons = g.consumers();
    for (int t : toposort(g)) {
        const Node& n = g.node(t);
        if (n.op == Op::MatMul) {
            const int lhs = n.srcs[0], B = n.srcs[1];
            const Node& L = g.node(lhs);
            if (L.op != Op::Mul && L.op != Op::Div) continue;
            for (int side = 0; side < 2; ++side) {
                if (L.op == Op::Div && side == 1) break; // only a / r
                const int a = L.srcs[side], s = L.srcs[1 - side];
                const Shape ss = g.node(s).shape, as = g.node(a).shape;
                const bool row_scale = ss.cols == 1 && ss.rows == as.rows && as.cols > 1;
                const bool col_scale = L.op == Op::Mul && ss.rows == 1 && ss.cols == as.cols &&
                                       as.cols > 1 && as.cols == n.shape.cols;
                if (!row_scale && !col_scale) continue; // R1 or R3
                const Op op = L.op;
                out.push_back(rebuild(g, t, [&](Graph& ng, const std::vector<int>& m) {
                    return ng.add_node(op, {ng.matmul(m[a], m[B]), m[s]});
                }));
            }
        }
        if (is_elementwise(n.op) && cons[t].size() >= 2) {
            std::set<std::pair<int, int>> edges;
            for (int c : cons[t])
                for (size_t k = 0; k < g.node(c).srcs.size(); ++k)
                    if (g.node(c).srcs[k] == t) edges.insert({c, static_cast<int>(k)});
            for (auto [c, k] : edges) out.push_back(clone_operand(g, c, k));
        }
    }
    return out;
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
    auto sig = [](const Graph& g) {
        std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> s;
        for (int id : g.inputs())
            s.push_back({g.node(id).name, {g.node(id).shape.rows, g.node(id).shape.cols}});
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        return s;
    };
    if (sig(a) != sig(b)) return false;
    for (int t = 0; t < trials; ++t) {
        std::mt19937 rng(seed + t);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        Feeds feeds;
        for (int id : a.inputs()) {
            const Node& n = a.node(id);
            if (feeds.count(n.name)) continue;
            HostTensor h{n.shape, std::vector<double>(n.shape.numel())};
            for (auto& x : h.data) x = u(rng);
            feeds[n.name] = h;
        }
        const HostTensor x = evaluate(a, feeds), y = evaluate(b, feeds);
        if (x.shape != y.shape) return false;
        double mx = 0;
        for (double v : x.data) mx = std::max(mx, std::fabs(v));
        for (size_t i = 0; i < x.data.size(); ++i)
            if (!(std::fabs(x.data[i] - y.data[i]) <= 1e-9 * (1 + mx))) return false;
    }
    return true;
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
    Cost c;
    for (const Kernel& k : plan.kernels) {
        double bytes = 4.0 * g.node(k.output).shape.numel(), flops = 0;
        for (int r : k.reads) bytes += 4.0 * g.node(r).shape.numel();
        for (int id : k.nodes)
            if (g.node(id).op == Op::MatMul) {
                const Node& n = g.node(id);
                flops += 2.0 * n.shape.rows * n.shape.cols * g.node(n.srcs[0]).shape.cols;
            }
        const double ms =
            std::max(bytes / (dev.bandwidth_gbps * 1e6), flops / (dev.flops_tflops * 1e9));
        c.bytes += bytes;
        c.flops += flops;
        c.ms += ms + dev.launch_us * 1e-3;
    }
    return c;
}

// TODO(C3b) breadth-first search over rewrites:
//   frontier = {g}; seen = {g.canonical()}; keep = [g]
//   repeat `depth` times: for each graph f in frontier, for each c in rewrite_once(f):
//       skip if c.canonical() was seen; mark it seen;
//       keep it (and put it in the next frontier) only if probably_equivalent(g, c)
//   For every kept graph build plan_fused + estimate, return sorted by cost.ms.
std::vector<Candidate> search(const Graph& g, const Device& dev, int depth) {
    std::vector<Graph> keep{g}, frontier{g};
    std::set<std::string> seen{g.canonical()};
    for (int d = 0; d < depth; ++d) {
        std::vector<Graph> next;
        for (const Graph& f : frontier)
            for (Graph& c : rewrite_once(f)) {
                if (!seen.insert(c.canonical()).second) continue;
                if (!probably_equivalent(g, c)) continue;
                next.push_back(c);
                keep.push_back(c);
            }
        frontier = std::move(next);
    }
    std::vector<Candidate> out;
    for (Graph& k : keep) {
        Plan p = plan_fused(k);
        Cost cost = estimate(k, p, dev);
        out.push_back({std::move(k), std::move(p), cost});
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const Candidate& a, const Candidate& b) { return a.cost.ms < b.cost.ms; });
    return out;
}

} // namespace tc
