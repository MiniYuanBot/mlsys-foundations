// Shared helpers for the lab3 tests and tools.
#pragma once

#include "check.hpp"
#include "tc/compiler.hpp"

#include <cmath>
#include <random>
#include <sstream>

namespace tc::test {

// Random feeds for every input, uniform in [-1, 1].
inline Feeds random_feeds(const Graph& g, unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Feeds f;
    for (int id : g.inputs()) {
        const Node& n = g.node(id);
        HostTensor h{n.shape, std::vector<double>(n.shape.numel())};
        for (auto& x : h.data) x = u(rng);
        f[n.name] = h;
    }
    return f;
}

inline void expect_close(const HostTensor& got, const HostTensor& want, double tol,
                         const std::string& what) {
    if (got.shape != want.shape) throw check::Failure(what + ": shape mismatch");
    double worst = 0;
    size_t at = 0;
    for (size_t i = 0; i < got.data.size(); ++i) {
        const double d = std::fabs(got.data[i] - want.data[i]) / (1 + std::fabs(want.data[i]));
        if (!(d <= worst)) {
            worst = d;
            at = i;
        }
    }
    if (!(worst <= tol)) {
        std::ostringstream os;
        os << what << ": element " << at << " = " << got.data[at] << ", expected " << want.data[at]
           << " (relative error " << worst << ")";
        throw check::Failure(os.str());
    }
}

// Run `plan` on the GPU and compare with the CPU interpreter.
inline void expect_plan_matches(Runtime& rt, const Graph& g, const Plan& plan,
                                const std::string& what, double tol = 2e-4, unsigned seed = 7) {
    const Feeds f = random_feeds(g, seed);
    expect_close(rt.run(g, plan, f), evaluate(g, f), tol, what);
}

inline int count_kind(const Plan& p, KernelKind k) {
    int n = 0;
    for (const auto& kk : p.kernels) n += kk.kind == k;
    return n;
}

} // namespace tc::test
