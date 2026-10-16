// Provided: unfused plans, plan validation, printing.
#include "tc/compiler.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tc {

const char *kind_name(KernelKind k) {
    switch (k) {
    case KernelKind::Elementwise:
        return "Elementwise";
    case KernelKind::Row:
        return "Row";
    case KernelKind::MatMul:
        return "MatMul";
    }
    return "?";
}

static KernelKind kind_of_single(Op op) {
    if (op == Op::MatMul) return KernelKind::MatMul;
    if (is_reduction(op)) return KernelKind::Row;
    return KernelKind::Elementwise;
}

Plan plan_unfused(const Graph &g) {
    Plan p;
    for (int id : toposort(g)) {
        const Op op = g.node(id).op;
        if (op == Op::Input || op == Op::Const) continue;
        p.kernels.push_back(Kernel{kind_of_single(op), id, {id}, {}});
    }
    finalize_plan(g, p);
    return p;
}

void finalize_plan(const Graph &g, Plan &plan) {
    std::set<int> materialised;
    for (int id : g.inputs()) materialised.insert(id);
    for (auto &k : plan.kernels) {
        if (k.nodes.empty() || k.nodes.back() != k.output)
            throw std::logic_error("plan: kernel nodes must end with its output");
        const std::set<int> inside(k.nodes.begin(), k.nodes.end());
        int reductions = 0, matmuls = 0;
        k.reads.clear();
        for (int id : k.nodes) {
            const Node &n = g.node(id);
            if (n.op == Op::Input || n.op == Op::Const)
                throw std::logic_error("plan: leaves are not computed in kernels");
            reductions += is_reduction(n.op);
            matmuls += n.op == Op::MatMul;
            for (int s : n.srcs) {
                if (inside.count(s) || g.node(s).op == Op::Const) continue;
                if (!materialised.count(s))
                    throw std::logic_error("plan: kernel for n" + std::to_string(k.output) +
                                           " reads n" + std::to_string(s) +
                                           " before it is materialised");
                if (std::find(k.reads.begin(), k.reads.end(), s) == k.reads.end())
                    k.reads.push_back(s);
            }
        }
        const KernelKind expect = matmuls      ? KernelKind::MatMul
                                  : reductions ? KernelKind::Row
                                               : KernelKind::Elementwise;
        if (matmuls > 1 || (matmuls && reductions))
            throw std::logic_error("plan: unsupported kernel contents");
        if (k.kind != expect)
            throw std::logic_error(std::string("plan: kernel kind ") + kind_name(k.kind) +
                                   " but contents need " + kind_name(expect));
        materialised.insert(k.output);
    }
    if (plan.kernels.empty() || plan.kernels.back().output != g.output())
        throw std::logic_error("plan: last kernel must produce the graph output");
}

std::string Plan::to_string(const Graph &g) const {
    std::ostringstream os;
    for (size_t i = 0; i < kernels.size(); ++i) {
        const auto &k = kernels[i];
        os << "kernel " << i << " " << kind_name(k.kind) << " -> n" << k.output << " ["
           << g.node(k.output).shape.rows << "x" << g.node(k.output).shape.cols << "]  nodes:";
        for (int id : k.nodes) os << " n" << id << ":" << op_name(g.node(id).op);
        os << "  reads:";
        for (int id : k.reads) os << " n" << id;
        os << "\n";
    }
    return os.str();
}

} // namespace tc
