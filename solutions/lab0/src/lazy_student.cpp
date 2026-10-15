// Part C: scheduling + code generation for the lazy engine. Fill in TODO(C*).
//
//   ctest --test-dir build/lab0 -L partC --output-on-failure
//   build/lab0/bench_fusion
//
// Suggested order: C1 -> C2 (tests only need the schedule) -> C3 (now
// realize_* works end to end) -> C4 (performance).
#include "mt/lazy.hpp"

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace mt::lazy {

[[maybe_unused]] static std::logic_error not_implemented(const char* what) {
    return std::logic_error(std::string("TODO ") + what + " is not implemented yet");
}

// TODO(C1) toposort: depth-first post-order from root.
//   visit(n): if n already visited, return; mark it; visit each src in order; append n.
// A recursive lambda (std::function) is fine for this lab's graph sizes.
std::vector<NodePtr> toposort(const NodePtr& root) {
    std::vector<NodePtr> order;
    std::unordered_set<const Node*> seen;
    std::function<void(const NodePtr&)> visit = [&](const NodePtr& n) {
        if (!seen.insert(n.get()).second) return;
        for (const auto& s : n->srcs) visit(s);
        order.push_back(n);
    };
    visit(root);
    return order;
}

// TODO(C2) schedule_fused. Steps:
//  1. order = toposort(root); count consumers: for every node n in order and
//     every s in n->srcs, ++uses[s] (an edge used twice counts twice).
//  2. materialised(n) = !is_leaf(n->op) && (n == root || n->op == Op::SumLast || uses[n] >= 2)
//  3. For each materialised node m, in topological order, build its kernel:
//       nodes = m plus every non-leaf, non-materialised node reachable from m
//               through srcs WITHOUT passing through another materialised node,
//       listed in topological order (filter `order`, or do a post-order walk).
// Check yourself against the schedules drawn in lecture L0 section 4.
std::vector<Kernel> schedule_fused(const NodePtr& root) {
    const auto order = toposort(root);
    std::unordered_map<const Node*, int> uses;
    for (const auto& n : order)
        for (const auto& s : n->srcs) ++uses[s.get()];
    auto materialised = [&](const NodePtr& n) {
        return !is_leaf(n->op) && (n == root || n->op == Op::SumLast || uses[n.get()] >= 2);
    };
    std::vector<Kernel> kernels;
    for (const auto& m : order) {
        if (!materialised(m)) continue;
        Kernel k{m, {}};
        std::function<void(const NodePtr&)> collect = [&](const NodePtr& n) {
            for (const auto& s : n->srcs)
                if (!is_leaf(s->op) && !materialised(s)) collect(s);
            k.nodes.push_back(n);
        };
        collect(m);
        kernels.push_back(std::move(k));
    }
    return kernels;
}

// TODO(C3) compile a kernel to postfix code.
//  * If kernel.output->op == SumLast: set p.reduce = true,
//    p.reduce_len = last dim of the SumLast's src shape, and start the walk
//    at the src (the reduction loop itself lives in execute()).
//  * emit(n):
//      n is a Const                   -> Instr{Const, value = n->scalar}
//      n is inside the kernel (and is not the SumLast output)
//                                     -> emit each src, then Instr{Unary|Binary, op = n->op}
//      otherwise (Input / other kernel's output) -> Instr{Load, buffer = index of n
//                                        in p.buffers, appending it on first use}
// Tip: put kernel.nodes into an unordered_set<const Node*> for the inside test.
// Example: relu(x*w + b) fused -> Load0 Load1 Mul Load2 Add Relu, buffers [x, w, b].
Program compile(const Kernel& kernel) {
    Program p;
    std::unordered_set<const Node*> inside;
    for (const auto& n : kernel.nodes) inside.insert(n.get());
    std::unordered_map<const Node*, int> slot;
    std::function<void(const NodePtr&)> emit = [&](const NodePtr& n) {
        if (n->op == Op::Const) {
            p.code.push_back(Instr{InstrKind::Const, Op::Const, -1, n->scalar});
        } else if (inside.count(n.get()) && n->op != Op::SumLast) {
            for (const auto& s : n->srcs) emit(s);
            p.code.push_back(
                Instr{is_unary(n->op) ? InstrKind::Unary : InstrKind::Binary, n->op, -1, 0.f});
        } else {
            auto it = slot.find(n.get());
            if (it == slot.end()) {
                it = slot.emplace(n.get(), static_cast<int>(p.buffers.size())).first;
                p.buffers.push_back(n);
            }
            p.code.push_back(Instr{InstrKind::Load, Op::Const, it->second, 0.f});
        }
    };
    NodePtr start = kernel.output;
    if (start->op == Op::SumLast) {
        p.reduce = true;
        start = start->srcs[0];
        p.reduce_len = start->shape.back();
    }
    emit(start);
    return p;
}

// TODO(C4) tiled interpreter.
// eval_scalar (lazy_core.cpp) pays a switch per instruction PER ELEMENT.
// Here you pay it once per instruction per TILE of n elements:
//   keep a stack of tiles, `float regs[16][kTile]` (static thread_local is ok),
//   Load   : copy inputs[b][begin .. begin+n) into a new tile
//   Const  : fill a new tile with the value
//   Unary  : apply in place on the top tile        (inner loop over n)
//   Binary : combine the top two tiles into one    (inner loop over n)
// Finally copy the single remaining tile into out[0..n).
// Write the inner loops so the compiler can vectorise them: switch on the
// op OUTSIDE the element loop, e.g. case Op::Add: for (t..) a[t] += b[t];
void eval_tile(const Program& p, const std::vector<const float*>& inputs, int64_t begin, int n,
               float* out) {
    static thread_local float regs[16][kTile];
    int sp = 0;
    for (const auto& ins : p.code) {
        switch (ins.kind) {
        case InstrKind::Load: {
            const float* src = inputs[ins.buffer] + begin;
            float* dst = regs[sp++];
            for (int t = 0; t < n; ++t) dst[t] = src[t];
            break;
        }
        case InstrKind::Const: {
            float* dst = regs[sp++];
            for (int t = 0; t < n; ++t) dst[t] = ins.value;
            break;
        }
        case InstrKind::Unary: {
            float* a = regs[sp - 1];
            switch (ins.op) {
            case Op::Neg:
                for (int t = 0; t < n; ++t) a[t] = -a[t];
                break;
            case Op::Relu:
                for (int t = 0; t < n; ++t) a[t] = a[t] > 0.f ? a[t] : 0.f;
                break;
            default:
                for (int t = 0; t < n; ++t) a[t] = apply_unary(ins.op, a[t]);
                break;
            }
            break;
        }
        case InstrKind::Binary: {
            float* a = regs[sp - 2];
            const float* b = regs[sp - 1];
            switch (ins.op) {
            case Op::Add:
                for (int t = 0; t < n; ++t) a[t] += b[t];
                break;
            case Op::Sub:
                for (int t = 0; t < n; ++t) a[t] -= b[t];
                break;
            case Op::Mul:
                for (int t = 0; t < n; ++t) a[t] *= b[t];
                break;
            case Op::Div:
                for (int t = 0; t < n; ++t) a[t] /= b[t];
                break;
            default:
                throw std::logic_error("eval_tile: bad binary op");
            }
            --sp;
            break;
        }
        }
        if (sp >= 16) throw std::runtime_error("eval_tile: expression too deep");
    }
    if (sp != 1) throw std::runtime_error("eval_tile: malformed program");
    for (int t = 0; t < n; ++t) out[t] = regs[0][t];
}

} // namespace mt::lazy
