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
    (void)root;
    throw not_implemented("C1 toposort");
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
    (void)root;
    throw not_implemented("C2 schedule_fused");
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
    (void)kernel;
    throw not_implemented("C3 compile");
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
void eval_tile(const Program& p, const std::vector<const float*>& inputs, int64_t begin, int n, float* out) {
    (void)p; (void)inputs; (void)begin; (void)n; (void)out;
    throw not_implemented("C4 eval_tile");
}

}  // namespace mt::lazy
