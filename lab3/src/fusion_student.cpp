// Part B1: the fusing partitioner.
//
//   build/lab3/test_b B1        build/lab3/tc_dump layernorm
//
// Algorithm ("absorb into consumers"), the same idea TorchInductor's
// scheduler and XLA's fusion pass start from:
//
//   visit the non-leaf nodes in REVERSE topological order (consumers first);
//   for node n:
//     if n is not the graph output, all consumers of n are in ONE kernel K,
//        and can_absorb(K, n) says yes      -> put n into K (it never touches memory)
//     else                                  -> n starts a new kernel (it is materialised)
//
// "All consumers in one kernel" is what allows recomputation-free fusion:
// softmax's e = exp(x - m) feeds both row_sum and the final division, both
// in the same row kernel, so e is never written out.
#include "tc/compiler.hpp"

#include <algorithm>
#include <functional>
#include <set>
#include <stdexcept>

namespace tc {

[[maybe_unused]] static std::logic_error todo(const char* what) {
    return std::logic_error(std::string("TODO ") + what + " is not implemented yet");
}

namespace {

// Kernel under construction.
struct Group {
    KernelKind kind;
    int output;
    std::set<int> nodes;
    int64_t rows;         // rows of the kernel's iteration space
    int64_t width;        // length of the column loop: output cols, or the reduced dim (0 = not yet known)
    int matmul = -1;      // MatMul kernels: the MatMul node
    std::set<int> prologue;  // MatMul kernels: nodes fused into the A operand
};

// TODO(B1a) the fusion rules. Return true if node n may join group K, given
// that every consumer of n is already in K. Rules:
//
//  n elementwise:
//    K Elementwise or Row: yes.
//    K MatMul: n must sit entirely on ONE side of the matmul.
//      prologue side: every consumer of n is either the MatMul with n as its
//                     FIRST operand (srcs[0]) or a node in K.prologue
//      epilogue side: no consumer of n is the MatMul and none is in K.prologue
//      Anything else (e.g. n is the matmul's second operand B, or n feeds
//      both sides) -> no.
//      Set *to_prologue to tell the caller which side n joined.
//  n reduction (RowSum / RowMax), input shape [r, w]:
//    K MatMul -> no.
//    K Elementwise / Row -> yes iff r == K.rows and (K.width == 0 or w == K.width)
//                           (one block per row, one column loop length)
//  n MatMul:
//    K Elementwise with n.shape == shape of K.output -> yes (K becomes MatMul, and
//      everything already in K is the epilogue)
//    otherwise -> no (Row kernels cannot contain a GEMM; one GEMM per kernel)
//
// `cons` = consumers of n (each edge once per use).
[[maybe_unused]] bool can_absorb(const Graph& g, const Group& K, int n, const std::vector<int>& cons, bool* to_prologue) {
    const Node& node = g.node(n);
    *to_prologue = false;
    (void)node; (void)K; (void)cons;
    throw todo("B1a can_absorb");
}

}  // namespace

// TODO(B1b) the driver. Most of it is bookkeeping; the decisions are in can_absorb.
//   1. order = toposort(g); cons = g.consumers(); kernel_of[id] = -1 for all.
//   2. For id in reverse(order), skipping Input/Const:
//        if id != g.output() and all consumers of id map to the same group k
//           and can_absorb(g, groups[k], id, cons[id], &pro):
//             add id to groups[k]; if pro add to prologue;
//             update the group: a reduction sets kind Row (if it was Elementwise) and
//             width = its input cols; a MatMul sets kind MatMul and matmul = id.
//        else: new group {kind from id's op, output id, rows = id's rows, ...}:
//             Elementwise: width = its cols if > 1, else 0 (unknown);
//             RowSum/RowMax: kind Row, width = input cols;
//             MatMul: kind MatMul, matmul = id.
//   3. Kernels must run producers first: groups were created consumers-first,
//      so emit them in REVERSE creation order. Each Kernel's nodes = its set,
//      sorted by position in `order`.
//   4. finalize_plan(g, plan) fills `reads` and validates; return the plan.
Plan plan_fused(const Graph& g) {
    (void)g;
    throw todo("B1b plan_fused");
}

}  // namespace tc
