// Part A1-A2: graph traversal and the CPU reference interpreter.
//
//   build/lab3/test_a A1      (or: ctest --test-dir build/lab3 -L partA)
//
// The interpreter is the ORACLE for everything else in this lab: generated
// GPU code is checked against it, and Part C uses it to decide whether two
// graphs compute the same function. Keep it simple and obviously correct.
#include "tc/compiler.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <unordered_set>

namespace tc {

[[maybe_unused]] static std::logic_error todo(const char* what) {
    return std::logic_error(std::string("TODO ") + what + " is not implemented yet");
}

// TODO(A1) toposort: depth-first post-order from g.output(), visiting srcs
// left to right; each node once. (Same algorithm as lab0 C1.)
std::vector<int> toposort(const Graph& g) {
    (void)g;
    throw todo("A1 toposort");
}

// TODO(A2) evaluate in double precision. For each node in topological order
// compute a HostTensor of node.shape:
//   Input    -> feeds.at(name) (check the shape; throw std::invalid_argument if wrong)
//   Const    -> the 1x1 value
//   unary    -> Neg: -x, Exp: exp, Tanh: tanh, Relu: max(x, 0), Rsqrt: 1/sqrt(x)
//   binary   -> broadcasting: element (r, c) of an operand with shape s is
//               at(s.rows == 1 ? 0 : r, s.cols == 1 ? 0 : c)
//               Add, Sub, Mul, Div, Max (std::max)
//   RowSum   -> out(r, 0) = sum over c;   RowMax -> max over c
//   MatMul   -> out(i, j) = sum over k of A(i, k) * B(k, j)
// Return the value of g.output().
HostTensor evaluate(const Graph& g, const Feeds& feeds) {
    (void)g; (void)feeds;
    throw todo("A2 evaluate");
}

}  // namespace tc
