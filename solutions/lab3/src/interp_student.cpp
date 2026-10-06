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
    std::vector<int> order;
    std::unordered_set<int> seen;
    std::function<void(int)> visit = [&](int id) {
        if (!seen.insert(id).second) return;
        for (int s : g.node(id).srcs) visit(s);
        order.push_back(id);
    };
    if (g.output() < 0) throw std::logic_error("toposort: graph has no output");
    visit(g.output());
    return order;
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
    std::vector<HostTensor> val(g.size());
    for (int id : toposort(g)) {
        const Node& n = g.node(id);
        HostTensor out{n.shape, std::vector<double>(n.shape.numel(), 0.0)};
        auto bget = [&](int src, int64_t r, int64_t c) {
            const HostTensor& t = val[src];
            return t.at(t.shape.rows == 1 ? 0 : r, t.shape.cols == 1 ? 0 : c);
        };
        if (n.op == Op::Input) {
            auto it = feeds.find(n.name);
            if (it == feeds.end()) throw std::invalid_argument("missing feed '" + n.name + "'");
            if (it->second.shape != n.shape) throw std::invalid_argument("feed '" + n.name + "' has the wrong shape");
            out = it->second;
        } else if (n.op == Op::Const) {
            out.data[0] = n.value;
        } else if (is_unary(n.op)) {
            const auto& a = val[n.srcs[0]].data;
            for (size_t i = 0; i < a.size(); ++i) {
                const double x = a[i];
                switch (n.op) {
                    case Op::Neg: out.data[i] = -x; break;
                    case Op::Exp: out.data[i] = std::exp(x); break;
                    case Op::Tanh: out.data[i] = std::tanh(x); break;
                    case Op::Relu: out.data[i] = x > 0 ? x : 0; break;
                    case Op::Rsqrt: out.data[i] = 1.0 / std::sqrt(x); break;
                    default: throw std::logic_error("unary");
                }
            }
        } else if (is_binary(n.op)) {
            for (int64_t r = 0; r < n.shape.rows; ++r)
                for (int64_t c = 0; c < n.shape.cols; ++c) {
                    const double x = bget(n.srcs[0], r, c), y = bget(n.srcs[1], r, c);
                    double z = 0;
                    switch (n.op) {
                        case Op::Add: z = x + y; break;
                        case Op::Sub: z = x - y; break;
                        case Op::Mul: z = x * y; break;
                        case Op::Div: z = x / y; break;
                        case Op::Max: z = std::max(x, y); break;
                        default: throw std::logic_error("binary");
                    }
                    out.data[r * n.shape.cols + c] = z;
                }
        } else if (is_reduction(n.op)) {
            const HostTensor& a = val[n.srcs[0]];
            for (int64_t r = 0; r < a.shape.rows; ++r) {
                double acc = n.op == Op::RowSum ? 0.0 : -INFINITY;
                for (int64_t c = 0; c < a.shape.cols; ++c)
                    acc = n.op == Op::RowSum ? acc + a.at(r, c) : std::max(acc, a.at(r, c));
                out.data[r] = acc;
            }
        } else if (n.op == Op::MatMul) {
            const HostTensor &a = val[n.srcs[0]], &b = val[n.srcs[1]];
            const int64_t M = a.shape.rows, K = a.shape.cols, N = b.shape.cols;
            for (int64_t i = 0; i < M; ++i)
                for (int64_t k = 0; k < K; ++k) {
                    const double x = a.at(i, k);
                    for (int64_t j = 0; j < N; ++j) out.data[i * N + j] += x * b.at(k, j);
                }
        }
        val[id] = std::move(out);
    }
    return val[g.output()];
}

}  // namespace tc
