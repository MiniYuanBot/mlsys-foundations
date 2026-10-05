// tinycomp IR: a DAG of operations on 2-D fp32 tensors.
//
// Every value has shape [rows, cols]. Broadcasting is NumPy-style but only in
// 2-D: a binary op accepts operands whose dims are each either equal to the
// result's or 1, e.g. [M,N] + [1,N] (row vector), [M,N] * [M,1] (per-row
// scale), x * 0.5 ([1,1] constant).
//
//   Graph g;
//   int x = g.input("x", {M, N});
//   int m = g.row_max(x);              // [M, 1]
//   int e = g.exp(g.sub(x, m));        // [M, N]
//   int y = g.div(e, g.row_sum(e));    // softmax
//   g.set_output(y);
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tc {

struct Shape {
    int64_t rows = 1, cols = 1;
    bool operator==(const Shape& o) const { return rows == o.rows && cols == o.cols; }
    bool operator!=(const Shape& o) const { return !(*this == o); }
    int64_t numel() const { return rows * cols; }
};

enum class Op {
    Input, Const,
    Neg, Exp, Tanh, Relu, Rsqrt,          // unary elementwise
    Add, Sub, Mul, Div, Max,              // binary elementwise (broadcasting)
    RowSum, RowMax,                       // [R,C] -> [R,1]
    MatMul,                               // [M,K] x [K,N] -> [M,N]
};

bool is_unary(Op op);
bool is_binary(Op op);
bool is_elementwise(Op op);  // unary or binary
bool is_reduction(Op op);
const char* op_name(Op op);

struct Node {
    Op op;
    std::vector<int> srcs;  // operand node ids
    Shape shape;
    std::string name;       // Input only
    double value = 0;       // Const only
};

class Graph {
public:
    int input(const std::string& name, Shape shape);
    int constant(double value);
    int neg(int a) { return unary(Op::Neg, a); }
    int exp(int a) { return unary(Op::Exp, a); }
    int tanh(int a) { return unary(Op::Tanh, a); }
    int relu(int a) { return unary(Op::Relu, a); }
    int rsqrt(int a) { return unary(Op::Rsqrt, a); }
    int add(int a, int b) { return binary(Op::Add, a, b); }
    int sub(int a, int b) { return binary(Op::Sub, a, b); }
    int mul(int a, int b) { return binary(Op::Mul, a, b); }
    int div(int a, int b) { return binary(Op::Div, a, b); }
    int max(int a, int b) { return binary(Op::Max, a, b); }
    int row_sum(int a) { return reduce(Op::RowSum, a); }
    int row_max(int a) { return reduce(Op::RowMax, a); }
    int matmul(int a, int b);
    // Generic constructor used by rewrites: checks and infers the shape.
    int add_node(Op op, std::vector<int> srcs, double value = 0, const std::string& name = "");

    void set_output(int id) { output_ = id; }
    int output() const { return output_; }
    const Node& node(int id) const { return nodes_.at(id); }
    int size() const { return static_cast<int>(nodes_.size()); }
    std::vector<int> inputs() const;  // ids of Input nodes, in creation order

    // Consumers of each node, counting every edge (x*x lists the mul twice).
    std::vector<std::vector<int>> consumers() const;
    // One-line-per-node dump, e.g. "n3 = Mul(n1, n2) [64x128]".
    std::string to_string() const;
    // Canonical text of the DAG reachable from the output, independent of node
    // numbering but sensitive to sharing (a node used twice vs two copies).
    // Used to deduplicate rewritten graphs.
    std::string canonical() const;

private:
    int unary(Op op, int a) { return add_node(op, {a}); }
    int binary(Op op, int a, int b) { return add_node(op, {a, b}); }
    int reduce(Op op, int a) { return add_node(op, {a}); }
    std::vector<Node> nodes_;
    int output_ = -1;
};

// Shape of a broadcasting binary op; throws std::invalid_argument if incompatible.
Shape broadcast(Shape a, Shape b);

}  // namespace tc
