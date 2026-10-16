// Provided: graph construction, shape inference, printing.
#include "tc/ir.hpp"

#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

namespace tc {

bool is_unary(Op op) {
    return op == Op::Neg || op == Op::Exp || op == Op::Tanh || op == Op::Relu || op == Op::Rsqrt;
}
bool is_binary(Op op) {
    return op == Op::Add || op == Op::Sub || op == Op::Mul || op == Op::Div || op == Op::Max;
}
bool is_elementwise(Op op) {
    return is_unary(op) || is_binary(op);
}
bool is_reduction(Op op) {
    return op == Op::RowSum || op == Op::RowMax;
}

const char *op_name(Op op) {
    switch (op) {
    case Op::Input:
        return "Input";
    case Op::Const:
        return "Const";
    case Op::Neg:
        return "Neg";
    case Op::Exp:
        return "Exp";
    case Op::Tanh:
        return "Tanh";
    case Op::Relu:
        return "Relu";
    case Op::Rsqrt:
        return "Rsqrt";
    case Op::Add:
        return "Add";
    case Op::Sub:
        return "Sub";
    case Op::Mul:
        return "Mul";
    case Op::Div:
        return "Div";
    case Op::Max:
        return "Max";
    case Op::RowSum:
        return "RowSum";
    case Op::RowMax:
        return "RowMax";
    case Op::MatMul:
        return "MatMul";
    }
    return "?";
}

Shape broadcast(Shape a, Shape b) {
    auto dim = [](int64_t x, int64_t y) {
        if (x == y || y == 1) return x;
        if (x == 1) return y;
        throw std::invalid_argument("broadcast: incompatible dims " + std::to_string(x) + " vs " +
                                    std::to_string(y));
    };
    return {dim(a.rows, b.rows), dim(a.cols, b.cols)};
}

int Graph::input(const std::string &name, Shape shape) {
    Node n{Op::Input, {}, shape, name, 0};
    nodes_.push_back(n);
    return size() - 1;
}

int Graph::constant(double value) {
    nodes_.push_back(Node{Op::Const, {}, {1, 1}, "", value});
    return size() - 1;
}

int Graph::matmul(int a, int b) {
    return add_node(Op::MatMul, {a, b});
}

int Graph::add_node(Op op, std::vector<int> srcs, double value, const std::string &name) {
    for (int s : srcs)
        if (s < 0 || s >= size()) throw std::out_of_range("add_node: bad operand id");
    Shape shape;
    if (op == Op::Input) {
        throw std::invalid_argument("use Graph::input");
    } else if (op == Op::Const) {
        shape = {1, 1};
    } else if (is_unary(op) || is_reduction(op)) {
        if (srcs.size() != 1) throw std::invalid_argument("unary op needs 1 operand");
        shape = node(srcs[0]).shape;
        if (is_reduction(op)) shape.cols = 1;
    } else if (is_binary(op)) {
        if (srcs.size() != 2) throw std::invalid_argument("binary op needs 2 operands");
        shape = broadcast(node(srcs[0]).shape, node(srcs[1]).shape);
    } else if (op == Op::MatMul) {
        if (srcs.size() != 2) throw std::invalid_argument("matmul needs 2 operands");
        const Shape a = node(srcs[0]).shape, b = node(srcs[1]).shape;
        if (a.cols != b.rows) throw std::invalid_argument("matmul: inner dims differ");
        shape = {a.rows, b.cols};
    }
    nodes_.push_back(Node{op, std::move(srcs), shape, name, value});
    return size() - 1;
}

std::vector<int> Graph::inputs() const {
    std::vector<int> ids;
    for (int i = 0; i < size(); ++i)
        if (nodes_[i].op == Op::Input) ids.push_back(i);
    return ids;
}

std::vector<std::vector<int>> Graph::consumers() const {
    std::vector<std::vector<int>> c(nodes_.size());
    for (int i = 0; i < size(); ++i)
        for (int s : nodes_[i].srcs) c[s].push_back(i);
    return c;
}

std::string Graph::to_string() const {
    std::ostringstream os;
    for (int i = 0; i < size(); ++i) {
        const Node &n = nodes_[i];
        os << "n" << i << " = " << op_name(n.op) << "(";
        if (n.op == Op::Input) os << n.name;
        if (n.op == Op::Const) os << n.value;
        for (size_t k = 0; k < n.srcs.size(); ++k) os << (k ? ", " : "") << "n" << n.srcs[k];
        os << ") [" << n.shape.rows << "x" << n.shape.cols << "]"
           << (i == output_ ? "  <- output" : "") << "\n";
    }
    return os.str();
}

std::string Graph::canonical() const {
    // Depth-first from the output. The first visit of a node prints it in full
    // and gives it a label; later visits print only "#label". So a shared
    // subexpression and two separate copies of it print differently, while two
    // graphs that differ only in node numbering print the same.
    std::map<int, int> label;
    std::function<std::string(int)> go = [&](int id) -> std::string {
        auto it = label.find(id);
        if (it != label.end()) return "#" + std::to_string(it->second);
        const Node &n = nodes_[id];
        std::string s;
        if (n.op == Op::Input)
            s = n.name;
        else if (n.op == Op::Const) {
            std::ostringstream os;
            os.precision(17);
            os << n.value;
            s = os.str();
        } else {
            s = std::string(op_name(n.op)) + "(";
            for (size_t k = 0; k < n.srcs.size(); ++k) s += (k ? "," : "") + go(n.srcs[k]);
            s += ")";
        }
        const int l = static_cast<int>(label.size());
        label[id] = l;
        return s + "@" + std::to_string(l);
    };
    if (output_ < 0) throw std::logic_error("canonical: graph has no output");
    return go(output_);
}

} // namespace tc
