// Provided part of the lazy engine: graph construction, the unfused
// scheduler, traffic accounting, the scalar interpreter and the executor.
#include "mt/lazy.hpp"

#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mt::lazy {

bool is_leaf(Op op) {
    return op == Op::Input || op == Op::Const;
}
bool is_unary(Op op) {
    return op == Op::Neg || op == Op::Exp || op == Op::Tanh || op == Op::Relu;
}
bool is_binary(Op op) {
    return op == Op::Add || op == Op::Sub || op == Op::Mul || op == Op::Div;
}
bool is_elementwise(Op op) {
    return is_unary(op) || is_binary(op);
}

const char* op_name(Op op) {
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
    case Op::Add:
        return "Add";
    case Op::Sub:
        return "Sub";
    case Op::Mul:
        return "Mul";
    case Op::Div:
        return "Div";
    case Op::SumLast:
        return "SumLast";
    }
    return "?";
}

float apply_unary(Op op, float x) {
    switch (op) {
    case Op::Neg:
        return -x;
    case Op::Exp:
        return std::exp(x);
    case Op::Tanh:
        return std::tanh(x);
    case Op::Relu:
        return x > 0.f ? x : 0.f;
    default:
        throw std::logic_error("apply_unary: not a unary op");
    }
}

float apply_binary(Op op, float a, float b) {
    switch (op) {
    case Op::Add:
        return a + b;
    case Op::Sub:
        return a - b;
    case Op::Mul:
        return a * b;
    case Op::Div:
        return a / b;
    default:
        throw std::logic_error("apply_binary: not a binary op");
    }
}

// ---------------------------------------------------------------- graph ----
static int next_id = 0;

static NodePtr make(Op op, std::vector<NodePtr> srcs, Shape shape) {
    auto n = std::make_shared<Node>();
    n->op = op;
    n->srcs = std::move(srcs);
    n->shape = std::move(shape);
    n->id = next_id++;
    return n;
}

LazyTensor LazyTensor::input(const Tensor& t) {
    auto n = make(Op::Input, {}, t.shape());
    n->value = t.contiguous();
    return LazyTensor(n);
}

LazyTensor LazyTensor::constant(float value) {
    auto n = make(Op::Const, {}, {});
    n->scalar = value;
    return LazyTensor(n);
}

static LazyTensor unary(Op op, const LazyTensor& x) {
    return LazyTensor(make(op, {x.node()}, x.shape()));
}

static LazyTensor binary(Op op, const LazyTensor& a, const LazyTensor& b) {
    const bool ac = a.node()->op == Op::Const, bc = b.node()->op == Op::Const;
    Shape shape;
    if (ac)
        shape = b.shape();
    else if (bc)
        shape = a.shape();
    else if (a.shape() == b.shape())
        shape = a.shape();
    else
        throw std::invalid_argument("lazy: binary op needs equal shapes or a scalar constant");
    return LazyTensor(make(op, {a.node(), b.node()}, shape));
}

LazyTensor LazyTensor::neg() const {
    return unary(Op::Neg, *this);
}
LazyTensor LazyTensor::exp() const {
    return unary(Op::Exp, *this);
}
LazyTensor LazyTensor::tanh() const {
    return unary(Op::Tanh, *this);
}
LazyTensor LazyTensor::relu() const {
    return unary(Op::Relu, *this);
}
LazyTensor LazyTensor::sum_last() const {
    if (shape().empty()) throw std::invalid_argument("sum_last on a 0-d tensor");
    Shape s(shape().begin(), shape().end() - 1);
    return LazyTensor(make(Op::SumLast, {node_}, s));
}
LazyTensor operator+(const LazyTensor& a, const LazyTensor& b) {
    return binary(Op::Add, a, b);
}
LazyTensor operator-(const LazyTensor& a, const LazyTensor& b) {
    return binary(Op::Sub, a, b);
}
LazyTensor operator*(const LazyTensor& a, const LazyTensor& b) {
    return binary(Op::Mul, a, b);
}
LazyTensor operator/(const LazyTensor& a, const LazyTensor& b) {
    return binary(Op::Div, a, b);
}
LazyTensor operator*(float s, const LazyTensor& a) {
    return LazyTensor::constant(s) * a;
}
LazyTensor operator+(float s, const LazyTensor& a) {
    return LazyTensor::constant(s) + a;
}
LazyTensor operator+(const LazyTensor& a, float s) {
    return a + LazyTensor::constant(s);
}
LazyTensor operator-(const LazyTensor& a, float s) {
    return a - LazyTensor::constant(s);
}
LazyTensor operator*(const LazyTensor& a, float s) {
    return a * LazyTensor::constant(s);
}
LazyTensor operator/(const LazyTensor& a, float s) {
    return a / LazyTensor::constant(s);
}

// ----------------------------------------------------------- scheduling ----
std::vector<Kernel> schedule_unfused(const NodePtr& root) {
    std::vector<Kernel> kernels;
    for (const auto& n : toposort(root))
        if (!is_leaf(n->op)) kernels.push_back(Kernel{n, {n}});
    return kernels;
}

static int64_t bytes_of(const Shape& s) {
    return numel_of(s) * static_cast<int64_t>(sizeof(float));
}

ExecStats analyze(const std::vector<Kernel>& kernels, const NodePtr& root) {
    ExecStats st;
    st.kernels = static_cast<int>(kernels.size());
    for (const auto& k : kernels) {
        std::set<const Node*> inside, reads;
        for (const auto& n : k.nodes) inside.insert(n.get());
        for (const auto& n : k.nodes)
            for (const auto& s : n->srcs)
                if (!inside.count(s.get()) && s->op != Op::Const && reads.insert(s.get()).second)
                    st.bytes_read += bytes_of(s->shape);
        st.bytes_written += bytes_of(k.output->shape);
        if (k.output != root) st.intermediate_bytes += bytes_of(k.output->shape);
    }
    return st;
}

std::string describe(const std::vector<Kernel>& kernels) {
    std::ostringstream os;
    for (size_t i = 0; i < kernels.size(); ++i) {
        const auto& k = kernels[i];
        os << "kernel " << i << " -> n" << k.output->id << " " << op_name(k.output->op) << " [";
        for (size_t d = 0; d < k.output->shape.size(); ++d)
            os << (d ? "," : "") << k.output->shape[d];
        os << "]  nodes:";
        std::set<const Node*> inside;
        for (const auto& n : k.nodes) {
            os << " n" << n->id << " " << op_name(n->op);
            inside.insert(n.get());
        }
        os << "  reads:";
        std::set<int> reads;
        for (const auto& n : k.nodes)
            for (const auto& s : n->srcs)
                if (!inside.count(s.get()) && s->op != Op::Const) reads.insert(s->id);
        for (int id : reads) os << " n" << id;
        os << "\n";
    }
    return os.str();
}

// --------------------------------------------------------- interpreters ----
float eval_scalar(const Program& p, const std::vector<const float*>& inputs, int64_t i) {
    float stack[64];
    int sp = 0;
    for (const auto& ins : p.code) {
        switch (ins.kind) {
        case InstrKind::Load:
            stack[sp++] = inputs[ins.buffer][i];
            break;
        case InstrKind::Const:
            stack[sp++] = ins.value;
            break;
        case InstrKind::Unary:
            stack[sp - 1] = apply_unary(ins.op, stack[sp - 1]);
            break;
        case InstrKind::Binary:
            --sp;
            stack[sp - 1] = apply_binary(ins.op, stack[sp - 1], stack[sp]);
            break;
        }
        if (sp >= 64) throw std::runtime_error("eval_scalar: expression too deep");
    }
    if (sp != 1)
        throw std::runtime_error("eval_scalar: malformed program (stack depth != 1 at end)");
    return stack[0];
}

// ------------------------------------------------------------ execution ----
Tensor execute(const std::vector<Kernel>& kernels, const NodePtr& root, Interp interp,
               ExecStats* stats) {
    std::unordered_map<const Node*, Tensor> buffers;
    for (const auto& n : toposort(root))
        if (n->op == Op::Input) buffers[n.get()] = n->value;

    std::vector<float> tile(kTile);
    for (const auto& k : kernels) {
        const Program p = compile(k);
        std::vector<const float*> inputs;
        for (const auto& b : p.buffers) {
            auto it = buffers.find(b.get());
            if (it == buffers.end())
                throw std::logic_error("execute: kernel reads n" + std::to_string(b->id) +
                                       " before any kernel produced it");
            inputs.push_back(it->second.data());
        }
        Tensor out = Tensor::zeros(k.output->shape);
        float* dst = out.data();
        const int64_t n_out = out.numel();
        if (!p.reduce) {
            if (interp == Interp::Scalar) {
                for (int64_t i = 0; i < n_out; ++i) dst[i] = eval_scalar(p, inputs, i);
            } else {
                for (int64_t i = 0; i < n_out; i += kTile) {
                    const int n = static_cast<int>(std::min<int64_t>(kTile, n_out - i));
                    eval_tile(p, inputs, i, n, dst + i);
                }
            }
        } else {
            const int64_t L = p.reduce_len;
            for (int64_t r = 0; r < n_out; ++r) {
                double acc = 0.0;
                if (interp == Interp::Scalar) {
                    for (int64_t j = 0; j < L; ++j) acc += eval_scalar(p, inputs, r * L + j);
                } else {
                    for (int64_t j = 0; j < L; j += kTile) {
                        const int n = static_cast<int>(std::min<int64_t>(kTile, L - j));
                        eval_tile(p, inputs, r * L + j, n, tile.data());
                        for (int t = 0; t < n; ++t) acc += tile[t];
                    }
                }
                dst[r] = static_cast<float>(acc);
            }
        }
        buffers[k.output.get()] = out;
    }
    if (stats) *stats = analyze(kernels, root);
    if (root->op == Op::Input) return root->value;
    if (root->op == Op::Const) return Tensor::full({}, root->scalar);
    return buffers.at(root.get());
}

Tensor realize_unfused(const LazyTensor& t, ExecStats* stats) {
    return execute(schedule_unfused(t.node()), t.node(), Interp::Scalar, stats);
}

Tensor realize_fused(const LazyTensor& t, Interp interp, ExecStats* stats) {
    return execute(schedule_fused(t.node()), t.node(), interp, stats);
}

} // namespace mt::lazy
