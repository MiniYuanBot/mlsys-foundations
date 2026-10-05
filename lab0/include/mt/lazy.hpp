// Part C: lazy evaluation and operator fusion, a tiny version of what
// XLA / TorchInductor / tinygrad do before generating kernels.
//
//   LazyTensor ops build a DAG of Nodes; nothing is computed yet.
//   realize() = toposort -> schedule (group nodes into kernels) -> execute.
//
// Without fusion every op is a kernel that writes a full-size temporary to
// memory and the next kernel reads it back. With fusion a chain of
// elementwise ops becomes ONE loop that keeps temporaries in registers.
//
// Simplifications (keep the lab small):
//   * elementwise binary ops need equal shapes, or one side is a scalar Const;
//   * the only reduction is sum over the last dim (SumLast);
//   * all buffers are contiguous float32.
#pragma once

#include "mt/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mt::lazy {

enum class Op { Input, Const, Neg, Exp, Tanh, Relu, Add, Sub, Mul, Div, SumLast };

bool is_leaf(Op op);         // Input, Const
bool is_unary(Op op);        // Neg, Exp, Tanh, Relu
bool is_binary(Op op);       // Add, Sub, Mul, Div
bool is_elementwise(Op op);  // unary or binary
const char* op_name(Op op);
float apply_unary(Op op, float x);
float apply_binary(Op op, float a, float b);

struct Node {
    Op op;
    std::vector<std::shared_ptr<Node>> srcs;
    Shape shape;
    Tensor value;        // Input only
    float scalar = 0.f;  // Const only
    int id = 0;          // creation order, handy for debugging / stable ordering
};
using NodePtr = std::shared_ptr<Node>;

class LazyTensor {
public:
    explicit LazyTensor(NodePtr node) : node_(std::move(node)) {}
    static LazyTensor input(const Tensor& t);
    static LazyTensor constant(float value);

    const NodePtr& node() const { return node_; }
    const Shape& shape() const { return node_->shape; }

    LazyTensor neg() const;
    LazyTensor exp() const;
    LazyTensor tanh() const;
    LazyTensor relu() const;
    LazyTensor sum_last() const;

    friend LazyTensor operator+(const LazyTensor& a, const LazyTensor& b);
    friend LazyTensor operator-(const LazyTensor& a, const LazyTensor& b);
    friend LazyTensor operator*(const LazyTensor& a, const LazyTensor& b);
    friend LazyTensor operator/(const LazyTensor& a, const LazyTensor& b);

private:
    NodePtr node_;
};
LazyTensor operator+(const LazyTensor& a, const LazyTensor& b);
LazyTensor operator-(const LazyTensor& a, const LazyTensor& b);
LazyTensor operator*(const LazyTensor& a, const LazyTensor& b);
LazyTensor operator/(const LazyTensor& a, const LazyTensor& b);
LazyTensor operator*(float s, const LazyTensor& a);
LazyTensor operator+(float s, const LazyTensor& a);
LazyTensor operator+(const LazyTensor& a, float s);
LazyTensor operator-(const LazyTensor& a, float s);
LazyTensor operator*(const LazyTensor& a, float s);
LazyTensor operator/(const LazyTensor& a, float s);

// ---- scheduling ----
// A Kernel computes exactly one materialised node (`output`) into a buffer.
// `nodes` are the non-leaf nodes it evaluates, in topological order, ending
// with `output`. Every src of these nodes is either inside the kernel, a
// Const, or a buffer (an Input or the output of an EARLIER kernel).
struct Kernel {
    NodePtr output;
    std::vector<NodePtr> nodes;
};

// C1: all nodes reachable from root, every node after all of its srcs,
// each node exactly once. Visit srcs left-to-right (depth-first post-order)
// so the order is deterministic.
std::vector<NodePtr> toposort(const NodePtr& root);

// Provided: one kernel per non-leaf node (what eager PyTorch does).
std::vector<Kernel> schedule_unfused(const NodePtr& root);

// C2: fused schedule. A non-leaf node is MATERIALISED (gets its own buffer and
// is the output of a kernel) iff
//   (a) it is the root, or
//   (b) it is a SumLast, or
//   (c) it has two or more consumers among the nodes reachable from root
//       (count each edge: x*x gives x two uses).
// Every other non-leaf node is inlined into the kernel of its single consumer.
// Kernels appear in the topological order of their output nodes.
std::vector<Kernel> schedule_fused(const NodePtr& root);

// ---- code generation: a postfix (stack machine) program per kernel ----
enum class InstrKind { Load, Const, Unary, Binary };
struct Instr {
    InstrKind kind;
    Op op = Op::Const;  // for Unary / Binary
    int buffer = -1;    // for Load: index into Program::buffers
    float value = 0.f;  // for Const
};
struct Program {
    std::vector<Instr> code;
    std::vector<NodePtr> buffers;  // nodes whose materialised values are loaded
    bool reduce = false;           // output is SumLast: code computes its src
    int64_t reduce_len = 1;        // size of the reduced (last) dim
};

// C3: compile a kernel to a postfix program. Walk the expression tree from
// kernel.output (or from its src when the output is SumLast) in post-order:
//   - a node inside this kernel     -> emit its children, then Unary/Binary
//   - a Const leaf                  -> Const
//   - anything else (Input or a node materialised by another kernel) -> Load
// Reuse the same buffer index if the same node is loaded twice.
Program compile(const Kernel& kernel);

// Provided interpreters. `inputs[b]` points at the data of Program::buffers[b].
// Scalar: evaluate the program for element i, one float at a time.
float eval_scalar(const Program& p, const std::vector<const float*>& inputs, int64_t i);
// C4: tiled interpreter. Evaluate elements [begin, begin+n) (n <= kTile)
// instruction by instruction on whole tiles, writing the results to out[0..n).
constexpr int kTile = 512;
void eval_tile(const Program& p, const std::vector<const float*>& inputs, int64_t begin, int n, float* out);

struct ExecStats {
    int kernels = 0;
    int64_t bytes_read = 0;          // each distinct buffer read by a kernel counts once per kernel
    int64_t bytes_written = 0;       // every kernel output buffer
    int64_t intermediate_bytes = 0;  // buffers allocated for non-root kernel outputs
};
// Provided: memory-traffic accounting of a schedule (no execution needed).
ExecStats analyze(const std::vector<Kernel>& kernels, const NodePtr& root);

enum class Interp { Scalar, Tiled };

// Provided: run the kernels in order and return the root's value.
Tensor execute(const std::vector<Kernel>& kernels, const NodePtr& root, Interp interp = Interp::Scalar,
               ExecStats* stats = nullptr);

// Convenience wrappers.
Tensor realize_unfused(const LazyTensor& t, ExecStats* stats = nullptr);
Tensor realize_fused(const LazyTensor& t, Interp interp = Interp::Scalar, ExecStats* stats = nullptr);

// Pretty-print a schedule, e.g.
//   kernel 0 -> n4 Relu [1024]  nodes: n2 Mul, n3 Add, n4 Relu  reads: n0 n1 n5
std::string describe(const std::vector<Kernel>& kernels);

}  // namespace mt::lazy
