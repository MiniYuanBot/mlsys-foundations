// Internals shared by codegen_core.cpp (provided) and codegen_student.cpp.
#pragma once

#include "tc/compiler.hpp"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace tc::detail {

// Everything codegen needs to know about one kernel.
struct Ctx {
    const Graph& g;
    const Kernel& k;
    std::map<int, std::string> buf; // materialised operand id -> "in0", "in1", ...
    std::set<int> inside;           // ids of k.nodes
};
Ctx make_ctx(const Graph& g, const Kernel& k);

// Float literal, e.g. literal(0.5) == "0.5f".
std::string literal(double v);
// The C expression for operand `id` as seen from inside the kernel:
//   computed inside this kernel -> "v<id>"     (an SSA register)
//   Const                       -> literal
//   materialised (a read)       -> emit_load(buf, shape)   (uses loop variables i, j)
std::string operand(const Ctx& c, int id);
// For every node in `nodes` (in order) append "<indent>const float v<id> = <emit_op(...)>;\n".
// The nodes must be elementwise; their operands must already be available.
void emit_ssa(const Ctx& c, const std::vector<int>& nodes, std::ostringstream& os,
              const std::string& indent);
// "extern \"C\" __global__ void tc_kernel(const float* __restrict__ in0, ..., float* __restrict__ out, <dims>)"
std::string signature(const Ctx& c, const std::string& dims);
// Device helpers available to every kernel: tc_block_sum / tc_block_max
// (block-wide reductions; every thread gets the result).
extern const char* kPrelude;

std::string codegen_elementwise(const Ctx& c); // provided (uses A3)
std::string codegen_row(const Ctx& c);         // B2
std::string codegen_matmul(const Ctx& c);      // provided template (uses B3)

// B3: the generated pieces spliced into the GEMM template.
struct MatmulHooks {
    std::string load_a; // body of `float tc_load_a(<ins>, long i, long j)`: returns A'(i, j)
    std::string load_b; // body of `float tc_load_b(<ins>, long i, long j)`: returns B(i, j)
    std::string
        epilogue; // body of `float tc_epilogue(float acc, <ins>, long i, long j)`: returns out(i, j)
};
MatmulHooks matmul_hooks(const Ctx& c);

} // namespace tc::detail
