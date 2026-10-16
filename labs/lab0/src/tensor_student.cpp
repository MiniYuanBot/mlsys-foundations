// Part A: strided tensor core.  Fill in every TODO(A*) below.
//
// Run the tests for this part with:
//   ctest --test-dir build/lab0 -L partA --output-on-failure
// or directly: build/lab0/test_a_tensor [name-filter]
//
// Rules of thumb:
//   * Views never copy: they return a Tensor that shares storage_ and only
//     changes shape / strides / offset.
//   * Throw std::out_of_range for bad indices / dims, std::invalid_argument
//     for bad shapes.  The tests check the exception type.
#include "mt/tensor.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mt {

[[maybe_unused]] static std::logic_error not_implemented(const char *what) {
    return std::logic_error(std::string("TODO ") + what + " is not implemented yet");
}

// Normalise a possibly negative dim (-1 means last) and range-check it.
[[maybe_unused]] static int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("dim out of range");
    return dim;
}

int64_t numel_of(const Shape &shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// TODO(A1) contiguous_strides
// The last dimension moves fastest: stride[-1] = 1,
// stride[d] = stride[d+1] * shape[d+1].
// Example: {2,3,4} -> {12,4,1};  {5} -> {1};  {} -> {}.
Strides contiguous_strides(const Shape &shape) {
    (void)shape;
    throw not_implemented("A1 contiguous_strides");
}

// TODO(A2) offset_of
// 1. index.size() must equal ndim(), otherwise std::out_of_range.
// 2. Each index[d] must satisfy 0 <= index[d] < shape[d], otherwise std::out_of_range.
//    (No negative indexing here; keep it simple.)
// 3. Return offset_ + sum(index[d] * strides_[d]).
// Worked example: shape {2,3}, strides {3,1}, offset 0: (1,2) -> 1*3 + 2*1 = 5.
// After transpose: shape {3,2}, strides {1,3}: (2,1) -> 2*1 + 1*3 = 5 (same element!).
int64_t Tensor::offset_of(const Index &index) const {
    (void)index;
    throw not_implemented("A2 offset_of");
}

// TODO(A3) is_contiguous
// True iff walking the logical elements in row-major order visits storage
// positions offset_, offset_+1, offset_+2, ... with no gaps.
// Hint: compare strides_ with contiguous_strides(shape_), BUT dimensions of
// size 1 can have any stride (you never step along them), and a tensor with
// zero elements is contiguous.
bool Tensor::is_contiguous() const {
    throw not_implemented("A3 is_contiguous");
}

// TODO(A4) permute and transpose
// permute(order): new shape[i] = shape[order[i]], new strides[i] = strides[order[i]].
// order must be a permutation of 0..ndim-1 (else std::invalid_argument).
// transpose(a, b) is permute with dims a and b swapped; accept negative dims.
Tensor Tensor::permute(const std::vector<int64_t> &order) const {
    (void)order;
    throw not_implemented("A4 permute");
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    (void)dim0;
    (void)dim1;
    throw not_implemented("A4 transpose");
}

// TODO(A5) slice
// Python-style x[start:stop:step] along one dim, step >= 1.
// * dim may be negative; out of range -> std::out_of_range.
// * step < 1 -> std::invalid_argument.
// * Clamp start and stop into [0, shape[dim]] (like Python); if stop <= start
//   the result has size 0 along dim.
// * New size  = ceil((stop - start) / step).
// * New offset = offset_ + start * strides_[dim];  new stride = strides_[dim] * step.
//   (For an empty result keep offset_ unchanged.)
// Example: arange(10).slice(0, 2, 9, 3) -> elements {2, 5, 8}, stride 3, offset 2.
Tensor Tensor::slice(int64_t dim, int64_t start, int64_t stop, int64_t step) const {
    (void)dim;
    (void)start;
    (void)stop;
    (void)step;
    throw not_implemented("A5 slice");
}

// TODO(A6) broadcast_shapes and broadcast_to
// Align shapes from the RIGHT. For each aligned pair (x, y):
//   x == y -> x;  x == 1 -> y;  y == 1 -> x;  else incompatible.
// Missing leading dims count as 1.  {3,1,5} + {4,5} -> {3,4,5}.
Shape broadcast_shapes(const Shape &a, const Shape &b) {
    (void)a;
    (void)b;
    throw not_implemented("A6 broadcast_shapes");
}

// broadcast_to(target): a VIEW with shape == target.
// * New leading dims get stride 0.
// * A dim of size 1 being stretched to size n gets stride 0
//   (every index along it reads the same storage element).
// * A dim that already matches keeps its stride.
// * Anything else -> std::invalid_argument.
Tensor Tensor::broadcast_to(const Shape &target) const {
    (void)target;
    throw not_implemented("A6 broadcast_to");
}

// TODO(A7) contiguous
// If already contiguous, return *this (no copy, shares storage).
// Otherwise allocate Tensor::zeros(shape_) and copy every logical element
// in row-major order. for_each_index(shape_, ...) visits indices in exactly
// that order, so the k-th visited element goes to out.data()[k].
Tensor Tensor::contiguous() const {
    throw not_implemented("A7 contiguous");
}

// TODO(A8) zip: elementwise binary op with broadcasting
// 1. shape = broadcast_shapes(a.shape(), b.shape())
// 2. view both inputs with broadcast_to(shape)  (no copies!)
// 3. out = Tensor::zeros(shape); fill out in row-major order with fn(a[idx], b[idx]).
// Look at map() in tensor_core.cpp for the pattern.
Tensor zip(const Tensor &a, const Tensor &b, const BinaryFn &fn) {
    (void)a;
    (void)b;
    (void)fn;
    throw not_implemented("A8 zip");
}

// TODO(A9) sum over one dim
// Output shape = input shape with `dim` removed (keepdim=false) or set to 1
// (keepdim=true). out[..., (dim), ...] = sum over j of x[..., j, ...].
// Hint: compute with keepdim=true first (the output index is the input index
// with idx[dim] set to 0), then .view() away the dim if keepdim is false.
// Accumulate in double for accuracy, store as float. dim may be negative.
Tensor sum(const Tensor &x, int64_t dim, bool keepdim) {
    (void)x;
    (void)dim;
    (void)keepdim;
    throw not_implemented("A9 sum");
}

} // namespace mt
