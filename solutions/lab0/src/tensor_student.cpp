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

[[maybe_unused]] static std::logic_error not_implemented(const char* what) {
    return std::logic_error(std::string("TODO ") + what + " is not implemented yet");
}

// Normalise a possibly negative dim (-1 means last) and range-check it.
[[maybe_unused]] static int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("dim out of range");
    return dim;
}

int64_t numel_of(const Shape& shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// TODO(A1) contiguous_strides
// The last dimension moves fastest: stride[-1] = 1,
// stride[d] = stride[d+1] * shape[d+1].
// Example: {2,3,4} -> {12,4,1};  {5} -> {1};  {} -> {}.
Strides contiguous_strides(const Shape& shape) {
    Strides strides(shape.size());
    int64_t running = 1;
    for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
        strides[d] = running;
        running *= shape[d];
    }
    return strides;
}

// TODO(A2) offset_of
// 1. index.size() must equal ndim(), otherwise std::out_of_range.
// 2. Each index[d] must satisfy 0 <= index[d] < shape[d], otherwise std::out_of_range.
//    (No negative indexing here; keep it simple.)
// 3. Return offset_ + sum(index[d] * strides_[d]).
// Worked example: shape {2,3}, strides {3,1}, offset 0: (1,2) -> 1*3 + 2*1 = 5.
// After transpose: shape {3,2}, strides {1,3}: (2,1) -> 2*1 + 1*3 = 5 (same element!).
int64_t Tensor::offset_of(const Index& index) const {
    if (static_cast<int64_t>(index.size()) != ndim())
        throw std::out_of_range("offset_of: wrong number of indices");
    int64_t pos = offset_;
    for (size_t d = 0; d < index.size(); ++d) {
        if (index[d] < 0 || index[d] >= shape_[d]) throw std::out_of_range("offset_of: index out of range");
        pos += index[d] * strides_[d];
    }
    return pos;
}

// TODO(A3) is_contiguous
// True iff walking the logical elements in row-major order visits storage
// positions offset_, offset_+1, offset_+2, ... with no gaps.
// Hint: compare strides_ with contiguous_strides(shape_), BUT dimensions of
// size 1 can have any stride (you never step along them), and a tensor with
// zero elements is contiguous.
bool Tensor::is_contiguous() const {
    if (numel() == 0) return true;
    int64_t expected = 1;
    for (int64_t d = ndim() - 1; d >= 0; --d) {
        if (shape_[d] == 1) continue;
        if (strides_[d] != expected) return false;
        expected *= shape_[d];
    }
    return true;
}

// TODO(A4) permute and transpose
// permute(order): new shape[i] = shape[order[i]], new strides[i] = strides[order[i]].
// order must be a permutation of 0..ndim-1 (else std::invalid_argument).
// transpose(a, b) is permute with dims a and b swapped; accept negative dims.
Tensor Tensor::permute(const std::vector<int64_t>& order) const {
    if (static_cast<int64_t>(order.size()) != ndim()) throw std::invalid_argument("permute: wrong rank");
    std::vector<bool> seen(order.size(), false);
    Shape shape(order.size());
    Strides strides(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        int64_t d = order[i];
        if (d < 0 || d >= ndim() || seen[d]) throw std::invalid_argument("permute: not a permutation");
        seen[d] = true;
        shape[i] = shape_[d];
        strides[i] = strides_[d];
    }
    return Tensor(storage_, shape, strides, offset_);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    dim0 = wrap_dim(dim0, ndim());
    dim1 = wrap_dim(dim1, ndim());
    std::vector<int64_t> order(ndim());
    for (int64_t i = 0; i < ndim(); ++i) order[i] = i;
    std::swap(order[dim0], order[dim1]);
    return permute(order);
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
    dim = wrap_dim(dim, ndim());
    if (step < 1) throw std::invalid_argument("slice: step must be >= 1");
    const int64_t n = shape_[dim];
    start = std::min(std::max<int64_t>(start, 0), n);
    stop = std::min(std::max<int64_t>(stop, 0), n);
    const int64_t length = stop > start ? (stop - start + step - 1) / step : 0;
    Shape shape = shape_;
    Strides strides = strides_;
    shape[dim] = length;
    strides[dim] = strides_[dim] * step;
    const int64_t offset = length > 0 ? offset_ + start * strides_[dim] : offset_;
    return Tensor(storage_, shape, strides, offset);
}

// TODO(A6) broadcast_shapes and broadcast_to
// Align shapes from the RIGHT. For each aligned pair (x, y):
//   x == y -> x;  x == 1 -> y;  y == 1 -> x;  else incompatible.
// Missing leading dims count as 1.  {3,1,5} + {4,5} -> {3,4,5}.
Shape broadcast_shapes(const Shape& a, const Shape& b) {
    const size_t n = std::max(a.size(), b.size());
    Shape out(n);
    for (size_t i = 0; i < n; ++i) {
        const int64_t x = i < n - a.size() ? 1 : a[i - (n - a.size())];
        const int64_t y = i < n - b.size() ? 1 : b[i - (n - b.size())];
        if (x != y && x != 1 && y != 1) throw std::invalid_argument("broadcast_shapes: incompatible");
        out[i] = x == 1 ? y : x;
    }
    return out;
}

// broadcast_to(target): a VIEW with shape == target.
// * New leading dims get stride 0.
// * A dim of size 1 being stretched to size n gets stride 0
//   (every index along it reads the same storage element).
// * A dim that already matches keeps its stride.
// * Anything else -> std::invalid_argument.
Tensor Tensor::broadcast_to(const Shape& target) const {
    if (target.size() < shape_.size()) throw std::invalid_argument("broadcast_to: target rank too small");
    const size_t lead = target.size() - shape_.size();
    Strides strides(target.size(), 0);
    for (size_t i = 0; i < shape_.size(); ++i) {
        const int64_t have = shape_[i], want = target[lead + i];
        if (have == want) strides[lead + i] = strides_[i];
        else if (have == 1) strides[lead + i] = 0;
        else throw std::invalid_argument("broadcast_to: incompatible shape");
    }
    return Tensor(storage_, target, strides, offset_);
}

// TODO(A7) contiguous
// If already contiguous, return *this (no copy, shares storage).
// Otherwise allocate Tensor::zeros(shape_) and copy every logical element
// in row-major order. for_each_index(shape_, ...) visits indices in exactly
// that order, so the k-th visited element goes to out.data()[k].
Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    Tensor out = Tensor::zeros(shape_);
    float* dst = out.data();
    int64_t k = 0;
    for_each_index(shape_, [&](const Index& idx) { dst[k++] = at(idx); });
    return out;
}

// TODO(A8) zip: elementwise binary op with broadcasting
// 1. shape = broadcast_shapes(a.shape(), b.shape())
// 2. view both inputs with broadcast_to(shape)  (no copies!)
// 3. out = Tensor::zeros(shape); fill out in row-major order with fn(a[idx], b[idx]).
// Look at map() in tensor_core.cpp for the pattern.
Tensor zip(const Tensor& a, const Tensor& b, const BinaryFn& fn) {
    const Shape shape = broadcast_shapes(a.shape(), b.shape());
    const Tensor av = a.broadcast_to(shape), bv = b.broadcast_to(shape);
    Tensor out = Tensor::zeros(shape);
    float* dst = out.data();
    int64_t k = 0;
    for_each_index(shape, [&](const Index& idx) { dst[k++] = fn(av.at(idx), bv.at(idx)); });
    return out;
}

// TODO(A9) sum over one dim
// Output shape = input shape with `dim` removed (keepdim=false) or set to 1
// (keepdim=true). out[..., (dim), ...] = sum over j of x[..., j, ...].
// Hint: compute with keepdim=true first (the output index is the input index
// with idx[dim] set to 0), then .view() away the dim if keepdim is false.
// Accumulate in double for accuracy, store as float. dim may be negative.
Tensor sum(const Tensor& x, int64_t dim, bool keepdim) {
    dim = wrap_dim(dim, x.ndim());
    Shape kept = x.shape();
    kept[dim] = 1;
    std::vector<double> acc(numel_of(kept), 0.0);
    const Strides out_strides = contiguous_strides(kept);
    for_each_index(x.shape(), [&](const Index& idx) {
        int64_t pos = 0;
        for (size_t d = 0; d < idx.size(); ++d)
            if (static_cast<int64_t>(d) != dim) pos += idx[d] * out_strides[d];
        acc[pos] += x.at(idx);
    });
    Tensor out = Tensor::zeros(kept);
    for (size_t i = 0; i < acc.size(); ++i) out.data()[i] = static_cast<float>(acc[i]);
    if (keepdim) return out;
    Shape squeezed;
    for (int64_t d = 0; d < x.ndim(); ++d)
        if (d != dim) squeezed.push_back(x.shape()[d]);
    return out.view(squeezed);
}

}  // namespace mt
