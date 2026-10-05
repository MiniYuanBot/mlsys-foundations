// minitensor: a tiny strided tensor library for lab0.
//
// A Tensor is "storage + view metadata":
//   storage : shared, flat std::vector<float>
//   shape   : logical size of each dimension
//   strides : how many storage elements to skip when that index grows by 1
//   offset  : storage index of logical element (0, 0, ..., 0)
//
// The storage index of logical element (i0, i1, ...) is
//   offset + i0 * strides[0] + i1 * strides[1] + ...
//
// View operations (transpose, permute, slice, broadcast_to, view) only change
// metadata and share storage. contiguous() is the only operation that copies.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mt {

using Shape = std::vector<int64_t>;
using Strides = std::vector<int64_t>;
using Index = std::vector<int64_t>;

// ---- Part A helpers (you implement these in src/tensor_student.cpp) --------

// Row-major ("C order") strides for a shape. {2,3,4} -> {12,4,1}.
Strides contiguous_strides(const Shape& shape);

// NumPy broadcasting rule for two shapes. {3,1,5} and {4,5} -> {3,4,5}.
// Throws std::invalid_argument if the shapes are incompatible.
Shape broadcast_shapes(const Shape& a, const Shape& b);

// Product of dimensions (1 for a 0-d shape).
int64_t numel_of(const Shape& shape);

class Tensor {
public:
    // ---- construction (provided) ----
    Tensor();  // 0-d tensor holding 0.0f
    static Tensor zeros(const Shape& shape);
    static Tensor full(const Shape& shape, float value);
    static Tensor from_vector(const Shape& shape, std::vector<float> values);
    static Tensor arange(int64_t n);  // 1-d {0, 1, ..., n-1}

    // Low-level constructor: wrap existing storage with explicit metadata.
    Tensor(std::shared_ptr<std::vector<float>> storage, Shape shape, Strides strides,
           int64_t offset);

    // ---- metadata (provided) ----
    const Shape& shape() const { return shape_; }
    const Strides& strides() const { return strides_; }
    int64_t offset() const { return offset_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_of(shape_); }
    int64_t size(int64_t dim) const;
    const std::shared_ptr<std::vector<float>>& storage() const { return storage_; }
    bool shares_storage_with(const Tensor& other) const { return storage_ == other.storage_; }
    // Pointer to logical element (0,...,0). Only meaningful for contiguous tensors
    // when you want to treat the data as a flat array.
    const float* data() const { return storage_->data() + offset_; }
    float* data() { return storage_->data() + offset_; }

    // ---- element access ----
    // Part A2: storage index of a logical index, with bounds checking.
    int64_t offset_of(const Index& index) const;
    float at(const Index& index) const { return (*storage_)[offset_of(index)]; }
    float& at(const Index& index) { return (*storage_)[offset_of(index)]; }

    // ---- views: metadata only, never copy (Part A3-A6) ----
    bool is_contiguous() const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(const std::vector<int64_t>& order) const;
    Tensor slice(int64_t dim, int64_t start, int64_t stop, int64_t step = 1) const;
    Tensor broadcast_to(const Shape& target) const;
    // Reinterpret a contiguous tensor with a new shape (provided; uses is_contiguous).
    Tensor view(const Shape& new_shape) const;

    // ---- copies ----
    Tensor contiguous() const;                     // Part A7
    Tensor reshape(const Shape& new_shape) const;  // provided: view if possible, else copy

    // Human-readable dump of metadata plus the storage grid (provided).
    std::string describe() const;
    // All elements in logical row-major order (provided; handy in tests).
    std::vector<float> to_vector() const;

private:
    std::shared_ptr<std::vector<float>> storage_;
    Shape shape_;
    Strides strides_;
    int64_t offset_ = 0;
};

// Visit every logical index of `shape` in row-major order (provided).
void for_each_index(const Shape& shape, const std::function<void(const Index&)>& fn);

// ---- operators ----
using UnaryFn = std::function<float(float)>;
using BinaryFn = std::function<float(float, float)>;

Tensor map(const Tensor& x, const UnaryFn& fn);                         // provided
Tensor zip(const Tensor& a, const Tensor& b, const BinaryFn& fn);       // Part A8
Tensor sum(const Tensor& x, int64_t dim, bool keepdim = false);         // Part A9

inline Tensor operator+(const Tensor& a, const Tensor& b) { return zip(a, b, [](float x, float y) { return x + y; }); }
inline Tensor operator-(const Tensor& a, const Tensor& b) { return zip(a, b, [](float x, float y) { return x - y; }); }
inline Tensor operator*(const Tensor& a, const Tensor& b) { return zip(a, b, [](float x, float y) { return x * y; }); }
inline Tensor operator/(const Tensor& a, const Tensor& b) { return zip(a, b, [](float x, float y) { return x / y; }); }

// 2-D matmul on top of the GEMM kernels in gemm.hpp (provided).
Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace mt
