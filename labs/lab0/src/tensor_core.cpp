// Provided part of minitensor. Read it, but you should not need to change it.
#include "mt/gemm.hpp"
#include "mt/tensor.hpp"

#include <sstream>
#include <stdexcept>

namespace mt {

Tensor::Tensor() : Tensor(std::make_shared<std::vector<float>>(1, 0.0f), {}, {}, 0) {}

Tensor::Tensor(std::shared_ptr<std::vector<float>> storage, Shape shape, Strides strides,
               int64_t offset)
    : storage_(std::move(storage)), shape_(std::move(shape)), strides_(std::move(strides)),
      offset_(offset) {
    if (!storage_) throw std::invalid_argument("null storage");
    if (shape_.size() != strides_.size())
        throw std::invalid_argument("shape/strides rank mismatch");
    for (auto d : shape_)
        if (d < 0) throw std::invalid_argument("negative dimension");
}

Tensor Tensor::zeros(const Shape &shape) {
    return full(shape, 0.0f);
}

Tensor Tensor::full(const Shape &shape, float value) {
    auto storage = std::make_shared<std::vector<float>>(numel_of(shape), value);
    return Tensor(storage, shape, contiguous_strides(shape), 0);
}

Tensor Tensor::from_vector(const Shape &shape, std::vector<float> values) {
    if (static_cast<int64_t>(values.size()) != numel_of(shape))
        throw std::invalid_argument("from_vector: value count does not match shape");
    auto storage = std::make_shared<std::vector<float>>(std::move(values));
    return Tensor(storage, shape, contiguous_strides(shape), 0);
}

Tensor Tensor::arange(int64_t n) {
    std::vector<float> values(n);
    for (int64_t i = 0; i < n; ++i) values[i] = static_cast<float>(i);
    return from_vector({n}, std::move(values));
}

int64_t Tensor::size(int64_t dim) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) throw std::out_of_range("size: dim out of range");
    return shape_[dim];
}

Tensor Tensor::view(const Shape &new_shape) const {
    if (numel_of(new_shape) != numel()) throw std::invalid_argument("view: element count changes");
    if (!is_contiguous())
        throw std::invalid_argument(
            "view: tensor is not contiguous, call contiguous() or reshape()");
    return Tensor(storage_, new_shape, contiguous_strides(new_shape), offset_);
}

Tensor Tensor::reshape(const Shape &new_shape) const {
    return is_contiguous() ? view(new_shape) : contiguous().view(new_shape);
}

std::vector<float> Tensor::to_vector() const {
    std::vector<float> out;
    out.reserve(numel());
    for_each_index(shape_, [&](const Index &idx) { out.push_back(at(idx)); });
    return out;
}

static std::string join(const std::vector<int64_t> &v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? ", " : "") << v[i];
    os << "]";
    return os.str();
}

// Example for a transposed 2x3 tensor:
//   shape=[3, 2] strides=[1, 3] offset=0 contiguous=no
//   logical -> storage index
//       0   3
//       1   4
//       2   5
//   storage: [#0 #1 #2 #3 #4 #5]   (# = reachable from this view)
std::string Tensor::describe() const {
    std::ostringstream os;
    os << "shape=" << join(shape_) << " strides=" << join(strides_) << " offset=" << offset_
       << " contiguous=" << (is_contiguous() ? "yes" : "no") << "\n";
    std::vector<bool> reached(storage_->size(), false);
    for_each_index(shape_, [&](const Index &idx) { reached[offset_of(idx)] = true; });
    if (ndim() == 2) {
        os << "logical -> storage index\n";
        for (int64_t i = 0; i < shape_[0]; ++i) {
            for (int64_t j = 0; j < shape_[1]; ++j) {
                os.width(4);
                os << offset_of({i, j});
            }
            os << "\n";
        }
    }
    os << "storage: [";
    for (size_t i = 0; i < storage_->size(); ++i)
        os << (i ? " " : "") << (reached[i] ? "#" : ".") << i;
    os << "]   (# = reachable from this view)\n";
    return os.str();
}

void for_each_index(const Shape &shape, const std::function<void(const Index &)> &fn) {
    if (numel_of(shape) == 0) return;
    Index idx(shape.size(), 0);
    while (true) {
        fn(idx);
        // Odometer increment: bump the last dimension, carry leftwards.
        int64_t d = static_cast<int64_t>(shape.size()) - 1;
        for (; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
        if (d < 0) return;
    }
}

Tensor map(const Tensor &x, const UnaryFn &fn) {
    Tensor out = Tensor::zeros(x.shape());
    float *dst = out.data();
    int64_t k = 0;
    for_each_index(x.shape(), [&](const Index &idx) { dst[k++] = fn(x.at(idx)); });
    return out;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
    if (a.ndim() != 2 || b.ndim() != 2 || a.size(1) != b.size(0))
        throw std::invalid_argument("matmul: expected [M,K] @ [K,N]");
    // GEMM kernels want row-major operands with unit column stride.
    Tensor ac = a.contiguous(), bc = b.contiguous();
    const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    Tensor c = Tensor::zeros({M, N});
    gemm(GemmKernel::Best, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), ac.data(),
         static_cast<int>(K), bc.data(), static_cast<int>(N), c.data(), static_cast<int>(N));
    return c;
}

} // namespace mt
