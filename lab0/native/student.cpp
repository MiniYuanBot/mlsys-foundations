#include "tensor.hpp"
#include <stdexcept>

float element(const View& view, std::size_t row, std::size_t column) {
    // TODO S00-1: logical bounds, checked offset using both strides, storage bounds.
    // The tests include transposed and padded views. Do not assume cols==row_stride.
    (void)view; (void)row; (void)column;
    throw std::logic_error("TODO S00-1: strided addressing");
}

std::vector<float> matmul(const View& a, const View& b) {
    // TODO S00-2: validate shapes; implement a correct layout-aware matmul.
    // First implement an easy version. Then keep it as baseline and add one
    // loop-order or packing optimization, measured on development shapes.
    (void)a; (void)b;
    throw std::logic_error("TODO S00-2: native tensor operator");
}

