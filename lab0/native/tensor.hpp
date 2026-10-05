#pragma once
#include <cstddef>
#include <vector>

struct View {
    const float* data;
    std::size_t rows, cols, row_stride, col_stride, storage_size;
};
// Read-only, positive strides. Return one logical element, with bounds checking.
float element(const View& view, std::size_t row, std::size_t column);
// Return contiguous row-major A@B. Output storage must not alias either input.
std::vector<float> matmul(const View& a, const View& b);

