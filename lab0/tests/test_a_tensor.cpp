// Part A tests. Names start with the TODO they exercise, so
//   test_a_tensor A5     runs only the slice tests.
#include "check.hpp"
#include "mt/tensor.hpp"

using namespace mt;

static Tensor iota(const Shape &shape) {
    std::vector<float> v(numel_of(shape));
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i);
    return Tensor::from_vector(shape, v);
}

TEST(A1_contiguous_strides) {
    CHECK_EQ(contiguous_strides({2, 3, 4}), (Strides{12, 4, 1}));
    CHECK_EQ(contiguous_strides({5}), (Strides{1}));
    CHECK_EQ(contiguous_strides({}), (Strides{}));
    CHECK_EQ(contiguous_strides({3, 1, 2}), (Strides{2, 2, 1}));
}

TEST(A2_offset_of_row_major) {
    Tensor t = iota({2, 3});
    CHECK_EQ(t.offset_of({0, 0}), 0);
    CHECK_EQ(t.offset_of({1, 2}), 5);
    CHECK_EQ(t.at({1, 0}), 3.0f);
}

TEST(A2_offset_of_checks_bounds) {
    Tensor t = iota({2, 3});
    CHECK_THROWS(t.offset_of({2, 0}), std::out_of_range);
    CHECK_THROWS(t.offset_of({0, 3}), std::out_of_range);
    CHECK_THROWS(t.offset_of({-1, 0}), std::out_of_range);
    CHECK_THROWS(t.offset_of({0}), std::out_of_range);
}

TEST(A2_offset_of_uses_strides_and_offset) {
    // A padded 2x3 matrix stored with row stride 4, starting at element 1.
    auto storage =
        std::make_shared<std::vector<float>>(std::vector<float>{9, 1, 2, 3, 9, 4, 5, 6, 9});
    Tensor t(storage, {2, 3}, {4, 1}, 1);
    CHECK_EQ(t.at({0, 0}), 1.0f);
    CHECK_EQ(t.at({1, 2}), 6.0f);
    CHECK_EQ(t.to_vector(), (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(A3_is_contiguous) {
    CHECK(iota({2, 3, 4}).is_contiguous());
    CHECK(Tensor::zeros({}).is_contiguous());
    auto s = std::make_shared<std::vector<float>>(12);
    CHECK(!Tensor(s, {2, 3}, {4, 1}, 0).is_contiguous()); // padded rows
    CHECK(!Tensor(s, {3, 2}, {1, 3}, 0).is_contiguous()); // transposed
    CHECK(Tensor(s, {1, 3}, {99, 1}, 0).is_contiguous()); // size-1 dim: stride irrelevant
    CHECK(Tensor(s, {0, 3}, {7, 7}, 0).is_contiguous());  // empty
}

TEST(A4_transpose_is_a_view) {
    Tensor t = iota({2, 3});
    Tensor tt = t.transpose(0, 1);
    CHECK_EQ(tt.shape(), (Shape{3, 2}));
    CHECK_EQ(tt.strides(), (Strides{1, 3}));
    CHECK(tt.shares_storage_with(t));
    CHECK(!tt.is_contiguous());
    CHECK_EQ(tt.at({2, 1}), 5.0f);
    tt.at({2, 1}) = 42.0f; // write through the view
    CHECK_EQ(t.at({1, 2}), 42.0f);
    CHECK_EQ(t.transpose(-1, -2).shape(), (Shape{3, 2}));
}

TEST(A4_permute) {
    Tensor t = iota({2, 3, 4});
    Tensor p = t.permute({2, 0, 1});
    CHECK_EQ(p.shape(), (Shape{4, 2, 3}));
    CHECK_EQ(p.strides(), (Strides{1, 12, 4}));
    CHECK_EQ(p.at({3, 1, 2}), t.at({1, 2, 3}));
    CHECK_THROWS(t.permute({0, 0, 1}), std::invalid_argument);
    CHECK_THROWS(t.permute({0, 1}), std::invalid_argument);
}

TEST(A5_slice_with_step) {
    Tensor s = Tensor::arange(10).slice(0, 2, 9, 3);
    CHECK_EQ(s.shape(), (Shape{3}));
    CHECK_EQ(s.strides(), (Strides{3}));
    CHECK_EQ(s.offset(), 2);
    CHECK_EQ(s.to_vector(), (std::vector<float>{2, 5, 8}));
}

TEST(A5_slice_columns_of_matrix) {
    Tensor t = iota({3, 4});
    Tensor c = t.slice(1, 1, 3); // columns 1..2
    CHECK_EQ(c.shape(), (Shape{3, 2}));
    CHECK_EQ(c.to_vector(), (std::vector<float>{1, 2, 5, 6, 9, 10}));
    CHECK(!c.is_contiguous());
    CHECK(c.shares_storage_with(t));
    Tensor r = t.slice(-2, 1, 100); // dim -2 = rows, stop clamped
    CHECK_EQ(r.shape(), (Shape{2, 4}));
    CHECK(r.is_contiguous());
}

TEST(A5_slice_edge_cases) {
    Tensor t = Tensor::arange(5);
    CHECK_EQ(t.slice(0, 3, 1).numel(), 0);
    CHECK_EQ(t.slice(0, -10, 2).to_vector(), (std::vector<float>{0, 1}));
    CHECK_THROWS(t.slice(0, 0, 5, 0), std::invalid_argument);
    CHECK_THROWS(t.slice(1, 0, 5), std::out_of_range);
    // slice of a slice composes offsets and strides
    CHECK_EQ(Tensor::arange(20).slice(0, 1, 20, 2).slice(0, 2, 8, 3).to_vector(),
             (std::vector<float>{5, 11}));
}

TEST(A6_broadcast_shapes) {
    CHECK_EQ(broadcast_shapes({3, 1, 5}, {4, 5}), (Shape{3, 4, 5}));
    CHECK_EQ(broadcast_shapes({1}, {7}), (Shape{7}));
    CHECK_EQ(broadcast_shapes({}, {2, 2}), (Shape{2, 2}));
    CHECK_THROWS(broadcast_shapes({3}, {4}), std::invalid_argument);
    CHECK_THROWS(broadcast_shapes({2, 3}, {3, 3}), std::invalid_argument);
}

TEST(A6_broadcast_to_uses_stride_zero) {
    Tensor row = Tensor::from_vector({1, 3}, {1, 2, 3});
    Tensor b = row.broadcast_to({4, 3});
    CHECK_EQ(b.shape(), (Shape{4, 3}));
    CHECK_EQ(b.strides(), (Strides{0, 1}));
    CHECK(b.shares_storage_with(row));
    CHECK_EQ(b.at({3, 2}), 3.0f);
    Tensor v = Tensor::from_vector({3}, {1, 2, 3}).broadcast_to({2, 2, 3});
    CHECK_EQ(v.strides(), (Strides{0, 0, 1}));
    CHECK_THROWS(row.broadcast_to({4, 2}), std::invalid_argument);
    CHECK_THROWS(row.broadcast_to({3}), std::invalid_argument);
}

TEST(A7_contiguous_copies_only_when_needed) {
    Tensor t = iota({2, 3});
    Tensor same = t.contiguous();
    CHECK(same.shares_storage_with(t));
    Tensor tt = t.transpose(0, 1).contiguous();
    CHECK(!tt.shares_storage_with(t));
    CHECK(tt.is_contiguous());
    CHECK_EQ(tt.strides(), (Strides{2, 1}));
    CHECK_EQ(tt.to_vector(), (std::vector<float>{0, 3, 1, 4, 2, 5}));
    // reshape of a non-contiguous tensor must go through contiguous()
    CHECK_EQ(t.transpose(0, 1).reshape({6}).to_vector(), (std::vector<float>{0, 3, 1, 4, 2, 5}));
    CHECK_THROWS(t.transpose(0, 1).view({6}), std::invalid_argument);
}

TEST(A7_contiguous_of_broadcast_materialises) {
    Tensor b = Tensor::from_vector({3}, {1, 2, 3}).broadcast_to({2, 3}).contiguous();
    CHECK_EQ(b.to_vector(), (std::vector<float>{1, 2, 3, 1, 2, 3}));
    CHECK_EQ(b.storage()->size(), 6u);
}

TEST(A8_zip_broadcasts) {
    Tensor a = iota({2, 3});
    Tensor col = Tensor::from_vector({2, 1}, {10, 20});
    CHECK_EQ((a + col).to_vector(), (std::vector<float>{10, 11, 12, 23, 24, 25}));
    Tensor row = Tensor::from_vector({3}, {1, 2, 3});
    CHECK_EQ((a * row).to_vector(), (std::vector<float>{0, 2, 6, 3, 8, 15}));
    CHECK_EQ((col - row).shape(), (Shape{2, 3}));
    CHECK_THROWS(a + Tensor::zeros({2}), std::invalid_argument);
}

TEST(A8_zip_on_views_does_not_mutate_inputs) {
    Tensor a = iota({3, 2});
    Tensor at = a.transpose(0, 1); // 2x3, non-contiguous
    Tensor out = at + at;
    CHECK(out.is_contiguous());
    CHECK_EQ(out.to_vector(), (std::vector<float>{0, 4, 8, 2, 6, 10}));
    CHECK_EQ(a.to_vector(), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}

TEST(A9_sum) {
    Tensor t = iota({2, 3, 4});
    Tensor s0 = sum(t, 0);
    CHECK_EQ(s0.shape(), (Shape{3, 4}));
    CHECK_EQ(s0.at({2, 3}), 11.0f + 23.0f);
    Tensor s1 = sum(t, 1, true);
    CHECK_EQ(s1.shape(), (Shape{2, 1, 4}));
    CHECK_EQ(s1.at({1, 0, 0}), 12.0f + 16.0f + 20.0f);
    Tensor s2 = sum(t, -1);
    CHECK_EQ(s2.shape(), (Shape{2, 3}));
    CHECK_EQ(s2.to_vector(), (std::vector<float>{6, 22, 38, 54, 70, 86}));
    CHECK_THROWS(sum(t, 3), std::out_of_range);
}

TEST(A9_sum_of_non_contiguous_view) {
    Tensor t = iota({2, 3});
    CHECK_EQ(sum(t.transpose(0, 1), 1).to_vector(), (std::vector<float>{3, 5, 7}));
    CHECK_EQ(sum(t.slice(1, 0, 3, 2), 0).to_vector(), (std::vector<float>{3, 7}));
}

TEST(A_capstone_softmax_from_views) {
    // softmax over the last dim, written only with ops you implemented.
    Tensor x = Tensor::from_vector({2, 3}, {1, 2, 3, 1, 1, 1});
    Tensor e = map(x, [](float v) { return std::exp(v); });
    Tensor p = e / sum(e, -1, true);
    CHECK_NEAR(p.at({0, 2}), std::exp(3.0) / (std::exp(1.0) + std::exp(2.0) + std::exp(3.0)), 1e-6);
    CHECK_NEAR(p.at({1, 1}), 1.0 / 3.0, 1e-6);
    CHECK_NEAR(sum(sum(p, -1), 0).at({}), 2.0, 1e-5);
}

int main(int argc, char **argv) {
    return run_tests(argc, argv);
}
