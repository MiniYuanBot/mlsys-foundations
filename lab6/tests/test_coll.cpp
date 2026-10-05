// Part A + B tests, run with 1, 2, 3, 4 and 8 ranks by CTest.
#include "coll/coll.hpp"
#include "mpitest.hpp"

#include <vector>

using namespace coll;

// rank-dependent integers: sums are exact in float, so we can compare exactly
static std::vector<float> make(size_t n, int rank) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<float>((rank + 1) * static_cast<int>(i % 13) - static_cast<int>(i % 5));
    return v;
}
static float expected_sum(size_t i, int P) {
    return static_cast<float>(static_cast<int>(i % 13) * P * (P + 1) / 2 - P * static_cast<int>(i % 5));
}
static const size_t SIZES[] = {1, 2, 3, 7, 8, 100, 1000, 65537};

TEST(A0_chunk_range_provided) {
    for (size_t n : {0, 1, 7, 10, 1000})
        for (int P : {1, 2, 3, 8}) {
            size_t prev = 0, maxc = 0, minc = n + 1;
            for (int i = 0; i < P; ++i) {
                auto [b, e] = chunk_range(n, P, i);
                CHECK(b == prev && e >= b);
                prev = e;
                maxc = std::max(maxc, e - b);
                minc = std::min(minc, e - b);
            }
            CHECK(prev == n && maxc - minc <= 1);
        }
}

TEST(A1_ring_shift) {
    const Comm c = Comm::world();
    for (size_t n : SIZES) {
        auto send = make(n, c.rank);
        std::vector<float> recv(n, -1.f);
        ring_shift(send.data(), recv.data(), n, c);
        CHECK(recv == make(n, c.left()));
    }
}

static void check_allreduce(AllreduceFn fn) {
    const Comm c = Comm::world();
    for (size_t n : SIZES) {
        auto buf = make(n, c.rank);
        fn(buf.data(), n, c);
        for (size_t i = 0; i < n; ++i) CHECK(buf[i] == expected_sum(i, c.size));
    }
}

TEST(A2_allreduce_star) { check_allreduce(allreduce_star); }

TEST(B0_allreduce_mpi_provided) { check_allreduce(allreduce_mpi); }

TEST(B1a_reduce_scatter_ring_owner_chunk) {
    const Comm c = Comm::world();
    for (size_t n : SIZES) {
        auto buf = make(n, c.rank);
        reduce_scatter_ring(buf.data(), n, c);
        auto [b, e] = chunk_range(n, c.size, c.rank);
        for (size_t i = b; i < e; ++i) CHECK(buf[i] == expected_sum(i, c.size));
    }
}

TEST(B1b_allgather_ring) {
    const Comm c = Comm::world();
    for (size_t n : SIZES) {
        std::vector<float> buf(n, -7.f);
        auto [b, e] = chunk_range(n, c.size, c.rank);
        for (size_t i = b; i < e; ++i) buf[i] = static_cast<float>(i) * 0.5f;  // only my chunk is valid
        allgather_ring(buf.data(), n, c);
        for (size_t i = 0; i < n; ++i) CHECK(buf[i] == static_cast<float>(i) * 0.5f);
    }
}

TEST(B1c_allreduce_ring) { check_allreduce(allreduce_ring); }

TEST(B1c_allreduce_ring_large) {
    const Comm c = Comm::world();
    const size_t n = (1 << 22) + 13;  // 16 MB: deadlocks if you use Send + Recv instead of Sendrecv
    auto buf = make(n, c.rank);
    allreduce_ring(buf.data(), n, c);
    for (size_t i = 0; i < n; i += 4097) CHECK(buf[i] == expected_sum(i, c.size));
}

TEST(B2_allreduce_recursive_doubling) {
    const Comm c = Comm::world();
    if (c.size & (c.size - 1)) {
        std::vector<float> buf(4);
        bool threw = false;
        try {
            allreduce_rd(buf.data(), 4, c);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
        return;
    }
    check_allreduce(allreduce_rd);
}

TEST(B_all_ranks_get_identical_bits) {
    // floating-point inputs: every rank must end with exactly the same bits
    const Comm c = Comm::world();
    for (AllreduceFn fn : {allreduce_ring, allreduce_star}) {
        std::vector<float> buf(10007);
        for (size_t i = 0; i < buf.size(); ++i) buf[i] = std::sin(0.37f * i + c.rank) * 1e3f;
        fn(buf.data(), buf.size(), c);
        std::vector<float> root = buf;
        MPI_Bcast(root.data(), static_cast<int>(root.size()), MPI_FLOAT, 0, c.comm);
        CHECK(root == buf);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = run_mpi_tests(argc, argv);
    MPI_Finalize();
    return rc;
}
