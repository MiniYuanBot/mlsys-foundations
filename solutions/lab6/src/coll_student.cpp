// Part A + B: collectives from point-to-point messages. Fill in TODO(A*, B*).
//
//   ctest --test-dir build/lab6 -L coll --output-on-failure   (runs with 1, 2, 3, 4, 8 ranks)
//   mpirun -np 4 build/lab6/test_coll B1                      (one rank count, one test group)
//
// Use MPI_Sendrecv for every exchange between neighbours: a plain MPI_Send
// followed by MPI_Recv DEADLOCKS on a ring when messages are large (every rank
// blocks in Send waiting for a receiver that is itself blocked in Send).
// Messages larger than INT_MAX floats are not needed in this lab.
// Draw the chunk movement for P = 3 on paper before writing B1: lecture L6 section 2.
#include "coll/coll.hpp"

#include <algorithm>
#include <vector>

namespace coll {

[[maybe_unused]] static NotImplemented todo(const char* what) {
    return NotImplemented(std::string("TODO ") + what + " is not implemented yet");
}

// Tags only need to be distinct between concurrent messages; one per step is plenty.
[[maybe_unused]] static constexpr int TAG = 600;

// ------------------------------------------------------------------- A1 ----
void ring_shift(const float* send, float* recv, size_t n, const Comm& c) {
    if (c.size == 1) {
        std::copy(send, send + n, recv);
        return;
    }
    MPI_Sendrecv(send, static_cast<int>(n), MPI_FLOAT, c.right(), TAG, recv, static_cast<int>(n),
                 MPI_FLOAT, c.left(), TAG, c.comm, MPI_STATUS_IGNORE);
}

// ------------------------------------------------------------------- A2 ----
// Rank 0: receive P-1 buffers (MPI_Recv from each rank r = 1..P-1 into a
// temporary), add them into buf, then MPI_Send buf to every r. Others:
// MPI_Send buf to 0, then MPI_Recv the result into buf.
// Cost: rank 0 moves 2(P-1)n floats by itself -- the bottleneck the ring removes.
void allreduce_star(float* buf, size_t n, const Comm& c) {
    const int N = static_cast<int>(n);
    if (c.rank == 0) {
        std::vector<float> tmp(n);
        for (int r = 1; r < c.size; ++r) {
            MPI_Recv(tmp.data(), N, MPI_FLOAT, r, TAG, c.comm, MPI_STATUS_IGNORE);
            for (size_t i = 0; i < n; ++i) buf[i] += tmp[i];
        }
        for (int r = 1; r < c.size; ++r) MPI_Send(buf, N, MPI_FLOAT, r, TAG + 1, c.comm);
    } else {
        MPI_Send(buf, N, MPI_FLOAT, 0, TAG, c.comm);
        MPI_Recv(buf, N, MPI_FLOAT, 0, TAG + 1, c.comm, MPI_STATUS_IGNORE);
    }
}

// ------------------------------------------------------------------- B1 ----
// TODO(B1a) ring reduce-scatter. P - 1 steps; at step s (0 .. P-2) rank r
//   sends    chunk (r - s - 1) mod P   to the right neighbour,
//   receives chunk (r - s - 2) mod P   from the left neighbour into a temporary
//            and ADDS it into that chunk of buf.
// After the last step, chunk r of rank r has been added up by every rank.
// (Check with P = 2: rank 0 sends chunk 1, receives chunk 0 -> owns chunk 0.)
// Chunks can have different sizes (chunk_range): size each message by its chunk.
void reduce_scatter_ring(float* buf, size_t n, const Comm& c) {
    const int P = c.size;
    if (P == 1) return;
    std::vector<float> tmp(chunk_range(n, P, 0).second); // chunk 0 is the largest
    for (int s = 0; s < P - 1; ++s) {
        const int si = ((c.rank - s - 1) % P + P) % P, ri = ((c.rank - s - 2) % P + P) % P;
        const auto [sb, se] = chunk_range(n, P, si);
        const auto [rb, re] = chunk_range(n, P, ri);
        MPI_Sendrecv(buf + sb, static_cast<int>(se - sb), MPI_FLOAT, c.right(), TAG + s, tmp.data(),
                     static_cast<int>(re - rb), MPI_FLOAT, c.left(), TAG + s, c.comm,
                     MPI_STATUS_IGNORE);
        for (size_t i = rb; i < re; ++i) buf[i] += tmp[i - rb];
    }
}

// TODO(B1b) ring all-gather. P - 1 steps; at step s rank r
//   sends    chunk (r - s) mod P       to the right,
//   receives chunk (r - s - 1) mod P   from the left, straight into buf (no add).
void allgather_ring(float* buf, size_t n, const Comm& c) {
    const int P = c.size;
    if (P == 1) return;
    for (int s = 0; s < P - 1; ++s) {
        const int si = ((c.rank - s) % P + P) % P, ri = ((c.rank - s - 1) % P + P) % P;
        const auto [sb, se] = chunk_range(n, P, si);
        const auto [rb, re] = chunk_range(n, P, ri);
        MPI_Sendrecv(buf + sb, static_cast<int>(se - sb), MPI_FLOAT, c.right(), TAG + 100 + s,
                     buf + rb, static_cast<int>(re - rb), MPI_FLOAT, c.left(), TAG + 100 + s,
                     c.comm, MPI_STATUS_IGNORE);
    }
}

// TODO(B1c) two lines.
void allreduce_ring(float* buf, size_t n, const Comm& c) {
    reduce_scatter_ring(buf, n, c);
    allgather_ring(buf, n, c);
}

// ------------------------------------------------------------------- B2 ----
// TODO(B2) recursive doubling. For mask = 1, 2, 4, ... < P:
//   partner = rank XOR mask; exchange the WHOLE buffer with the partner
//   (MPI_Sendrecv into a temporary) and add.
// After log2(P) steps everyone has the sum. Throw std::invalid_argument
// if P is not a power of two (check: P & (P - 1)).
// Latency log2(P) * alpha instead of the ring's 2(P-1) * alpha, but every step
// moves the whole buffer: log2(P) * n floats instead of the ring's ~2n.
void allreduce_rd(float* buf, size_t n, const Comm& c) {
    if (c.size & (c.size - 1))
        throw std::invalid_argument("allreduce_rd: P must be a power of two");
    std::vector<float> tmp(n);
    for (int mask = 1; mask < c.size; mask <<= 1) {
        const int partner = c.rank ^ mask;
        MPI_Sendrecv(buf, static_cast<int>(n), MPI_FLOAT, partner, TAG + 200 + mask, tmp.data(),
                     static_cast<int>(n), MPI_FLOAT, partner, TAG + 200 + mask, c.comm,
                     MPI_STATUS_IGNORE);
        for (size_t i = 0; i < n; ++i) buf[i] += tmp[i];
    }
}

} // namespace coll
