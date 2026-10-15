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
    (void)send;
    (void)recv;
    (void)n;
    (void)c;
    throw todo("A1 ring_shift");
}

// ------------------------------------------------------------------- A2 ----
// Rank 0: receive P-1 buffers (MPI_Recv from each rank r = 1..P-1 into a
// temporary), add them into buf, then MPI_Send buf to every r. Others:
// MPI_Send buf to 0, then MPI_Recv the result into buf.
// Cost: rank 0 moves 2(P-1)n floats by itself -- the bottleneck the ring removes.
void allreduce_star(float* buf, size_t n, const Comm& c) {
    (void)buf;
    (void)n;
    (void)c;
    throw todo("A2 allreduce_star");
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
    (void)buf;
    (void)n;
    (void)c;
    throw todo("B1a reduce_scatter_ring");
}

// TODO(B1b) ring all-gather. P - 1 steps; at step s rank r
//   sends    chunk (r - s) mod P       to the right,
//   receives chunk (r - s - 1) mod P   from the left, straight into buf (no add).
void allgather_ring(float* buf, size_t n, const Comm& c) {
    (void)buf;
    (void)n;
    (void)c;
    throw todo("B1b allgather_ring");
}

// TODO(B1c) two lines.
void allreduce_ring(float* buf, size_t n, const Comm& c) {
    (void)buf;
    (void)n;
    (void)c;
    throw todo("B1c allreduce_ring");
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
    (void)buf;
    (void)n;
    (void)c;
    throw todo("B2 allreduce_rd");
}

} // namespace coll
