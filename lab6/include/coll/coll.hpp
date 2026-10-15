// lab6 collectives, written with MPI point-to-point only (MPI_Send / MPI_Recv /
// MPI_Sendrecv). All buffers are float, all operations are SUM, all ranks of
// the communicator call each function with the same n.
//
// Chunking (used by the ring algorithms and by ZeRO): n elements are split into
// P contiguous chunks whose sizes differ by at most one; chunk_range(n, P, i)
// gives [begin, end) of chunk i. Chunk i is "owned" by rank i.
#pragma once

#include <mpi.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace coll {

struct Comm {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    static Comm world();
    int left() const { return (rank - 1 + size) % size; }
    int right() const { return (rank + 1) % size; }
};

struct NotImplemented : std::logic_error {
    using std::logic_error::logic_error;
};

// Provided. Chunk i of n elements split P ways: the first n % P chunks get one extra.
std::pair<size_t, size_t> chunk_range(size_t n, int P, int i);

// ---------------------------------------------------------------- Part A ----
// A1: every rank sends `send` (n floats) to its right neighbour and receives
//     the left neighbour's data into `recv`. With P == 1, copy send to recv.
void ring_shift(const float* send, float* recv, size_t n, const Comm& c);
// A2: the naive "star" allreduce: every rank sends its buffer to rank 0, rank 0
//     adds them up and sends the result back to everyone. In place.
void allreduce_star(float* buf, size_t n, const Comm& c);

// ---------------------------------------------------------------- Part B ----
// B1a: ring reduce-scatter, in place. Afterwards chunk `c.rank` of buf holds the
//      sum over all ranks of that chunk; other chunks hold partial garbage.
void reduce_scatter_ring(float* buf, size_t n, const Comm& c);
// B1b: ring all-gather, in place. Before: chunk `c.rank` of buf is valid on each
//      rank. Afterwards every chunk is valid on every rank.
void allgather_ring(float* buf, size_t n, const Comm& c);
// B1c: ring allreduce = reduce-scatter + all-gather.
void allreduce_ring(float* buf, size_t n, const Comm& c);
// B2: recursive-doubling allreduce. Requires P to be a power of two (throw
//     std::invalid_argument otherwise). log2(P) steps, each exchanging the whole buffer.
void allreduce_rd(float* buf, size_t n, const Comm& c);

// Baseline: the MPI library's own MPI_Allreduce (in place). Provided.
void allreduce_mpi(float* buf, size_t n, const Comm& c);

using AllreduceFn = void (*)(float*, size_t, const Comm&);

} // namespace coll
