// Provided parts of the collectives library.
#include "coll/coll.hpp"

namespace coll {

Comm Comm::world() {
    Comm c;
    c.comm = MPI_COMM_WORLD;
    MPI_Comm_rank(c.comm, &c.rank);
    MPI_Comm_size(c.comm, &c.size);
    return c;
}

std::pair<size_t, size_t> chunk_range(size_t n, int P, int i) {
    const size_t base = n / P, extra = n % P;
    const size_t begin = i * base + std::min<size_t>(i, extra);
    return {begin, begin + base + (static_cast<size_t>(i) < extra ? 1 : 0)};
}

void allreduce_mpi(float* buf, size_t n, const Comm& c) {
    MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_FLOAT, MPI_SUM, c.comm);
}

} // namespace coll
