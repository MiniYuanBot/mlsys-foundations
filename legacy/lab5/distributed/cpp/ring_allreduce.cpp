// Original teaching scaffold. Host-memory MPI only; no CUDA/NCCL performance claim.
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

void student_ring_allreduce(std::vector<double>& values, MPI_Comm comm) {
  int rank = 0, world = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world);
  if (world == 1) return;
  // Scaffold constraint: count divisible by world, contiguous equal chunks.
  // TODO 1: allocate one chunk receive buffer; compute prev/next ranks.
  // TODO 2: world-1 reduce-scatter steps via MPI_Sendrecv; add received chunk.
  // TODO 3: world-1 all-gather steps via MPI_Sendrecv; copy received chunk.
  // Derive ownership after reduce-scatter before deciding the all-gather indices.
  // Never use MPI_Allreduce, MPI_Reduce, or MPI_Allgather in this implementation.
  // Every rank must call the same sequence; different step tags avoid mismatches.
  (void)values;
  (void)rank;
  throw std::runtime_error("TODO: implement ring reduce-scatter + all-gather");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  const bool reference = argc == 2 && std::strcmp(argv[1], "--reference") == 0;
  const int count = 2 * world;
  std::vector<double> values(count), oracle(count);
  for (int i = 0; i < count; ++i) values[i] = 10.0 * rank + i;
  MPI_Allreduce(values.data(), oracle.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  try {
    if (reference) values = oracle;
    else student_ring_allreduce(values, MPI_COMM_WORLD);
    int local_ok = 1;
    for (int i = 0; i < count; ++i) {
      const double hand = 10.0 * world * (world - 1) / 2.0 + world * i;
      if (!std::isfinite(values[i]) || std::abs(values[i] - hand) > 1e-12 ||
          std::abs(values[i] - oracle[i]) > 1e-12)
        local_ok = 0;
    }
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << (reference ? "REFERENCE" : "STUDENT") << " world=" << world << " result=";
      for (double value : values) std::cout << value << " ";
      std::cout << (all_ok ? "PASS\n" : "FAIL\n");
    }
    MPI_Finalize();
    return all_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 2);  // Fail all ranks promptly, including unimplemented code.
    return 2;
  }
}
