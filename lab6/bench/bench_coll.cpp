// Collective benchmarks.
//
//   mpirun -np 2 build/lab6/bench_coll --pingpong > results/lab6/pingpong.csv   (for the alpha-beta fit)
//   mpirun -np 4 build/lab6/bench_coll           > results/lab6/allreduce_p4.csv
//
// allreduce rows: algorithm, P, message bytes, median time over repeats,
// "algbw" = bytes / time and "busbw" = algbw * 2(P-1)/P (the nccl-tests
// convention: the per-link traffic an optimal ring needs, comparable across P).
#include "coll/coll.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace coll;

static double median_time(const std::function<void()>& fn, const Comm& c, int reps) {
    std::vector<double> t;
    fn();  // warm-up
    for (int r = 0; r < reps; ++r) {
        MPI_Barrier(c.comm);
        const double t0 = MPI_Wtime();
        fn();
        double dt = MPI_Wtime() - t0, mx;
        MPI_Allreduce(&dt, &mx, 1, MPI_DOUBLE, MPI_MAX, c.comm);
        t.push_back(mx);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const Comm c = Comm::world();
    const bool pingpong = argc > 1 && std::string(argv[1]) == "--pingpong";
    if (pingpong) {
        if (c.size != 2) {
            if (c.rank == 0) std::fprintf(stderr, "--pingpong needs exactly 2 ranks\n");
            MPI_Finalize();
            return 2;
        }
        if (c.rank == 0) std::printf("bytes,one_way_us\n");
        for (size_t bytes = 4; bytes <= (size_t(64) << 20); bytes *= 4) {
            std::vector<char> buf(bytes);
            const int reps = bytes > (1 << 20) ? 20 : 200;
            const double t = median_time([&] {
                if (c.rank == 0) {
                    MPI_Send(buf.data(), static_cast<int>(bytes), MPI_BYTE, 1, 0, c.comm);
                    MPI_Recv(buf.data(), static_cast<int>(bytes), MPI_BYTE, 1, 0, c.comm, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buf.data(), static_cast<int>(bytes), MPI_BYTE, 0, 0, c.comm, MPI_STATUS_IGNORE);
                    MPI_Send(buf.data(), static_cast<int>(bytes), MPI_BYTE, 0, 0, c.comm);
                }
            }, c, reps);
            if (c.rank == 0) std::printf("%zu,%.3f\n", bytes, t / 2 * 1e6);
        }
        MPI_Finalize();
        return 0;
    }
    struct Algo {
        const char* name;
        AllreduceFn fn;
    };
    std::vector<Algo> algos = {{"mpi", allreduce_mpi}, {"ring", allreduce_ring}, {"star", allreduce_star}};
    if ((c.size & (c.size - 1)) == 0) algos.push_back({"rd", allreduce_rd});
    if (c.rank == 0) std::printf("algo,P,bytes,us,algbw_GBps,busbw_GBps\n");
    for (size_t bytes = 4; bytes <= (size_t(64) << 20); bytes *= 4) {
        const size_t n = bytes / 4;
        std::vector<float> buf(n, 1.f);
        for (const auto& a : algos) {
            double t;
            try {
                t = median_time([&] { a.fn(buf.data(), n, c); }, c, bytes > (4 << 20) ? 10 : 50);
            } catch (const NotImplemented& e) {
                if (c.rank == 0 && bytes == 4) std::fprintf(stderr, "# skip %s: %s\n", a.name, e.what());
                continue;
            }
            const double alg = bytes / t / 1e9;
            if (c.rank == 0)
                std::printf("%s,%d,%zu,%.2f,%.3f,%.3f\n", a.name, c.size, bytes, t * 1e6, alg,
                            c.size > 1 ? alg * 2 * (c.size - 1) / c.size : 0.0);
            std::fflush(stdout);
        }
    }
    MPI_Finalize();
    return 0;
}
