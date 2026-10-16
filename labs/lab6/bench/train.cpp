// Data-parallel training driver for Part C measurements.
//
//   mpirun -np 4 build/lab6/train --mode sync|overlap|zero1 [--bucket-mb 4] [--steps 20]
//          [--batch 16] [--hidden 1024] [--layers 3] [--algo ring|mpi]
//
// Prints one CSV row (rank 0): mean step time and its parts, measured after
// 3 warm-up steps (max over ranks):
//   compute    forward + backward (in overlap mode, buckets are being reduced meanwhile)
//   comm_wait  after backward, until the averaged gradients are available
//              (sync: the whole allreduce; overlap: what was not hidden)
//   optimizer  the Adam update (zero1: reduce-scatter + shard update + all-gather,
//              i.e. its communication is counted here)
#include "coll/train.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace coll;

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    const Comm c = Comm::world();
    std::string mode = "sync", algo = "ring";
    double bucket_mb = 4;
    int steps = 20, batch = 16, hidden = 1024, layers = 3;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--mode")
            mode = v;
        else if (a == "--bucket-mb")
            bucket_mb = std::atof(v.c_str());
        else if (a == "--steps")
            steps = std::atoi(v.c_str());
        else if (a == "--batch")
            batch = std::atoi(v.c_str());
        else if (a == "--hidden")
            hidden = std::atoi(v.c_str());
        else if (a == "--layers")
            layers = std::atoi(v.c_str());
        else if (a == "--algo")
            algo = v;
    }
    const AllreduceFn fn = algo == "mpi" ? allreduce_mpi : allreduce_ring;
    std::vector<int> dims = {hidden};
    for (int l = 0; l < layers; ++l) dims.push_back(hidden);
    dims.push_back(16);
    MLP m(dims, 1);
    CommThread ct(c, fn);
    Bucketer bk(m, static_cast<size_t>(bucket_mb * (1 << 20)), ct, c.size);
    Zero1Adam z(m.num_params(), c, Adam{});
    std::vector<float> x, y, mm, vv;
    if (mode != "zero1") {
        mm.assign(m.num_params(), 0.f);
        vv.assign(m.num_params(), 0.f);
    }
    double t_step = 0, t_compute = 0, t_wait = 0, t_opt = 0, comm_busy0 = 0;
    float loss = 0;
    const int warm = 3;
    for (int s = 0; s < warm + steps; ++s) {
        if (s == warm) {
            MPI_Barrier(c.comm);
            comm_busy0 = ct.busy_seconds();
        }
        make_batch(dims.front(), dims.back(), (static_cast<size_t>(s) * c.size + c.rank) * batch,
                   batch, x, y);
        const double t0 = MPI_Wtime();
        loss = m.forward(x.data(), y.data(), batch);
        double t1, t2;
        if (mode == "overlap") {
            m.backward([&](int l) { bk.on_layer_ready(l); });
            t1 = MPI_Wtime();
            bk.finish();
            t2 = MPI_Wtime();
            adam_update(m.params.data(), m.grads.data(), mm.data(), vv.data(), m.num_params(),
                        s + 1, Adam{});
        } else if (mode == "zero1") {
            m.backward();
            t1 = t2 = MPI_Wtime();
            z.step(m.params, m.grads);
        } else { // sync: backward, then one allreduce of everything
            m.backward();
            t1 = MPI_Wtime();
            fn(m.grads.data(), m.num_params(), c);
            for (float &g : m.grads) g /= c.size;
            t2 = MPI_Wtime();
            adam_update(m.params.data(), m.grads.data(), mm.data(), vv.data(), m.num_params(),
                        s + 1, Adam{});
        }
        const double t3 = MPI_Wtime();
        if (s >= warm) {
            t_step += t3 - t0;
            t_compute += t1 - t0;
            t_wait += t2 - t1;
            t_opt += t3 - t2;
        }
    }
    double mine[4] = {t_step / steps, t_compute / steps, t_wait / steps, t_opt / steps}, mx[4];
    MPI_Reduce(mine, mx, 4, MPI_DOUBLE, MPI_MAX, 0, c.comm);
    const double opt_mb =
        (mode == "zero1" ? z.state_floats() : 2 * m.num_params()) * 4.0 / (1 << 20);
    if (c.rank == 0) {
        std::printf("mode,algo,P,batch,bucket_mb,params_M,grad_MB,step_ms,compute_ms,comm_wait_ms,"
                    "optimizer_ms,"
                    "comm_thread_busy_ms,buckets_per_step,opt_state_MB_per_rank,loss\n");
        std::printf("%s,%s,%d,%d,%.2f,%.2f,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.5f\n",
                    mode.c_str(), algo.c_str(), c.size, batch, mode == "overlap" ? bucket_mb : 0.0,
                    m.num_params() / 1e6, m.num_params() * 4.0 / (1 << 20), mx[0] * 1e3,
                    mx[1] * 1e3, mx[2] * 1e3, mx[3] * 1e3,
                    (ct.busy_seconds() - comm_busy0) / steps * 1e3,
                    mode == "overlap" ? double(bk.buckets()) / (warm + steps) : 0.0, opt_mb, loss);
    }
    MPI_Finalize();
    return 0;
}
