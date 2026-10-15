// Part C tests, run with 1, 2 and 4 ranks. Data parallelism must be
// INVISIBLE: P ranks with B samples each == one process with P*B samples.
#include "coll/train.hpp"
#include "mpitest.hpp"

#include <algorithm>

using namespace coll;

static const std::vector<int> DIMS = {32, 48, 48, 8};
static const int B = 4, STEPS = 5;

// Reference: one process, the whole global batch, SGD or Adam.
static std::vector<float> reference(int P, bool adam) {
    MLP m(DIMS, 1);
    std::vector<float> x, y, mm(m.num_params()), vv(m.num_params());
    for (int s = 0; s < STEPS; ++s) {
        make_batch(DIMS.front(), DIMS.back(), static_cast<size_t>(s) * B * P, B * P, x, y);
        m.forward(x.data(), y.data(), B * P);
        m.backward();
        if (adam)
            adam_update(m.params.data(), m.grads.data(), mm.data(), vv.data(), m.num_params(),
                        s + 1, Adam{});
        else
            sgd_update(m.params.data(), m.grads.data(), m.num_params(), 0.05f);
    }
    return m.params;
}

static void local_batch(const Comm& c, int s, std::vector<float>& x, std::vector<float>& y) {
    make_batch(DIMS.front(), DIMS.back(),
               static_cast<size_t>(s) * B * c.size + static_cast<size_t>(c.rank) * B, B, x, y);
}

static double max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::fabs(double(a[i]) - b[i]) / (1e-3 + std::fabs(b[i])));
    return worst;
}

TEST(C0_sync_ddp_equals_big_batch_provided) {
    const Comm c = Comm::world();
    MLP m(DIMS, 1);
    std::vector<float> x, y;
    for (int s = 0; s < STEPS; ++s) {
        local_batch(c, s, x, y);
        m.forward(x.data(), y.data(), B);
        m.backward();
        allreduce_mpi(m.grads.data(), m.num_params(), c);
        for (float& g : m.grads) g /= c.size;
        sgd_update(m.params.data(), m.grads.data(), m.num_params(), 0.05f);
    }
    CHECK(max_rel_diff(m.params, reference(c.size, false)) < 1e-4);
}

static void bucketed_run(size_t bucket_bytes, int expect_buckets) {
    const Comm c = Comm::world();
    MLP m(DIMS, 1);
    CommThread ct(c, allreduce_ring);
    Bucketer bk(m, bucket_bytes, ct, c.size);
    std::vector<float> x, y;
    for (int s = 0; s < STEPS; ++s) {
        local_batch(c, s, x, y);
        m.forward(x.data(), y.data(), B);
        m.backward([&](int l) { bk.on_layer_ready(l); });
        bk.finish();
        sgd_update(m.params.data(), m.grads.data(), m.num_params(), 0.05f);
    }
    CHECK(bk.buckets() == expect_buckets * STEPS);
    CHECK(ct.submitted() == expect_buckets * STEPS);
    CHECK(max_rel_diff(m.params, reference(c.size, false)) < 1e-4);
}

TEST(C1_bucketed_one_bucket_per_layer) {
    bucketed_run(4, 3);
} // tiny cap: every layer flushes
TEST(C1_bucketed_single_bucket) {
    bucketed_run(1 << 30, 1);
} // huge cap: one bucket at the end
TEST(C1_bucketed_mixed) {
    // layer sizes in backward order: 392, 2352, 1584 floats. cap 2000 floats:
    // 392 (no), +2352 = 2744 -> flush; 1584 (no) -> flushed by finish()  => 2 buckets
    bucketed_run(2000 * 4, 2);
}

TEST(C1_bucketer_rejects_out_of_order_layers) {
    const Comm c = Comm::world();
    MLP m(DIMS, 1);
    CommThread ct(c, allreduce_ring);
    Bucketer bk(m, 1 << 20, ct, c.size);
    bool threw = false;
    try {
        bk.on_layer_ready(0); // layer 2 must come first
    } catch (const NotImplemented&) {
        throw; // "not implemented" is not the error we are looking for
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(C2_zero1_equals_ddp_adam) {
    const Comm c = Comm::world();
    MLP m(DIMS, 1);
    Zero1Adam opt(m.num_params(), c, Adam{});
    std::vector<float> x, y;
    for (int s = 0; s < STEPS; ++s) {
        local_batch(c, s, x, y);
        m.forward(x.data(), y.data(), B);
        m.backward();
        opt.step(m.params, m.grads);
    }
    CHECK(max_rel_diff(m.params, reference(c.size, true)) < 1e-3);
    // optimizer state: 2 * ceil(N / P) floats instead of 2N
    const size_t N = m.num_params();
    CHECK(opt.state_floats() <= 2 * ((N + c.size - 1) / c.size));
    // every rank must hold identical parameters
    std::vector<float> root = m.params;
    MPI_Bcast(root.data(), static_cast<int>(root.size()), MPI_FLOAT, 0, c.comm);
    CHECK(root == m.params);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE && mpitest::rank() == 0)
        std::fprintf(
            stderr,
            "warning: MPI_THREAD_MULTIPLE not provided (%d); the comm thread may misbehave\n",
            provided);
    const int rc = run_mpi_tests(argc, argv);
    MPI_Finalize();
    return rc;
}
