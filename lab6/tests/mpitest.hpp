// A tiny test framework for MPI programs. Every rank runs every test; a test
// passes only if it passes on ALL ranks. Run under mpirun:
//   mpirun -np 4 test_coll [name-filter]
// CHECK failures throw on the failing rank; since a collective may then be
// left half-done on other ranks, write tests so that every rank finishes its
// collectives BEFORE checking results (compute first, check after).
#pragma once

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mpitest {
struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct Case {
    const char* name;
    std::function<void()> fn;
};
inline std::vector<Case>& registry() {
    static std::vector<Case> c;
    return c;
}
struct Register {
    Register(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};
inline int rank() {
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}
inline int size() {
    int s;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}
}  // namespace mpitest

#define TEST(name)                                                        \
    static void test_##name();                                            \
    static mpitest::Register register_##name(#name, test_##name);         \
    static void test_##name()

#define CHECK(cond)                                                                                        \
    do {                                                                                                   \
        if (!(cond)) {                                                                                     \
            std::ostringstream os_;                                                                        \
            os_ << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ") failed on rank " << mpitest::rank(); \
            throw mpitest::Failure(os_.str());                                                             \
        }                                                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                \
    do {                                                                                                     \
        const double va_ = (a), vb_ = (b);                                                                   \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                                              \
            std::ostringstream os_;                                                                          \
            os_ << __FILE__ << ":" << __LINE__ << ": " #a " = " << va_ << ", " #b " = " << vb_ << " on rank " \
                << mpitest::rank();                                                                          \
            throw mpitest::Failure(os_.str());                                                               \
        }                                                                                                    \
    } while (0)

inline int run_mpi_tests(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    const int r = mpitest::rank(), P = mpitest::size();
    int passed = 0, failed = 0;
    for (const auto& c : mpitest::registry()) {
        if (!filter.empty() && std::string(c.name).find(filter) == std::string::npos) continue;
        int ok = 1;
        std::string msg;
        try {
            c.fn();
        } catch (const std::exception& e) {
            ok = 0;
            msg = e.what();
        }
        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!ok) std::fprintf(stderr, "[rank %d] %s: %s\n", r, c.name, msg.c_str());
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == 0) std::printf("[ %s ] %s  (P=%d)\n", all_ok ? "PASS" : "FAIL", c.name, P);
        std::fflush(stdout);
        (all_ok ? passed : failed)++;
    }
    if (r == 0) std::printf("\n%d passed, %d failed  (P=%d)\n", passed, failed, P);
    return failed == 0 && passed > 0 ? 0 : 1;
}
