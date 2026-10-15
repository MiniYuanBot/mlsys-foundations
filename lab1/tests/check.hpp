// A 60-line test framework so lab0 has no third-party dependency.
//
//   TEST(name) { CHECK(cond); CHECK_EQ(a, b); CHECK_NEAR(a, b, tol); CHECK_THROWS(expr, Type); }
//   int main(int argc, char** argv) { return run_tests(argc, argv); }
//
// `test_x substring` runs only tests whose name contains `substring`.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace check {
struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct Case {
    const char* name;
    std::function<void()> fn;
};
inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}
struct Register {
    Register(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};
inline std::string where(const char* file, int line) {
    std::ostringstream os;
    os << file << ":" << line << ": ";
    return os.str();
}
} // namespace check

#define TEST(name)                                                                                 \
    static void test_##name();                                                                     \
    static check::Register register_##name(#name, test_##name);                                    \
    static void test_##name()

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) throw check::Failure(check::where(__FILE__, __LINE__) + "CHECK(" #cond ")");  \
    } while (0)

#define CHECK_EQ(a, b)                                                                             \
    do {                                                                                           \
        const auto va_ = (a);                                                                      \
        const auto vb_ = (b);                                                                      \
        if (!(va_ == vb_)) {                                                                       \
            std::ostringstream os_;                                                                \
            os_ << check::where(__FILE__, __LINE__) << "CHECK_EQ(" #a ", " #b ")";                 \
            throw check::Failure(os_.str());                                                       \
        }                                                                                          \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                      \
    do {                                                                                           \
        const double va_ = (a), vb_ = (b);                                                         \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                                    \
            std::ostringstream os_;                                                                \
            os_ << check::where(__FILE__, __LINE__) << #a " = " << va_ << " but " #b " = " << vb_; \
            throw check::Failure(os_.str());                                                       \
        }                                                                                          \
    } while (0)

#define CHECK_THROWS(expr, Type)                                                                   \
    do {                                                                                           \
        bool thrown_ = false;                                                                      \
        try {                                                                                      \
            (void)(expr);                                                                          \
        } catch (const Type&) {                                                                    \
            thrown_ = true;                                                                        \
        }                                                                                          \
        if (!thrown_)                                                                              \
            throw check::Failure(check::where(__FILE__, __LINE__) + #expr                          \
                                 " did not throw " #Type);                                         \
    } while (0)

inline int run_tests(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int passed = 0, failed = 0;
    for (const auto& c : check::registry()) {
        if (!filter.empty() && std::string(c.name).find(filter) == std::string::npos) continue;
        try {
            c.fn();
            std::printf("[ PASS ] %s\n", c.name);
            ++passed;
        } catch (const std::exception& e) {
            std::printf("[ FAIL ] %s\n         %s\n", c.name, e.what());
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 && passed > 0 ? 0 : 1;
}
