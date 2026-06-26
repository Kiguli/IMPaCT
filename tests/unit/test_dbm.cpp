// ============================================================================
// CONTRACT TESTS — DBM zone abstraction for timed automata.
// Hand cases + a grid differential oracle for intersect / delay(up) / reset, whose
// continuous semantics are checked against an explicit finite-grid set model.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>
#include <functional>

using namespace impact::dbm;

// enumerate clock valuations on {0,0.5,...,MAX}^2 and compare two membership predicates
static void gridAgree(int n, double MAX, double step,
                      const std::function<bool(const std::vector<double>&)>& a,
                      const std::function<bool(const std::vector<double>&)>& b) {
    REQUIRE(n == 2);
    for (double x = 0; x <= MAX + 1e-9; x += step)
        for (double y = 0; y <= MAX + 1e-9; y += step) {
            std::vector<double> v = {x, y};
            CHECK(a(v) == b(v));
        }
}

static Zone box2(long long x1max, long long x2max) {       // 0<=x1<=x1max, 0<=x2<=x2max
    Zone z(2);
    constrain(z, 1, 0, Bound::leq(x1max));
    constrain(z, 2, 0, Bound::leq(x2max));
    return z;
}

TEST_CASE("dbm: universe contains all non-negative valuations; negatives excluded") {
    Zone z(2);
    CHECK(contains(z, {0.0, 0.0}));
    CHECK(contains(z, {3.5, 10.0}));
    CHECK(!contains(z, {-0.5, 0.0}));     // clocks are >= 0
    CHECK(!isEmpty(z));
}

TEST_CASE("dbm: contradictory constraints -> empty") {
    Zone z(2);
    constrain(z, 1, 0, Bound::leq(2));    // x1 <= 2
    constrain(z, 0, 1, Bound::leq(-3));   // -x1 <= -3  i.e. x1 >= 3
    CHECK(isEmpty(z));
}

TEST_CASE("dbm: difference constraint guard") {
    Zone z(2);
    constrain(z, 1, 2, Bound::leq(1));    // x1 - x2 <= 1
    CHECK(contains(z, {2.0, 1.5}));       // 0.5 <= 1 ok
    CHECK(!contains(z, {3.0, 1.0}));      // 2 > 1 violates
}

TEST_CASE("dbm: intersect == grid set intersection") {
    Zone a = box2(3, 4);
    Zone b(2); constrain(b, 2, 0, Bound::leq(2)); constrain(b, 1, 2, Bound::leq(1)); // x2<=2, x1-x2<=1
    Zone r = intersect(a, b);
    gridAgree(2, 5.0, 0.5,
        [&](const std::vector<double>& v){ return contains(r, v); },
        [&](const std::vector<double>& v){ return contains(a, v) && contains(b, v); });
    CHECK(!isEmpty(r));
}

TEST_CASE("dbm: up (delay) == exists d>=0 with v-d in zone") {
    Zone a = box2(3, 2);
    constrain(a, 0, 1, Bound::leq(-1));            // x1 >= 1
    Zone r = a;
    up(r);
    gridAgree(2, 6.0, 0.5,
        [&](const std::vector<double>& v){ return contains(r, v); },
        [&](const std::vector<double>& v){
            for (double d = 0; d <= 6.0 + 1e-9; d += 0.5) {
                double x = v[0]-d, y = v[1]-d;
                if (x < -1e-9 || y < -1e-9) continue;
                if (contains(a, {x, y})) return true;
            }
            return false;
        });
}

TEST_CASE("dbm: reset == project, setting the clock to 0") {
    Zone a = box2(3, 3);
    constrain(a, 1, 2, Bound::leq(1));             // x1 - x2 <= 1
    Zone r = a; reset(r, 1);                        // x1 := 0
    gridAgree(2, 4.0, 0.5,
        [&](const std::vector<double>& v){ return contains(r, v); },
        [&](const std::vector<double>& v){
            if (v[0] > 1e-9) return false;          // x1 must be 0 after reset
            for (double u = 0; u <= 4.0 + 1e-9; u += 0.5)   // some original x1=u with same x2
                if (contains(a, {u, v[1]})) return true;
            return false;
        });
}

TEST_CASE("dbm: inclusion test") {
    Zone big = box2(5, 5);
    Zone small = box2(2, 2);
    CHECK(includes(big, small));     // small subset of big
    CHECK(!includes(small, big));    // big not subset of small
    CHECK(includes(big, big));
}
