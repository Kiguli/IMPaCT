// ============================================================================
// CONTRACT TESTS — Phase 1a: O-maximization inner robust Bellman solve.
// impact::omax::optimize must match these hand-derived optima exactly, and
// must agree with an INDEPENDENT brute-force vertex enumeration on random boxes.
//
// IMMUTABLE: do not edit expected values to match an implementation. The
// expected values are derived by hand below and re-derivable from the algorithm
// spec (Givan-Leach-Dean 2000). Cross-checked in tests/oracles/oracles.py.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>

using impact::omax::optimize;
using impact::omax::Sense;
using std::vector;

static constexpr double TOL = 1e-9;

// --- Independent oracle: brute-force the polytope vertices ------------------
// Vertices of {p : lower<=p<=upper, sum p = 1} have all coords at a bound
// except (at most) one "pivot" that absorbs the residual. Enumerate (pivot,
// bitmask of which non-pivot coords are at upper) and keep feasible vertices.
// This is an independent method from sort-and-assign, so it is a real cross
// check, not greedy-vs-greedy.
static double brute_optimum(const vector<double>& lo, const vector<double>& up,
                            const vector<double>& V, Sense sense) {
    const int n = (int)V.size();
    double best = (sense == Sense::Min) ? std::numeric_limits<double>::infinity()
                                        : -std::numeric_limits<double>::infinity();
    for (int pivot = 0; pivot < n; ++pivot) {
        for (int mask = 0; mask < (1 << (n - 1)); ++mask) {
            vector<double> p(n);
            double assigned = 0.0;
            int bit = 0;
            for (int i = 0; i < n; ++i) {
                if (i == pivot) continue;
                bool atUpper = (mask >> bit) & 1;
                p[i] = atUpper ? up[i] : lo[i];
                assigned += p[i];
                ++bit;
            }
            double resid = 1.0 - assigned;
            if (resid < lo[pivot] - 1e-12 || resid > up[pivot] + 1e-12) continue; // infeasible vertex
            p[pivot] = resid;
            double val = 0.0;
            for (int i = 0; i < n; ++i) val += p[i] * V[i];
            if (sense == Sense::Min) best = std::min(best, val);
            else                     best = std::max(best, val);
        }
    }
    return best;
}

static void check_valid_distribution(const vector<double>& p,
                                     const vector<double>& lo,
                                     const vector<double>& up) {
    double s = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        CHECK(p[i] >= lo[i] - TOL);
        CHECK(p[i] <= up[i] + TOL);
        s += p[i];
    }
    CHECK(s == doctest::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("omax: hand example A — distinct values, slack present") {
    vector<double> lo = {0.1, 0.2, 0.1};
    vector<double> up = {0.6, 0.7, 0.8};
    vector<double> V  = {0.0, 0.5, 1.0};

    auto rmin = optimize(lo, up, V, Sense::Min);
    CHECK(rmin.value == doctest::Approx(0.25));
    CHECK(rmin.p[0] == doctest::Approx(0.6));
    CHECK(rmin.p[1] == doctest::Approx(0.3));
    CHECK(rmin.p[2] == doctest::Approx(0.1));
    check_valid_distribution(rmin.p, lo, up);

    auto rmax = optimize(lo, up, V, Sense::Max);
    CHECK(rmax.value == doctest::Approx(0.8));
    CHECK(rmax.p[0] == doctest::Approx(0.1));
    CHECK(rmax.p[1] == doctest::Approx(0.2));
    CHECK(rmax.p[2] == doctest::Approx(0.7));
    check_valid_distribution(rmax.p, lo, up);
}

TEST_CASE("omax: no slack — lowers already sum to 1") {
    vector<double> lo = {0.3, 0.3, 0.4};
    vector<double> up = {1.0, 1.0, 1.0};
    vector<double> V  = {0.0, 0.5, 1.0};
    auto rmin = optimize(lo, up, V, Sense::Min);
    auto rmax = optimize(lo, up, V, Sense::Max);
    CHECK(rmin.value == doctest::Approx(0.55));
    CHECK(rmax.value == doctest::Approx(0.55)); // forced distribution, min==max
}

TEST_CASE("omax: single successor must take all mass") {
    vector<double> lo = {0.0}, up = {1.0}, V = {0.7};
    CHECK(optimize(lo, up, V, Sense::Min).value == doctest::Approx(0.7));
    CHECK(optimize(lo, up, V, Sense::Max).value == doctest::Approx(0.7));
    CHECK(optimize(lo, up, V, Sense::Max).p[0] == doctest::Approx(1.0));
}

TEST_CASE("omax: tied values — any split gives same value") {
    vector<double> lo = {0.0, 0.0}, up = {1.0, 1.0}, V = {0.5, 0.5};
    CHECK(optimize(lo, up, V, Sense::Min).value == doctest::Approx(0.5));
    CHECK(optimize(lo, up, V, Sense::Max).value == doctest::Approx(0.5));
}

TEST_CASE("omax: pivot exactly consumes residual") {
    // lowers sum 0.4, residual 0.6; index0 gap exactly 0.6.
    vector<double> lo = {0.1, 0.3}, up = {0.7, 0.3}, V = {0.0, 1.0};
    auto rmin = optimize(lo, up, V, Sense::Min);
    CHECK(rmin.p[0] == doctest::Approx(0.7));
    CHECK(rmin.p[1] == doctest::Approx(0.3));
    CHECK(rmin.value == doctest::Approx(0.3));
}

TEST_CASE("omax: infeasible boxes throw") {
    CHECK_THROWS_AS(optimize({0.6, 0.6}, {1.0, 1.0}, {0.0, 1.0}, Sense::Min),
                    std::invalid_argument);                       // sum(lower) > 1
    CHECK_THROWS_AS(optimize({0.0, 0.0}, {0.3, 0.3}, {0.0, 1.0}, Sense::Max),
                    std::invalid_argument);                       // sum(upper) < 1
    CHECK_THROWS_AS(optimize({0.5}, {0.4}, {1.0}, Sense::Min),
                    std::invalid_argument);                       // lower > upper
    CHECK_THROWS_AS(optimize({0.0, 0.0}, {1.0}, {0.0, 1.0}, Sense::Min),
                    std::invalid_argument);                       // size mismatch
}

TEST_CASE("omax: randomized differential vs brute-force vertex enumeration") {
    std::mt19937 rng(12345);                       // fixed seed -> deterministic
    std::uniform_int_distribution<int> ndist(1, 5);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    int checked = 0;
    for (int trial = 0; trial < 2000 && checked < 500; ++trial) {
        int n = ndist(rng);
        vector<double> lo(n), up(n), V(n);
        for (int i = 0; i < n; ++i) {
            double a = u01(rng) * 0.5;          // lower in [0,0.5]
            double b = a + u01(rng) * 0.6;      // upper >= lower
            lo[i] = a; up[i] = std::min(b, 1.0);
            V[i] = u01(rng);
        }
        double sl = 0, su = 0;
        for (int i = 0; i < n; ++i) { sl += lo[i]; su += up[i]; }
        if (sl > 1.0 || su < 1.0) continue;      // skip infeasible boxes
        ++checked;
        for (Sense s : {Sense::Min, Sense::Max}) {
            auto r = optimize(lo, up, V, s);
            check_valid_distribution(r.p, lo, up);
            double oracle = brute_optimum(lo, up, V, s);
            CHECK(r.value == doctest::Approx(oracle).epsilon(1e-9));
        }
    }
    CHECK(checked > 100); // sanity: we actually exercised many feasible instances
}
