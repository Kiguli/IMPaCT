// ============================================================================
// CONTRACT TESTS — sparse interval-MDP abstraction (ISSUE-0006 / scalability).
//  * kernel transitionInterval1D verified vs brute-force sampling of the mean,
//  * pruning is lossless: sparse synthesis == dense-window synthesis,
//  * sparsity/scaling: stored nonzeros per state stay bounded as the grid grows
//    (O(nnz) memory, like IntervalMDP.jl) — the fix for the dense-matrix OOM.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace impact::abstraction;

TEST_CASE("abstraction kernel: transitionInterval1D matches brute-force over mean range") {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> U(-3.0, 3.0), S(0.1, 1.5), Wd(0.2, 2.0), Md(-4.0, 4.0);
    for (int t = 0; t < 1500; ++t) {
        double a = U(rng), b = a + Wd(rng);
        double sigma = S(rng);
        double m1 = Md(rng), m2 = m1 + Wd(rng);
        Bound bd = transitionInterval1D(m1, m2, sigma, a, b);
        double bmin = 2.0, bmax = -1.0;
        const int K = 4000;
        for (int s = 0; s <= K; ++s) {
            double mu = m1 + (m2 - m1) * s / K;
            double mass = massInInterval(mu, sigma, a, b);
            bmin = std::min(bmin, mass);
            bmax = std::max(bmax, mass);
        }
        CHECK(bd.lo == doctest::Approx(bmin).epsilon(2e-3));
        CHECK(bd.hi >= bmax - 2e-3);           // closed-form peak >= any sample
        CHECK(bd.hi <= bmax + 2e-3);           // and not above the true peak
        CHECK(bd.lo <= bd.hi + 1e-12);
        CHECK(bd.lo >= -1e-12);
        CHECK(bd.hi <= 1.0 + 1e-12);
    }
}

TEST_CASE("abstraction kernel: box bound == product of 1-D bounds") {
    std::vector<double> muLo{-1, 0.5}, muHi{0.2, 1.0}, sig{0.4, 0.6}, aLo{-0.5, 0.0}, aHi{0.5, 1.5};
    Bound box = transitionIntervalBox(muLo, muHi, sig, aLo, aHi);
    Bound b0 = transitionInterval1D(muLo[0], muHi[0], sig[0], aLo[0], aHi[0]);
    Bound b1 = transitionInterval1D(muLo[1], muHi[1], sig[1], aLo[1], aHi[1]);
    CHECK(box.lo == doctest::Approx(b0.lo * b1.lo));
    CHECK(box.hi == doctest::Approx(b0.hi * b1.hi));
}

static System1D demoSystem() {
    System1D s;
    s.a = 1.0; s.b = 1.0; s.sigma = 0.5;
    s.xlb = -3.0; s.xub = 3.0; s.eta = 0.5;
    s.ulb = -1.0; s.uub = 1.0; s.ueta = 0.5;
    s.tlo = 2.0; s.thi = 3.0;
    return s;
}

TEST_CASE("abstraction: pruning is lossless (sparse synthesis == dense-window synthesis)") {
    System1D sys = demoSystem();
    SparseReach dense  = buildSparseReach1D(sys, /*prune=*/0.0);
    SparseReach sparse = buildSparseReach1D(sys, /*prune=*/1e-7);
    REQUIRE(dense.nCells == sparse.nCells);

    auto rd = impact::solve::maxReachOptimistic(dense.model, dense.targets, 1e-7);
    auto rs = impact::solve::maxReachOptimistic(sparse.model, sparse.targets, 1e-7);
    for (int i = 0; i < dense.nCells; ++i) {
        double vd = 0.5 * (rd.lower[i] + rd.upper[i]);
        double vs = 0.5 * (rs.lower[i] + rs.upper[i]);
        CHECK(std::fabs(vd - vs) < 2e-3);          // pruning below 1e-7 changes value negligibly
        CHECK(vd >= -1e-9);
        CHECK(vd <= 1.0 + 1e-9);
    }
    // sparse stores strictly fewer nonzeros than the dense-window build
    CHECK(sparse.nnz <= dense.nnz);
}

TEST_CASE("abstraction: sanity — a cell inside the target reaches w.p. 1") {
    System1D sys = demoSystem();
    SparseReach r = buildSparseReach1D(sys, 1e-9);
    auto val = impact::solve::maxReachOptimistic(r.model, r.targets, 1e-7);
    // cell whose centre is in [2,3] is an absorbing target cell -> value 1
    int target_cell = (int)std::floor((2.25 - sys.xlb) / sys.eta);
    CHECK(0.5 * (val.lower[target_cell] + val.upper[target_cell]) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("abstraction: O(N) sparsity — grow the domain at fixed step/noise, nnz/cell bounded") {
    // The honest sparsity win: with eta and sigma FIXED the Gaussian kernel spans a
    // constant number of cells, so as the domain (and hence state count) grows the
    // stored nonzeros per state stay bounded => total nnz is O(N), not O(N^2).
    // (Refining eta at fixed sigma would instead widen the kernel-in-cells.)
    for (double half : {3.0, 15.0, 75.0, 375.0}) {
        System1D sys = demoSystem();                // eta=0.5, sigma=0.5 fixed
        sys.xlb = -half; sys.xub = half;
        sys.tlo = half - 1.0; sys.thi = half;
        SparseReach r = buildSparseReach1D(sys, 1e-7);
        double ratio = (double)r.nnz / (double)(r.nCells + 2);
        CHECK(ratio < 80.0);                         // window-limited, independent of N
        CHECK(r.nCells == (int)std::llround(2 * half / sys.eta));
    }
}
