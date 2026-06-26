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

// --- n-D abstraction (anchored on the verified 1-D builder) -----------------
static SystemND demo2D(bool coupled) {
    SystemND s;
    s.dim_x = 2; s.dim_u = 2;
    s.xlb = {-3, -3}; s.xub = {3, 3}; s.eta = {0.5, 0.5};
    s.ulb = {-1, -1}; s.uub = {1, 1}; s.ueta = {2.0, 2.0};   // 2 pts/dim -> 4 actions
    s.A = coupled ? std::vector<std::vector<double>>{{0.8, 0.1}, {0.1, 0.8}}
                  : std::vector<std::vector<double>>{{0.8, 0.0}, {0.0, 0.8}};
    s.B = {{1, 0}, {0, 1}};
    s.c = {0, 0};
    s.sigma = {0.5, 0.5};
    s.tlo = {2, 2}; s.thi = {3, 3};
    return s;
}

TEST_CASE("abstraction nD: dim=1 reproduces the verified 1-D builder") {
    System1D s = demoSystem();
    SparseReach r1 = buildSparseReach1D(s, 1e-9);
    SystemND nd;
    nd.dim_x = 1; nd.dim_u = 1;
    nd.xlb = {s.xlb}; nd.xub = {s.xub}; nd.eta = {s.eta};
    nd.ulb = {s.ulb}; nd.uub = {s.uub}; nd.ueta = {s.ueta};
    nd.A = {{s.a}}; nd.B = {{s.b}}; nd.c = {0.0}; nd.sigma = {s.sigma};
    nd.tlo = {s.tlo}; nd.thi = {s.thi};
    SparseReach rn = buildSparseReachND(nd, 1e-9);
    REQUIRE(r1.nCells == rn.nCells);
    auto v1 = impact::solve::maxReachOptimistic(r1.model, r1.targets, 1e-7);
    auto vn = impact::solve::maxReachOptimistic(rn.model, rn.targets, 1e-7);
    for (int i = 0; i < r1.nCells; ++i)
        CHECK(0.5 * (v1.lower[i] + v1.upper[i])
              == doctest::Approx(0.5 * (vn.lower[i] + vn.upper[i])).epsilon(1e-6));
}

TEST_CASE("abstraction nD: 2-D decoupled, pruning lossless (sparse == dense)") {
    SystemND s = demo2D(false);
    SparseReach dense = buildSparseReachND(s, 0.0);
    SparseReach sparse = buildSparseReachND(s, 1e-7);
    auto rd = impact::solve::maxReachOptimistic(dense.model, dense.targets, 1e-7);
    auto rs = impact::solve::maxReachOptimistic(sparse.model, sparse.targets, 1e-7);
    for (int i = 0; i < dense.nCells; ++i)
        CHECK(std::fabs(0.5 * (rd.lower[i] + rd.upper[i]) - 0.5 * (rs.lower[i] + rs.upper[i])) < 3e-3);
    CHECK(sparse.nnz <= dense.nnz);
}

TEST_CASE("abstraction nD: 2-D coupled affine is sound; target reaches w.p. 1") {
    SystemND s = demo2D(true);
    SparseReach r = buildSparseReachND(s, 1e-9);
    auto v = impact::solve::maxReachOptimistic(r.model, r.targets, 1e-7);
    double mx = 0.0;
    for (int i = 0; i < r.nCells; ++i) {
        CHECK(v.lower[i] <= v.upper[i] + 1e-9);
        CHECK(v.lower[i] >= -1e-9);
        CHECK(v.upper[i] <= 1.0 + 1e-9);
        mx = std::max(mx, 0.5 * (v.lower[i] + v.upper[i]));
    }
    CHECK(mx == doctest::Approx(1.0).epsilon(1e-6));   // target cells reach w.p. 1
}

TEST_CASE("abstraction nD: 2-D O(N) sparsity (grow domain, bounded nnz/cell)") {
    // nnz/cell converges to a CONSTANT (~ 2-D kernel window x #actions, independent of
    // N) as the boundary fraction shrinks; the rise 3->6->12 is the boundary effect
    // converging, not unbounded growth. Bounded => total nnz is O(N).
    double prev = 0.0;
    for (double half : {6.0, 12.0, 24.0}) {
        SystemND s = demo2D(false);
        s.xlb = {-half, -half}; s.xub = {half, half};
        s.tlo = {half - 1, half - 1}; s.thi = {half, half};
        SparseReach r = buildSparseReachND(s, 1e-7);
        double ratio = (double)r.nnz / (double)(r.nCells + 2);
        CHECK(ratio < 700.0);                       // bounded by full 2-D window x actions
        if (prev > 0.0) CHECK(ratio < prev * 1.25); // converging (not growing ~linearly)
        prev = ratio;
    }
}

// --- nonlinear dynamics via interval arithmetic: ARCH Van der Pol -------------
// x0' = x0 + 0.1 x1 ; x1' = x1 + 0.1(-x0 + (1-x0)^2 x1).  No input (verification).
static void vpMean(const std::vector<double>& cl, const std::vector<double>& ch,
                   const std::vector<double>&, std::vector<double>& muLo, std::vector<double>& muHi) {
    Ival X0(cl[0], ch[0]), X1(cl[1], ch[1]);
    Ival f0 = X0 + 0.1 * X1;
    Ival f1 = X1 + 0.1 * ((-1.0 * X0) + isquare(Ival(1.0) - X0) * X1);
    muLo = {f0.lo, f1.lo}; muHi = {f0.hi, f1.hi};
}

TEST_CASE("abstraction nonlinear: Van der Pol interval mean bound is SOUND") {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> X(-5.0, 4.8);
    for (int t = 0; t < 800; ++t) {
        double x0l = X(rng), x1l = X(rng);
        std::vector<double> cl{x0l, x1l}, ch{x0l + 0.2, x1l + 0.2}, muLo, muHi, dummy;
        vpMean(cl, ch, dummy, muLo, muHi);
        for (int s0 = 0; s0 <= 12; ++s0) for (int s1 = 0; s1 <= 12; ++s1) {
            double x0 = cl[0] + (ch[0]-cl[0]) * s0 / 12.0;
            double x1 = cl[1] + (ch[1]-cl[1]) * s1 / 12.0;
            double f0 = x0 + 0.1 * x1;
            double f1 = x1 + 0.1 * (-x0 + (1 - x0)*(1 - x0)*x1);
            CHECK(f0 >= muLo[0] - 1e-9); CHECK(f0 <= muHi[0] + 1e-9);   // enclosure is sound
            CHECK(f1 >= muLo[1] - 1e-9); CHECK(f1 <= muHi[1] + 1e-9);
        }
    }
}

TEST_CASE("abstraction nonlinear: Van der Pol sparse interval-MC build is sound") {
    GridSpec g;
    g.dim_x = 2; g.dim_u = 0;
    g.xlb = {-5, -5}; g.xub = {5, 5}; g.eta = {0.2, 0.2};
    g.sigma = {0.2, 0.2};
    g.tlo = {-1.2, -2.9}; g.thi = {-0.9, -2.0};
    SparseReach r = buildSparseReachGeneral(g, vpMean, 1e-7);
    CHECK(r.nCells == 2500);
    auto lo = impact::solve::maxReachPessimistic(r.model, r.targets, 1e-6);
    auto hi = impact::solve::maxReachOptimistic(r.model, r.targets, 1e-6);
    double mxhi = 0.0;
    for (int i = 0; i < r.nCells; ++i) {
        CHECK(lo.lower[i] <= hi.upper[i] + 1e-9);     // pessimistic <= optimistic (valid interval)
        CHECK(lo.lower[i] >= -1e-9);
        CHECK(hi.upper[i] <= 1.0 + 1e-9);
        mxhi = std::max(mxhi, hi.upper[i]);
    }
    CHECK(mxhi == doctest::Approx(1.0).epsilon(1e-6));  // target cells reach w.p. 1
}

// --- integration: sparse abstraction + LTLf DFA + product (Phase 2 over continuous) ---
TEST_CASE("integration: 'F region' over the abstraction == plain reachability to region cells") {
    // Full-dynamics IMDP (empty target box) of a small 2-D affine system.
    SystemND s = demo2D(false);
    s.xlb = {-1, -1}; s.xub = {1, 1}; s.eta = {0.5, 0.5};   // tiny 4x4 grid -> fast
    s.ulb = {-1, -1}; s.uub = {1, 1}; s.ueta = {2.0, 2.0};
    s.tlo = {1e18, 1e18}; s.thi = {-1e18, -1e18};      // empty -> full IMDP
    SparseReach ab = buildSparseReachND(s, 1e-9);
    const int N = ab.nCells, Nd0 = (int)std::llround((s.xub[0]-s.xlb[0])/s.eta[0]);

    // region R = [2,3]x[2,3]; label cells by centre; reach-target = R cells.
    auto centre = [&](int c, double& x0, double& x1) {
        int j0 = c % Nd0, j1 = c / Nd0;
        x0 = s.xlb[0] + (j0+0.5)*s.eta[0]; x1 = s.xlb[1] + (j1+0.5)*s.eta[1];
    };
    std::vector<impact::ltl::Letter> labels(ab.model.size());
    std::set<int> Rcells;
    for (int c = 0; c < N; ++c) { double x0,x1; centre(c,x0,x1);
        if (x0>=0.4&&x0<=1.0&&x1>=0.4&&x1<=1.0) { labels[c].insert("r"); Rcells.insert(c); } }

    auto* aut = impact::ltl::compileFinite("F r", {"r"});
    impact::ltl::DFA dfa = impact::ltl::toDFA(aut);
    auto Pr = impact::product::build(ab.model, labels, dfa, 0);
    auto vp = impact::solve::maxReachOptimistic(Pr.model, Pr.targets, 1e-7);
    auto vr = impact::solve::maxReachOptimistic(ab.model, Rcells, 1e-7);   // plain reach to R

    const int nQ = dfa.nStates;
    for (int c = 0; c < N; ++c) {
        int q = dfa.trans[dfa.start][impact::ltl::letterIndex(dfa, labels[c])];
        double prod = 0.5*(vp.lower[c*nQ+q] + vp.upper[c*nQ+q]);
        double reach = 0.5*(vr.lower[c] + vr.upper[c]);
        CHECK(std::fabs(prod - reach) < 3e-3);   // F r  ==  reach R
    }
    impact::ltl::destroy(aut);
}
