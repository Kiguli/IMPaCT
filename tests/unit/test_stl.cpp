// ============================================================================
// CONTRACT TESTS — bounded Signal Temporal Logic on Interval MDPs.
// Atomic-predicate->cell mapping, the bounded temporal operators (reduced to the
// verified finite-horizon DP), and a differential oracle for bounded-globally.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <set>
#include <vector>
#include <cstdint>

using namespace impact::solve;
namespace stl = impact::stl;

static double mid(const IntervalResult& r, int s) { return 0.5 * (r.lower[s] + r.upper[s]); }

TEST_CASE("stl predicateCells: mu(x)=x>0 over a 1-D grid") {
    // grid x in [-1,1], eta 0.5, 4 cells: centres -0.75,-0.25,0.25,0.75 -> cells 2,3 satisfy x>0.
    auto cells = stl::predicateCells({-1.0}, {0.5}, {4}, [](const std::vector<double>& x){ return x[0] > 0; });
    CHECK(cells == std::set<int>{2, 3});
}

TEST_CASE("stl predicateCells: 2-D box predicate, row-major linearization") {
    // 3x3 grid, lb=(0,0), eta=1 ; centres at .5,1.5,2.5. Predicate x0>1 && x1>1 holds
    // for centres 1.5 and 2.5 in each dim -> j0,j1 in {1,2}. lin = j0 + j1*3 (stride_0=1)
    // -> {1+3, 2+3, 1+6, 2+6} = {4,5,7,8}.
    auto cells = stl::predicateCells({0.0,0.0}, {1.0,1.0}, {3,3},
                    [](const std::vector<double>& x){ return x[0] > 1 && x[1] > 1; });
    CHECK(cells == std::set<int>{4,5,7,8});
    // a tighter predicate selecting the single top-right cell (centre 2.5,2.5 -> lin 8)
    auto one = stl::predicateCells({0.0,0.0}, {1.0,1.0}, {3,3},
                    [](const std::vector<double>& x){ return x[0] > 2 && x[1] > 2; });
    CHECK(one == std::set<int>{8});
}

TEST_CASE("stl F_[0,b]: bounded eventually == bounded reach in b steps") {
    IMDPModel m = { {{ {1,1,1} }}, {{ {2,1,1} }}, {{ {3,1,1} }}, {{ {3,1,1} }} };  // chain 0->1->2->3
    CHECK(mid(stl::eventuallyBounded(m, {3}, 2, true), 0) == doctest::Approx(0.0));
    CHECK(mid(stl::eventuallyBounded(m, {3}, 3, true), 0) == doctest::Approx(1.0));
}

TEST_CASE("stl G_[0,b]: bounded always") {
    // psi = {0,1}. chain 0->1->2 leaves psi at step 2. G holds for b<=1, fails for b>=2 from state 0.
    IMDPModel m = { {{ {1,1,1} }}, {{ {2,1,1} }}, {{ {2,1,1} }} };
    CHECK(mid(stl::globallyBounded(m, {0,1}, 0, true), 0) == doctest::Approx(1.0));
    CHECK(mid(stl::globallyBounded(m, {0,1}, 1, true), 0) == doctest::Approx(1.0));   // states 0,1 both in psi
    CHECK(mid(stl::globallyBounded(m, {0,1}, 2, true), 0) == doctest::Approx(0.0));   // reaches 2 (∉psi)
    // psi = all -> always 1
    CHECK(mid(stl::globallyBounded(m, {0,1,2}, 5, true), 0) == doctest::Approx(1.0));
}

TEST_CASE("stl G_[0,b]: robust vs optimistic when nature can break invariance") {
    // psi = {0,1}; at 0, nature sends >=0.5 to 1 and <=0.5 to 2 (∉psi).
    IMDPModel m = { {{ {1,0.5,1.0}, {2,0.0,0.5} }}, {{ {1,1,1} }}, {{ {2,1,1} }} };
    auto gp = stl::globallyBounded(m, {0,1}, 1, true);    // robust: nature maximizes leaving -> 0.5
    auto go = stl::globallyBounded(m, {0,1}, 1, false);   // optimistic: nature keeps in psi -> 1
    CHECK(mid(gp, 0) == doctest::Approx(0.5).epsilon(1e-3));
    CHECK(mid(go, 0) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("stl F_[a,b] window: free evolution then bounded reach") {
    // chain 0->1->2->3->4 (target 4). F_[2,4] from 0: after 2 free steps at state 2, reach 4 within 2 -> yes.
    IMDPModel m = { {{ {1,1,1} }}, {{ {2,1,1} }}, {{ {3,1,1} }}, {{ {4,1,1} }}, {{ {4,1,1} }} };
    CHECK(mid(stl::eventuallyWindow(m, {4}, 2, 4, true), 0) == doctest::Approx(1.0));
    // F_[0,3] cannot reach 4 (needs 4 steps) -> 0
    CHECK(mid(stl::eventuallyBounded(m, {4}, 3, true), 0) == doctest::Approx(0.0));
}

// ---- differential oracle: bounded-globally on POINT MDPs via explicit DP --------
namespace { struct Lcg { uint64_t s; uint32_t nx(){ s=s*6364136223846793005ULL+1442695040888963407ULL; return (uint32_t)(s>>33);} int in(int a,int b){return a+(int)(nx()%(uint32_t)(b-a+1));} }; }

TEST_CASE("stl G_[0,b]: differential vs explicit DP on random point MDPs") {
    Lcg rng{0x5713ULL};
    for (int t = 0; t < 500; ++t) {
        int n = rng.in(2, 5);
        IMDPModel m(n);
        for (int s = 0; s < n; ++s) {
            int na = rng.in(1, 2);
            for (int a = 0; a < na; ++a) {
                int k = rng.in(1, n); std::vector<int> tos; std::vector<double> w; double sum = 0;
                for (int e = 0; e < k; ++e) { tos.push_back(rng.in(0,n-1)); double x = rng.in(1,9); w.push_back(x); sum += x; }
                ActionDist act; for (int e = 0; e < k; ++e) { double p = w[e]/sum; act.push_back({tos[e], p, p}); }
                m[s].push_back(std::move(act));
            }
        }
        std::set<int> psi; for (int s = 0; s < n; ++s) if (rng.in(0,2)) psi.insert(s);
        int b = rng.in(0, 6);
        // explicit DP: W0[s]=[s∈psi]; Wt[s]=[s∈psi]*max_a sum p*W{t-1}
        std::vector<double> W(n, 0.0); for (int s = 0; s < n; ++s) W[s] = psi.count(s) ? 1.0 : 0.0;
        for (int it = 0; it < b; ++it) {
            std::vector<double> Wn(n, 0.0);
            for (int s = 0; s < n; ++s) {
                if (!psi.count(s)) { Wn[s] = 0.0; continue; }
                double best = 0.0; for (const auto& act : m[s]) { double q = 0; for (const auto& iv : act) q += iv.lo*W[iv.to]; best = std::max(best,q); }
                Wn[s] = best;
            }
            W = std::move(Wn);
        }
        auto r = stl::globallyBounded(m, psi, b, true);
        for (int s = 0; s < n; ++s) CHECK(r.lower[s] == doctest::Approx(W[s]).epsilon(1e-9));
    }
}
