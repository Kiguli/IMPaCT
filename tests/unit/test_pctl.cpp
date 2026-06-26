// ============================================================================
// CONTRACT TESTS — PCTL / CTL model checking on Interval MDPs.
// Verified by reduction to the already-verified reach/safety solver + hand cases +
// a differential oracle (explicit finite-horizon DP) for bounded-until.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <set>
#include <vector>
#include <cstdint>

using namespace impact::solve;
using namespace impact::pctl;

static const double EPS = 1e-7;
static double mid(const IntervalResult& r, int s) { return 0.5 * (r.lower[s] + r.upper[s]); }

TEST_CASE("pctl next: X psi one-step probability") {
    IMDPModel m = { /*0*/ {{ {1,0.5,0.5}, {2,0.5,0.5} }}, /*1*/ {{ {1,1,1} }}, /*2*/ {{ {2,1,1} }} };
    auto r = nextPessimistic(m, {1}, EPS);
    CHECK(mid(r, 0) == doctest::Approx(0.5));
    CHECK(mid(r, 1) == doctest::Approx(1.0));   // already in psi after the step
    CHECK(mid(r, 2) == doctest::Approx(0.0));
}

TEST_CASE("pctl until: phi=all reduces to reachability (F psi)") {
    IMDPModel m = {
        /*0*/ {{ {1,0.4,0.6}, {2,0.4,0.6} }},
        /*1*/ {{ {3,1,1} }}, /*2*/ {{ {2,1,1} }}, /*3*/ {{ {3,1,1} }},
    };
    std::set<int> all = {0,1,2,3}, target = {3};
    auto u = untilPessimistic(m, all, target, EPS);
    auto f = maxReachPessimistic(m, target, EPS);
    for (int s = 0; s < 4; ++s) CHECK(mid(u, s) == doctest::Approx(mid(f, s)).epsilon(1e-6));
    CHECK(mid(u, 0) == doctest::Approx(0.4).epsilon(1e-3));
}

TEST_CASE("pctl until: constraint phi excludes the only route -> 0") {
    // 0 -> 1 -> 2 (target). phi = {0,2} excludes 1, so 2 is unreachable while staying in phi.
    IMDPModel m = { /*0*/ {{ {1,1,1} }}, /*1*/ {{ {2,1,1} }}, /*2*/ {{ {2,1,1} }} };
    auto u = untilPessimistic(m, {0,2}, {2}, EPS);
    CHECK(mid(u, 0) == doctest::Approx(0.0));
    auto u2 = untilPessimistic(m, {0,1,2}, {2}, EPS);   // route allowed
    CHECK(mid(u2, 0) == doctest::Approx(1.0));
}

TEST_CASE("pctl bounded until: reachable in exactly d steps iff k>=d") {
    // chain 0->1->2->3 (target). From 0 it takes 3 steps.
    IMDPModel m = { {{ {1,1,1} }}, {{ {2,1,1} }}, {{ {3,1,1} }}, {{ {3,1,1} }} };
    std::set<int> all = {0,1,2,3}, t = {3};
    CHECK(mid(boundedUntilPessimistic(m, all, t, 2, EPS), 0) == doctest::Approx(0.0));
    CHECK(mid(boundedUntilPessimistic(m, all, t, 3, EPS), 0) == doctest::Approx(1.0));
    CHECK(mid(boundedUntilPessimistic(m, all, t, 5, EPS), 0) == doctest::Approx(1.0));
}

TEST_CASE("pctl bounded until -> unbounded until as k grows") {
    IMDPModel m = {
        /*0*/ {{ {0,0.5,0.5}, {1,0.5,0.5} }},   // geometric reach of 1
        /*1*/ {{ {1,1,1} }},
    };
    std::set<int> all = {0,1}, t = {1};
    double ub = mid(untilPessimistic(m, all, t, EPS), 0);
    double bk = mid(boundedUntilPessimistic(m, all, t, 60, EPS), 0);
    CHECK(ub == doctest::Approx(1.0).epsilon(1e-3));
    CHECK(bk == doctest::Approx(ub).epsilon(1e-3));
}

TEST_CASE("pctl threshold verdicts (sound interval)") {
    CHECK(check(0.4, 0.6, Cmp::Ge, 0.3) == Verdict::Sat);
    CHECK(check(0.4, 0.6, Cmp::Ge, 0.7) == Verdict::Unsat);
    CHECK(check(0.4, 0.6, Cmp::Ge, 0.5) == Verdict::Unknown);   // straddles threshold
    CHECK(check(0.4, 0.6, Cmp::Lt, 0.7) == Verdict::Sat);
    CHECK(check(0.4, 0.6, Cmp::Gt, 0.6) == Verdict::Unsat);
}

TEST_CASE("pctl CTL sugar: EF and AG") {
    // 0 -> 1(goal) w.p.0.5 / 2(sink) w.p.0.5
    IMDPModel m = { {{ {1,0.5,0.5}, {2,0.5,0.5} }}, {{ {1,1,1} }}, {{ {2,1,1} }} };
    auto ef = EF(m, {1}, EPS);
    CHECK(ef.count(0)); CHECK(ef.count(1)); CHECK(!ef.count(2));   // 2 can never reach 1
    // AG over phi={0,1}: from 0 you may fall into 2 (¬phi), so not AG; state 1 stays in phi forever.
    auto ag = AG(m, {0,1}, EPS);
    CHECK(ag.count(1)); CHECK(!ag.count(0));
}

// ---- differential oracle: bounded-until on POINT MDPs via explicit DP ----------
namespace {
struct Lcg { uint64_t s; uint32_t nx(){ s=s*6364136223846793005ULL+1442695040888963407ULL; return (uint32_t)(s>>33);} int in(int a,int b){return a+(int)(nx()%(uint32_t)(b-a+1));} };
}
TEST_CASE("pctl bounded until: differential vs explicit DP on random point MDPs") {
    Lcg rng{0xD1CE5BULL};
    for (int t = 0; t < 600; ++t) {
        int n = rng.in(2, 5);
        IMDPModel m(n);
        for (int s = 0; s < n; ++s) {
            int na = rng.in(1, 2);
            for (int a = 0; a < na; ++a) {
                int k = rng.in(1, n);
                // a point distribution: split mass 1 over k random successors
                std::vector<int> tos; std::vector<double> w; double sum = 0;
                for (int e = 0; e < k; ++e) { tos.push_back(rng.in(0, n-1)); double x = rng.in(1,9); w.push_back(x); sum += x; }
                ActionDist act;
                for (int e = 0; e < k; ++e) { double p = w[e]/sum; act.push_back({tos[e], p, p}); }
                m[s].push_back(std::move(act));
            }
        }
        std::set<int> phi, psi;
        for (int s = 0; s < n; ++s) { if (rng.in(0,3)) phi.insert(s); if (rng.in(0,3)==0) psi.insert(s); }
        int K = rng.in(0, 6);

        // explicit DP oracle (point MDP: controller maxes expected value; no nature)
        std::vector<double> V(n, 0.0); for (int s : psi) V[s] = 1.0;
        for (int it = 0; it < K; ++it) {
            std::vector<double> Vn(n, 0.0);
            for (int s = 0; s < n; ++s) {
                if (psi.count(s)) { Vn[s] = 1.0; continue; }
                if (!phi.count(s)) { Vn[s] = 0.0; continue; }
                double best = 0.0;
                for (const auto& act : m[s]) { double q = 0; for (const auto& iv : act) q += iv.lo * V[iv.to]; best = std::max(best, q); }
                Vn[s] = best;
            }
            V = std::move(Vn);
        }
        auto r = boundedUntilPessimistic(m, phi, psi, K, EPS);
        for (int s = 0; s < n; ++s) CHECK(r.lower[s] == doctest::Approx(V[s]).epsilon(1e-9));
    }
}
