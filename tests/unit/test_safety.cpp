// ============================================================================
// CONTRACT TESTS — safety synthesis: max P(never reach `avoid`) = 1 - min-reach.
// Verified by hand cases + a differential vs reachability on single-action models
// (where the controller has no choice, so safety == 1 - reach).
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <set>
#include <vector>
#include <random>
#include <cmath>

using namespace impact::solve;

static const double EPS = 1e-7;
static double mid(const IntervalResult& r, int s) { return 0.5*(r.lower[s]+r.upper[s]); }

TEST_CASE("safety: controller can avoid -> safety 1") {
    // 0: a->1(avoid) | b->2(safe sink) ; 1 avoid sink ; 2 safe sink
    IMDPModel m = { {{ {1,1,1} }, { {2,1,1} }}, {{ {1,1,1} }}, {{ {2,1,1} }} };
    auto sf = maxSafetyPessimistic(m, {1}, EPS);
    CHECK(mid(sf, 0) == doctest::Approx(1.0).epsilon(1e-6));   // pick b
    CHECK(mid(sf, 2) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(mid(sf, 1) == doctest::Approx(0.0).epsilon(1e-6));   // already in avoid
}

TEST_CASE("safety: avoid unavoidable -> safety 0") {
    IMDPModel m = { {{ {1,1,1} }}, {{ {1,1,1} }} };            // 0 -> 1(avoid) forced
    auto sf = maxSafetyPessimistic(m, {1}, EPS);
    CHECK(mid(sf, 0) == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("safety: coin to avoid -> safety 0.5") {
    IMDPModel m = { {{ {1,0.5,0.5}, {2,0.5,0.5} }}, {{ {1,1,1} }}, {{ {2,1,1} }} };
    auto sf = maxSafetyPessimistic(m, {1}, EPS);
    CHECK(mid(sf, 0) == doctest::Approx(0.5).epsilon(1e-3));
}

TEST_CASE("safety: point MDP pessimistic == optimistic") {
    IMDPModel m = { {{ {1,0.3,0.3}, {2,0.7,0.7} }}, {{ {1,1,1} }}, {{ {2,1,1} }} };
    auto sp = maxSafetyPessimistic(m, {1}, EPS);
    auto so = maxSafetyOptimistic(m, {1}, EPS);
    for (int s = 0; s < 3; ++s) CHECK(mid(sp,s) == doctest::Approx(mid(so,s)).epsilon(1e-4));
}

TEST_CASE("safety: differential vs reachability on single-action interval MDPs") {
    // With one action per state the controller has no choice, so
    // safety == 1 - reach(avoid) (nature adversarial both ways).
    std::mt19937 rng(909);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    int checked = 0;
    for (int trial = 0; trial < 4000 && checked < 400; ++trial) {
        int n = 2 + (int)(rng() % 5);
        std::set<int> avoid; avoid.insert((int)(rng() % n));
        IMDPModel m(n);
        for (int s = 0; s < n; ++s) {
            std::vector<int> succ;
            for (int t = 0; t < n; ++t) if (rng() & 1u) succ.push_back(t);
            if (succ.empty()) succ.push_back((int)(rng() % n));
            std::vector<double> w(succ.size()); double sum = 0;
            for (double& x : w) { x = u01(rng) + 1e-3; sum += x; }
            ActionDist act;
            for (size_t k = 0; k < succ.size(); ++k) { double p = w[k]/sum;
                double r = 0.1 * u01(rng);
                act.push_back({succ[k], std::max(0.0,p-r), std::min(1.0,p+r)}); }
            m[s].push_back(act);                              // exactly one action
        }
        ++checked;
        auto sf = maxSafetyPessimistic(m, avoid, EPS);
        auto rc = maxReachOptimistic(m, avoid, EPS);         // single action: nature-max reach
        for (int s = 0; s < n; ++s) {
            CHECK(sf.lower[s] <= sf.upper[s] + 1e-9);
            CHECK(std::fabs(mid(sf, s) - (1.0 - mid(rc, s))) < 3e-3);   // safety == 1 - reach
        }
    }
    CHECK(checked > 150);
}
