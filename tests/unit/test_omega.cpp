// ============================================================================
// CONTRACT TESTS — Phase 3: omega-regular (Büchi) via accepting-MEC reachability.
// Verified by reduction to the already-verified reachability solver + hand cases.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <set>
#include <vector>
#include <algorithm>

using namespace impact::solve;
using impact::omega::maxBuchiOptimistic;
using impact::omega::maxBuchiPessimistic;
using impact::omega::acceptingMECStates;

static const double EPS = 1e-7;

TEST_CASE("buchi: accepting absorbing state -> equals reachability of it") {
    // 0 ->1 @0.5, ->2(sink) @0.5 ; 1 self-loop (accepting) ; 2 sink.
    IMDPModel m = {
        /*0*/ {{ {1,0.5,0.5}, {2,0.5,0.5} }},
        /*1*/ {{ {1,1,1} }},        // accepting absorbing
        /*2*/ {{ {2,1,1} }},
    };
    auto b = maxBuchiOptimistic(m, {1}, EPS);
    auto r = maxReachOptimistic(m, {1}, EPS);   // {1} is a self-loop MEC
    for (int s = 0; s < 3; ++s)
        CHECK(0.5*(b.lower[s]+b.upper[s]) == doctest::Approx(0.5*(r.lower[s]+r.upper[s])).epsilon(1e-6));
    CHECK(0.5*(b.lower[0]+b.upper[0]) == doctest::Approx(0.5));
}

TEST_CASE("buchi: GF(true) = 1 (every path ends in a MEC)") {
    IMDPModel m = { /*0*/ {{ {1,1,1} }}, /*1*/ {{ {1,1,1} }} };
    auto b = maxBuchiOptimistic(m, {0, 1}, EPS);
    CHECK(0.5*(b.lower[0]+b.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(0.5*(b.lower[1]+b.upper[1]) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("buchi: transient accepting state (not in any MEC) -> value 0") {
    // 0 (accepting, transient) -> 1 ; 1 self-loop (not accepting).
    IMDPModel m = { /*0*/ {{ {1,1,1} }}, /*1*/ {{ {1,1,1} }} };
    auto b = maxBuchiOptimistic(m, {0}, EPS);
    CHECK(0.5*(b.lower[0]+b.upper[0]) == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("buchi: recurrence in an end component, reached with prob 0.5") {
    // EC {0,1} (0<->1), 0 accepting ; start 2 ->0 @0.5, ->3(sink) @0.5 ; 3 sink.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {0,1,1} }},
        /*2*/ {{ {0,0.5,0.5}, {3,0.5,0.5} }},
        /*3*/ {{ {3,1,1} }},
    };
    auto b = maxBuchiOptimistic(m, {0}, EPS);
    CHECK(0.5*(b.lower[0]+b.upper[0]) == doctest::Approx(1.0).epsilon(1e-6));   // inside the EC
    CHECK(0.5*(b.lower[1]+b.upper[1]) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(0.5*(b.lower[2]+b.upper[2]) == doctest::Approx(0.5).epsilon(1e-3));   // start
    CHECK(0.5*(b.lower[3]+b.upper[3]) == doctest::Approx(0.0).epsilon(1e-6));   // sink
}

TEST_CASE("buchi: acceptingMECStates picks the right components") {
    // MEC {0,1} (0<->1), MEC {3} self-loop; 2 transient; 4 transient.
    // accepting = {1} -> only {0,1} is a good MEC.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {0,1,1} }},
        /*2*/ {{ {0,1,1} }},
        /*3*/ {{ {3,1,1} }},
        /*4*/ {{ {3,1,1} }},
    };
    auto good = acceptingMECStates(m, {1});
    std::sort(good.begin(), good.end());
    CHECK(good == std::vector<int>{0, 1});
    auto good2 = acceptingMECStates(m, {3});
    CHECK(good2 == std::vector<int>{3});
    auto good3 = acceptingMECStates(m, {2});   // 2 is transient -> no good MEC
    CHECK(good3.empty());
}

TEST_CASE("buchi: pessimistic == optimistic on point MDPs (no nature choice)") {
    IMDPModel m = {
        /*0*/ {{ {1,0.5,0.5}, {2,0.5,0.5} }},
        /*1*/ {{ {1,1,1} }},
        /*2*/ {{ {2,1,1} }},
    };
    auto bo = maxBuchiOptimistic(m, {1}, EPS);
    auto bp = maxBuchiPessimistic(m, {1}, EPS);
    for (int s = 0; s < 3; ++s)
        CHECK(0.5*(bo.lower[s]+bo.upper[s]) == doctest::Approx(0.5*(bp.lower[s]+bp.upper[s])).epsilon(1e-4));
}
