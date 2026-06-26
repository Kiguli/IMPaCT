// ============================================================================
// CONTRACT TESTS — Phase 1c: sound robust interval iteration (reachability).
// IMMUTABLE hand-derived expectations. Oracle values are exact and re-derivable;
// the numpy cross-check lives in tests/oracles/oracles.py.
//
// Models 4 and 5 are the load-bearing soundness tests: they FAIL unless the
// implementation does Prob0/Prob1 + end-component handling. Naive value
// iteration from [0,1] gets stuck above the true value on Model 4's end
// component {0,1} — the whole point of interval iteration (Haddad-Monmege 2018).
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <set>

using namespace impact::solve;

static const double EPS = 1e-7;

static void check_sound(const IntervalResult& r, int s, double truth) {
    CHECK(r.lower[s] <= truth + 1e-9);                 // lower is a true lower bound
    CHECK(r.upper[s] >= truth - 1e-9);                 // upper is a true upper bound
    CHECK(r.upper[s] - r.lower[s] <= 2 * EPS + 1e-9);  // gap closed to <= 2*eps
    CHECK(0.5 * (r.lower[s] + r.upper[s]) == doctest::Approx(truth).epsilon(1e-4));
}

TEST_CASE("ii Model 1: point-probability one-step branch (reach 0.5)") {
    // 0: ->1 @0.5, ->2(sink) @0.5 ; 1: ->3(target)@1 ; 2 sink ; 3 target
    IMDPModel m = {
        /*0*/ {{ {1,0.5,0.5}, {2,0.5,0.5} }},
        /*1*/ {{ {3,1,1} }},
        /*2*/ {{ {2,1,1} }},
        /*3*/ {{ {3,1,1} }},
    };
    auto r = maxReachPessimistic(m, {3}, EPS);
    check_sound(r, 0, 0.5);
    check_sound(r, 1, 1.0);
    check_sound(r, 2, 0.0);
    check_sound(r, 3, 1.0);
}

TEST_CASE("ii Model 2: interval branch — pessimistic 0.4, optimistic 0.6") {
    IMDPModel m = {
        /*0*/ {{ {1,0.4,0.6}, {2,0.4,0.6} }},
        /*1*/ {{ {3,1,1} }},
        /*2*/ {{ {2,1,1} }},
        /*3*/ {{ {3,1,1} }},
    };
    check_sound(maxReachPessimistic(m, {3}, EPS), 0, 0.4); // nature minimizes mass to 1
    check_sound(maxReachOptimistic (m, {3}, EPS), 0, 0.6); // nature maximizes mass to 1
}

TEST_CASE("ii Model 3: self-loop that still reaches target w.p. 1") {
    // 0: ->0 @0.5, ->1(target) @0.5  => P(reach)=1
    IMDPModel m = {
        /*0*/ {{ {0,0.5,0.5}, {1,0.5,0.5} }},
        /*1*/ {{ {1,1,1} }},
    };
    check_sound(maxReachPessimistic(m, {1}, EPS), 0, 1.0);
}

TEST_CASE("ii Model 4: end component {0,1} with NO path to target => value 0") {
    // 0: a0->1 ; 1: a0->0  (controller can loop forever) ; target {2} unreachable.
    // Naive VI from [0,1] is STUCK at upper=1 here; correct preprocessing => 0.
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }},
        /*1*/ {{ {0,1,1} }},
        /*2*/ {{ {2,1,1} }},
    };
    auto r = maxReachPessimistic(m, {2}, EPS);
    check_sound(r, 0, 0.0);
    check_sound(r, 1, 0.0);
}

TEST_CASE("ii Model 5: end component WITH an exit action to target => value 1") {
    // 0: a0->1 (stay in EC) | a1->2 (exit to target) ; 1: a0->0 ; target {2}
    IMDPModel m = {
        /*0*/ {{ {1,1,1} }, { {2,1,1} }},
        /*1*/ {{ {0,1,1} }},
        /*2*/ {{ {2,1,1} }},
    };
    auto r = maxReachPessimistic(m, {2}, EPS);
    check_sound(r, 0, 1.0);  // controller chooses the exit
    check_sound(r, 1, 1.0);  // 1 -> 0 -> exit
}
