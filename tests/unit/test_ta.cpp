// ============================================================================
// CONTRACT TESTS — timed-automaton zone-graph reachability.
// Timing-feasible vs infeasible reachability (guards + invariants), multi-clock
// coupling, and termination (extrapolation) on a resetting self-loop.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>

using namespace impact::ta;

TEST_CASE("ta: reachable when the guard is timing-feasible") {
    // l0 --(x>=2, reset x)--> l1 ; no invariants. Delay to x=2, take edge -> l1.
    TA ta;
    ta.nLoc = 2; ta.nClocks = 1; ta.init = 0;
    ta.invariant = { {}, {} };
    ta.edges = { { 0, { clkGe(1, 2) }, { 1 }, 1 } };
    ta.kmax = { 0, 2 };
    CHECK(reachable(ta, 1));
    CHECK(reachable(ta, 0));      // init trivially reachable
}

TEST_CASE("ta: unreachable when an invariant blocks the guard") {
    // same edge, but invariant x<=1 at l0 forbids delaying to x>=2.
    TA ta;
    ta.nLoc = 2; ta.nClocks = 1; ta.init = 0;
    ta.invariant = { { clkLe(1, 1) }, {} };
    ta.edges = { { 0, { clkGe(1, 2) }, { 1 }, 1 } };
    ta.kmax = { 0, 2 };
    bool cap = false;
    CHECK(!reachable(ta, 1, 200000, &cap));
    CHECK(!cap);                 // decided, not capped
}

TEST_CASE("ta: sequential timing with resets") {
    // l0 --(x>=1,reset)--> l1 --(x>=1,reset)--> l2. l2 reachable (1 then 1 more).
    TA ta;
    ta.nLoc = 3; ta.nClocks = 1; ta.init = 0;
    ta.invariant = { {}, {}, {} };
    ta.edges = { { 0, { clkGe(1, 1) }, { 1 }, 1 },
                 { 1, { clkGe(1, 1) }, { 1 }, 2 } };
    ta.kmax = { 0, 1 };
    CHECK(reachable(ta, 2));
}

TEST_CASE("ta: two clocks couple under delay (x and y advance together)") {
    // l0 --(x>=2 & y<=3, reset x)--> l1. From 0,0 delay to (2,2): y=2<=3 -> reachable.
    TA okTA;
    okTA.nLoc = 2; okTA.nClocks = 2; okTA.init = 0;
    okTA.invariant = { {}, {} };
    okTA.edges = { { 0, { clkGe(1, 2), clkLe(2, 3) }, { 1 }, 1 } };
    okTA.kmax = { 0, 2, 3 };
    CHECK(reachable(okTA, 1));

    // invariant y<=1 at l0: x and y rise together, so x>=2 forces y>=2>1 -> blocked.
    TA blocked = okTA;
    blocked.invariant = { { clkLe(2, 1) }, {} };
    CHECK(!reachable(blocked, 1));
}

TEST_CASE("ta: terminates on a resetting self-loop (extrapolation)") {
    // l0 self-loops forever (x>=1, reset x); target l1 is disconnected.
    TA ta;
    ta.nLoc = 2; ta.nClocks = 1; ta.init = 0;
    ta.invariant = { {}, {} };
    ta.edges = { { 0, { clkGe(1, 1) }, { 1 }, 0 } };
    ta.kmax = { 0, 1 };
    bool cap = false;
    CHECK(!reachable(ta, 1, 100000, &cap));
    CHECK(!cap);                 // finite zone graph -> terminates without hitting the cap
}
