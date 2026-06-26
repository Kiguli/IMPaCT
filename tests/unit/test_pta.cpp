// ============================================================================
// CONTRACT TESTS — probabilistic timed automata -> symbolic MDP -> maxReach.
// Verifiable Pmax(reach location) hand cases: timing-gated probabilistic edges,
// controller choice, sequential resets, and invariant-blocked unreachability.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>

using namespace impact::pta;
using impact::ta::clkGe;
using impact::ta::clkLe;

TEST_CASE("pta: single probabilistic edge -> Pmax = branch probability") {
    // l0 --(x>=2)--> { 0.7: l1(target), 0.3: l2(sink) }
    PTA p;
    p.nLoc = 3; p.nClocks = 1; p.init = 0;
    p.invariant = { {}, {}, {} };
    p.edges = { { 0, { clkGe(1, 2) }, { {0.7, {}, 1}, {0.3, {}, 2} } } };
    p.kmax = { 0, 2 };
    CHECK(maxReachLocation(p, 1) == doctest::Approx(0.7).epsilon(1e-4));
    CHECK(maxReachLocation(p, 2) == doctest::Approx(0.3).epsilon(1e-4));

    auto smdp = build(p, 1);
    CHECK(smdp.nSym >= 3);
    CHECK(!smdp.targets.empty());
}

TEST_CASE("pta: invariant blocks the guard -> location unreachable (Pmax 0)") {
    PTA p;
    p.nLoc = 3; p.nClocks = 1; p.init = 0;
    p.invariant = { { clkLe(1, 1) }, {}, {} };       // x<=1 forbids delaying to x>=2
    p.edges = { { 0, { clkGe(1, 2) }, { {0.7, {}, 1}, {0.3, {}, 2} } } };
    p.kmax = { 0, 2 };
    CHECK(maxReachLocation(p, 1) == doctest::Approx(0.0));
}

TEST_CASE("pta: controller picks the better edge") {
    // two edges from l0; controller maximizes -> 0.9.
    PTA p;
    p.nLoc = 3; p.nClocks = 1; p.init = 0;
    p.invariant = { {}, {}, {} };
    p.edges = {
        { 0, { clkGe(1, 1) }, { {0.5, {}, 1}, {0.5, {}, 2} } },
        { 0, { clkGe(1, 1) }, { {0.9, {}, 1}, {0.1, {}, 2} } },
    };
    p.kmax = { 0, 1 };
    CHECK(maxReachLocation(p, 1) == doctest::Approx(0.9).epsilon(1e-4));
}

TEST_CASE("pta: sequential edges with reset compose multiplicatively") {
    // l0 --(x>=1, reset x)--> l1 ; l1 --(x>=1)--> {0.5: l2(target), 0.5: l3}
    PTA p;
    p.nLoc = 4; p.nClocks = 1; p.init = 0;
    p.invariant = { {}, {}, {}, {} };
    p.edges = {
        { 0, { clkGe(1, 1) }, { {1.0, {1}, 1} } },
        { 1, { clkGe(1, 1) }, { {0.5, {}, 2}, {0.5, {}, 3} } },
    };
    p.kmax = { 0, 1 };
    CHECK(maxReachLocation(p, 2) == doctest::Approx(0.5).epsilon(1e-4));
}

TEST_CASE("pta digital: Pmax agrees with the independent zone engine") {
    // closed (non-strict) PTAs only. Build a few and cross-check the two engines.
    std::vector<PTA> models;
    { PTA p; p.nLoc=3; p.nClocks=1; p.init=0; p.invariant={{},{},{}};
      p.edges={ {0,{clkGe(1,2)},{{0.7,{},1},{0.3,{},2}}} }; p.kmax={0,2}; models.push_back(p); }
    { PTA p; p.nLoc=3; p.nClocks=1; p.init=0; p.invariant={{clkLe(1,1)},{},{}};
      p.edges={ {0,{clkGe(1,2)},{{0.7,{},1},{0.3,{},2}}} }; p.kmax={0,2}; models.push_back(p); }
    { PTA p; p.nLoc=3; p.nClocks=1; p.init=0; p.invariant={{},{},{}};
      p.edges={ {0,{clkGe(1,1)},{{0.5,{},1},{0.5,{},2}}}, {0,{clkGe(1,1)},{{0.9,{},1},{0.1,{},2}}} };
      p.kmax={0,1}; models.push_back(p); }
    { PTA p; p.nLoc=4; p.nClocks=1; p.init=0; p.invariant={{},{},{},{}};
      p.edges={ {0,{clkGe(1,1)},{{1.0,{1},1}}}, {1,{clkGe(1,1)},{{0.5,{},2},{0.5,{},3}}} };
      p.kmax={0,1}; models.push_back(p); }
    for (const PTA& p : models)
        for (int tgt = 0; tgt < p.nLoc; ++tgt)
            CHECK(maxReachLocationDigital(p, tgt) == doctest::Approx(maxReachLocation(p, tgt)).epsilon(1e-4));
}

TEST_CASE("pta digital: Pmin -- invariant forces the edge vs waiting it out") {
    // edge to target is the only edge. With an invariant x<=2 the controller MUST
    // take it by x=2 -> Pmin=1. Without an invariant it can wait forever -> Pmin=0.
    PTA forced;
    forced.nLoc = 2; forced.nClocks = 1; forced.init = 0;
    forced.invariant = { { clkLe(1, 2) }, {} };
    forced.edges = { { 0, { clkGe(1, 0) }, { {1.0, {}, 1} } } };
    forced.kmax = { 0, 2 };
    CHECK(minReachLocationDigital(forced, 1) == doctest::Approx(1.0));

    PTA waitOut = forced;
    waitOut.invariant = { {}, {} };          // no deadline -> can avoid forever
    CHECK(minReachLocationDigital(waitOut, 1) == doctest::Approx(0.0));
}

TEST_CASE("pta digital: Pmin picks the lower-probability forced edge") {
    // invariant x<=1 forces an edge by x=1; controller minimizes -> picks e2 (0.3).
    PTA p;
    p.nLoc = 3; p.nClocks = 1; p.init = 0;
    p.invariant = { { clkLe(1, 1) }, {}, {} };
    p.edges = {
        { 0, { clkGe(1, 0) }, { {0.5, {}, 1}, {0.5, {}, 2} } },
        { 0, { clkGe(1, 0) }, { {0.3, {}, 1}, {0.7, {}, 2} } },
    };
    p.kmax = { 0, 1 };
    CHECK(minReachLocationDigital(p, 1) == doctest::Approx(0.3).epsilon(1e-4));
    CHECK(maxReachLocationDigital(p, 1) == doctest::Approx(0.5).epsilon(1e-4));   // Pmax picks e1
}

TEST_CASE("pta: target == init -> 1 ; no edges -> deadlock") {
    PTA p;
    p.nLoc = 2; p.nClocks = 1; p.init = 0;
    p.invariant = { {}, {} };
    p.edges = {};                              // deadlocked at l0
    p.kmax = { 0, 1 };
    CHECK(maxReachLocation(p, 0) == doctest::Approx(1.0));
    CHECK(maxReachLocation(p, 1) == doctest::Approx(0.0));
}
