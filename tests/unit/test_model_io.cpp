// ============================================================================
// CONTRACT TESTS — text parsers for the visualizer model formats (.pta, .pomdp).
// Parse, then check the parsed model solves to the known value via the verified
// engines (ties the parsers to the oracle-tested solvers).
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

// ---- .pta parser -> PTA zone reachability ----------------------------------
TEST_CASE("pta_io: parse a probabilistic timed automaton and solve Pmax") {
    const char* M =
        "clocks 1\ninit 0\nkmax 1 3\ntarget 1\n"
        "edge 0 | 1>=2 | 0.7 1 ; 0.3 2\n"
        "edge 2 | 1>=0 | 1.0 2\n";
    auto parsed = impact::pta_io::parse(M);
    CHECK(parsed.pta.nClocks == 1);
    CHECK(parsed.pta.nLoc == 3);
    CHECK(parsed.target == 1);
    CHECK(parsed.pta.edges.size() == 2);
    CHECK(impact::pta::maxReachLocation(parsed.pta, parsed.target) == doctest::Approx(0.7).epsilon(1e-4));
    // digital engine agrees (independent cross-check)
    CHECK(impact::pta::maxReachLocationDigital(parsed.pta, parsed.target) == doctest::Approx(0.7).epsilon(1e-4));
}

TEST_CASE("pta_io: invariant + reset parse correctly") {
    const char* M =
        "clocks 1\ninit 0\nkmax 1 2\ntarget 1\n"
        "inv 0 1<=1\n"
        "edge 0 | 1>=2 | 1.0 1\n";
    auto parsed = impact::pta_io::parse(M);
    CHECK(parsed.pta.invariant[0].size() == 1);              // x<=1 invariant
    CHECK(impact::pta::maxReachLocation(parsed.pta, 1) == doctest::Approx(0.0));   // blocked
}

// ---- .pomdp parser -> finite-horizon belief reachability -------------------
TEST_CASE("pomdp_io: parse a POMDP and solve finite-horizon reach") {
    const char* M =
        "states 2\nactions 1\nobs 1\ninit 0:1\ntarget 1\nhorizon 3\n"
        "T 0 0 : 0:0.5 1:0.5\n"
        "T 0 1 : 1:1\n";
    auto parsed = impact::pomdp_io::parse(M);
    CHECK(parsed.pomdp.nStates == 2);
    CHECK(parsed.horizon == 3);
    CHECK(parsed.target.count(1));
    CHECK(impact::pomdp::maxReachFiniteHorizon(parsed.pomdp, parsed.target, parsed.horizon)
          == doctest::Approx(0.875));   // 1 - 0.5^3
}

TEST_CASE("pomdp_io: missing dims throw") {
    CHECK_THROWS(impact::pomdp_io::parse("actions 1\nobs 1\n"));   // no states
}
