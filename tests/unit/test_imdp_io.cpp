// ============================================================================
// CONTRACT TESTS — explicit (interval) MDP exchange format (cross-tool benchmarking).
// Parse -> solve a known value; round-trip parse(write(.)) preserves the solution.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <string>

using impact::io::parse;
using impact::io::write;
using impact::io::Problem;

static const char* MODEL =
    "# interval-MDP: nature minimizes mass to state 1 (which reaches target)\n"
    "states 4\n"
    "init 0\n"
    "label target 3\n"
    "tran 0 0  1:0.4:0.6 2:0.4:0.6\n"
    "tran 1 0  3:1:1\n"
    "tran 2 0  2:1:1\n"
    "tran 3 0  3:1:1\n";

TEST_CASE("imdp_io: parse structure") {
    Problem p = parse(MODEL);
    CHECK(p.nStates == 4);
    CHECK(p.init == 0);
    CHECK(p.labels.at("target") == std::set<int>{3});
    CHECK(p.model[0].size() == 1);                 // one action
    CHECK(p.model[0][0].size() == 2);              // two successors
    CHECK(p.model[0][0][0].to == 1);
    CHECK(p.model[0][0][0].lo == doctest::Approx(0.4));
    CHECK(p.model[0][0][0].hi == doctest::Approx(0.6));
}

TEST_CASE("imdp_io: parse -> solve a known robust value") {
    Problem p = parse(MODEL);
    auto pess = impact::solve::maxReachPessimistic(p.model, p.labels.at("target"), 1e-7);
    auto opt  = impact::solve::maxReachOptimistic (p.model, p.labels.at("target"), 1e-7);
    CHECK(0.5*(pess.lower[p.init]+pess.upper[p.init]) == doctest::Approx(0.4).epsilon(1e-3));
    CHECK(0.5*(opt.lower[p.init]+opt.upper[p.init])  == doctest::Approx(0.6).epsilon(1e-3));
}

TEST_CASE("imdp_io: write round-trips (same solution)") {
    Problem p = parse(MODEL);
    Problem q = parse(write(p));
    REQUIRE(q.nStates == p.nStates);
    CHECK(q.labels.at("target") == p.labels.at("target"));
    auto a = impact::solve::maxReachPessimistic(p.model, p.labels.at("target"), 1e-7);
    auto b = impact::solve::maxReachPessimistic(q.model, q.labels.at("target"), 1e-7);
    for (int s = 0; s < p.nStates; ++s)
        CHECK(0.5*(a.lower[s]+a.upper[s]) == doctest::Approx(0.5*(b.lower[s]+b.upper[s])).epsilon(1e-6));
}
