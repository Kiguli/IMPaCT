// ============================================================================
// CONTRACT TESTS — LTL fragment dispatcher. Each route must match the underlying
// verified solver (reach/safety/until/next/buchi/persistence/patrol); arbitrary LTL
// outside the fragment must be reported as out-of-fragment (-> LDBA, ISSUE-0016).
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <set>

using namespace impact::solve;
namespace ls = impact::ltlspec;
namespace om = impact::omega;
namespace pc = impact::pctl;

static const double EPS = 1e-7;
static double mid(const IntervalResult& r, int s) { return 0.5 * (r.lower[s] + r.upper[s]); }
static void same(const IntervalResult& a, const IntervalResult& b, int n) {
    for (int s = 0; s < n; ++s) CHECK(mid(a, s) == doctest::Approx(mid(b, s)).epsilon(1e-6));
}

TEST_CASE("ltlspec F phi == reach ; G phi == safety") {
    IMDPModel m = {
        /*0*/ {{ {1,0.4,0.6}, {2,0.4,0.6} }},
        /*1*/ {{ {3,1,1} }}, /*2*/ {{ {2,1,1} }}, /*3*/ {{ {3,1,1} }},
    };
    ls::Labels L = { {"goal", {3}}, {"bad", {2}} };
    same(ls::synthesize(m, L, "F goal", true, EPS), maxReachPessimistic(m, {3}, EPS), 4);
    // G (!bad) == stay out of state 2 == safety(avoid={2})
    same(ls::synthesize(m, L, "G (!bad)", true, EPS), maxSafetyPessimistic(m, {2}, EPS), 4);
    CHECK(mid(ls::synthesize(m, L, "F goal", true, EPS), 0) == doctest::Approx(0.4).epsilon(1e-3));
}

TEST_CASE("ltlspec a U b == until ; X a == next") {
    IMDPModel m = { /*0*/ {{ {1,1,1} }}, /*1*/ {{ {2,1,1} }}, /*2*/ {{ {2,1,1} }} };
    ls::Labels L = { {"a", {0,1}}, {"b", {2}} };
    same(ls::synthesize(m, L, "a U b", true, EPS), pc::untilPessimistic(m, {0,1}, {2}, EPS), 3);
    same(ls::synthesize(m, L, "X b", true, EPS), pc::nextPessimistic(m, {2}, EPS), 3);
}

TEST_CASE("ltlspec G F r == Buchi ; F G p == persistence") {
    IMDPModel m = { /*0*/ {{ {1,1,1} }}, /*1*/ {{ {0,1,1} }} };  // 0<->1 cycle
    ls::Labels L = { {"r", {1}}, {"p", {0,1}} };
    same(ls::synthesize(m, L, "G F r", true, EPS), om::maxBuchiPessimistic(m, {1}, EPS), 2);
    same(ls::synthesize(m, L, "F G p", true, EPS), om::maxPersistencePessimistic(m, {0,1}, EPS), 2);
    CHECK(mid(ls::synthesize(m, L, "G F r", true, EPS), 0) == doctest::Approx(1.0));
}

TEST_CASE("ltlspec patrol: (G F r0) & (G F r2) == generalized Buchi") {
    IMDPModel m = { /*0*/ {{ {1,1,1} }}, /*1*/ {{ {2,1,1} }}, /*2*/ {{ {0,1,1} }} };  // 0->1->2->0
    ls::Labels L = { {"r0", {0}}, {"r2", {2}} };
    auto got = ls::synthesize(m, L, "(G F r0) & (G F r2)", true, EPS);
    auto ref = om::maxGenBuchiPessimistic(m, { {0}, {2} }, EPS);
    same(got, ref, 3);
    CHECK(mid(got, 0) == doctest::Approx(1.0));
}

TEST_CASE("ltlspec optimistic flavour routes too") {
    IMDPModel m = { /*0*/ {{ {1,0.4,0.6}, {2,0.4,0.6} }}, /*1*/ {{ {3,1,1} }}, /*2*/ {{ {2,1,1} }}, /*3*/ {{ {3,1,1} }} };
    ls::Labels L = { {"goal", {3}} };
    same(ls::synthesize(m, L, "F goal", false, EPS), maxReachOptimistic(m, {3}, EPS), 4);
}

TEST_CASE("ltlspec atom -> 0/1 indicator") {
    IMDPModel m = { {{ {0,1,1} }}, {{ {1,1,1} }} };
    ls::Labels L = { {"a", {1}} };
    auto r = ls::synthesize(m, L, "a", true, EPS);
    CHECK(mid(r, 0) == doctest::Approx(0.0));
    CHECK(mid(r, 1) == doctest::Approx(1.0));
}

TEST_CASE("ltlspec tokenizer tolerates commas and &&/|| (no crash)") {
    IMDPModel m = { /*0*/ {{ {1,1,1} }}, /*1*/ {{ {0,1,1} }} };
    ls::Labels L = { {"r0", {0}}, {"r2", {1}} };
    // commas as separators: equivalent to a space; '&&' accepted as '&'
    same(ls::synthesize(m, L, "(G F r0) && (G F r2)", true, EPS),
         om::maxGenBuchiPessimistic(m, { {0}, {1} }, EPS), 2);
    // a stray comma must NOT throw "bad character" (clear parse outcome instead)
    CHECK_NOTHROW(ls::synthesize(m, L, "F r0", true, EPS));
}

TEST_CASE("ltlspec out-of-fragment formulas throw (-> LDBA / ISSUE-0016)") {
    IMDPModel m = { {{ {1,1,1} }}, {{ {1,1,1} }} };
    ls::Labels L = { {"a", {0}}, {"b", {1}} };
    CHECK_THROWS(ls::synthesize(m, L, "F (a U b)", true, EPS));    // nested temporal under F
    CHECK_THROWS(ls::synthesize(m, L, "(F a) U b", true, EPS));    // temporal operand of U
    CHECK_THROWS(ls::synthesize(m, L, "G F a | G F b", true, EPS)); // disjunction of recurrences
    CHECK_THROWS(ls::synthesize(m, L, "F nope", true, EPS));        // unknown atom
}
