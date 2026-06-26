// ============================================================================
// CONTRACT TESTS — Phase 2: LTLf / co-safe-LTL front-end.
// Tested by LANGUAGE MEMBERSHIP so the contract is independent of the back-end
// (Spot / Owl / Lydia). Semantics: LTLf (finite-trace), per De Giacomo-Vardi.
// A trace is a list of letters; a letter is the set of APs true at that step.
//
// IMMUTABLE: accept/reject expectations are derived directly from LTLf semantics
// (see tests/TEST_PLAN.md §2). These stay red until Phase 2 implements the
// front-end (currently impact::ltl::* throws "not implemented").
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

using impact::ltl::Letter;
using impact::ltl::FiniteTrace;
using impact::ltl::compileFinite;
using impact::ltl::acceptsFinite;
using impact::ltl::destroy;

// Small helper to assert membership for a compiled formula.
struct Compiled {
    impact::ltl::Automaton* a;
    Compiled(const std::string& f, const std::vector<std::string>& aps) { a = compileFinite(f, aps); }
    ~Compiled() { destroy(a); }
    bool accepts(const FiniteTrace& t) const { return acceptsFinite(a, t); }
};

TEST_CASE("ltlf: F a (eventually a)") {
    Compiled c("F a", {"a"});
    CHECK(c.accepts({ {"a"} }));
    CHECK(c.accepts({ {}, {"a"} }));
    CHECK(c.accepts({ {"a"}, {} }));
    CHECK_FALSE(c.accepts({ {} }));
    CHECK_FALSE(c.accepts({ {}, {} }));
}

TEST_CASE("ltlf: a U b (a until b)") {
    Compiled c("a U b", {"a", "b"});
    CHECK(c.accepts({ {"b"} }));               // b immediately
    CHECK(c.accepts({ {"a"}, {"b"} }));
    CHECK(c.accepts({ {"a"}, {"a"}, {"b"} }));
    CHECK(c.accepts({ {"a","b"} }));           // b holds at step 0
    CHECK_FALSE(c.accepts({ {"a"}, {"a"} }));  // b never holds
    CHECK_FALSE(c.accepts({ {}, {"b"} }));     // a false before first b
}

TEST_CASE("ltlf: G a (a holds at every step, finite-trace)") {
    Compiled c("G a", {"a"});
    CHECK(c.accepts({ {"a"} }));
    CHECK(c.accepts({ {"a"}, {"a"}, {"a"} }));
    CHECK_FALSE(c.accepts({ {"a"}, {} }));
    CHECK_FALSE(c.accepts({ {}, {"a"} }));
}

TEST_CASE("ltlf: F(a & X b) — a now, b next, sometime") {
    Compiled c("F(a & X b)", {"a", "b"});
    CHECK(c.accepts({ {"a"}, {"b"} }));
    CHECK(c.accepts({ {}, {"a"}, {"b"} }));
    CHECK(c.accepts({ {"a","b"}, {"b"} }));     // a@0 and b@1
    CHECK_FALSE(c.accepts({ {"a"} }));          // no next step for b
    CHECK_FALSE(c.accepts({ {"a","b"} }));      // X b has no successor
}

TEST_CASE("ltlf: Package-Delivery ordering — F(pickup & F deliver)") {
    Compiled c("F(pickup & F deliver)", {"pickup", "deliver"});
    CHECK(c.accepts({ {"pickup"}, {"deliver"} }));
    CHECK(c.accepts({ {}, {"pickup"}, {}, {"deliver"} }));
    CHECK_FALSE(c.accepts({ {"deliver"}, {"pickup"} })); // delivered before pickup
    CHECK_FALSE(c.accepts({ {"pickup"} }));              // never delivered
}
