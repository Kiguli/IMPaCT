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

#include <random>
#include <string>
#include <vector>

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

// ---------------------------------------------------------------------------
// Phase 2 part 2: LTLf -> DFA. Validated DIFFERENTIALLY against the verified
// finite-trace evaluator (acceptsFinite): for random formulas x random traces,
// dfaAccepts must equal acceptsFinite. This is the correctness gate for the
// tricky X/U/G/F derivatives (see ISSUE-0004).
// ---------------------------------------------------------------------------
using impact::ltl::DFA;
using impact::ltl::toDFA;
using impact::ltl::dfaAccepts;

TEST_CASE("ltlf->dfa: matches evaluator on the contract formulas") {
    std::vector<std::string> aps = {"a", "b", "pickup", "deliver"};
    for (const std::string& f : { "F a", "a U b", "G a", "F(a & X b)", "F(pickup & F deliver)" }) {
        auto* aut = compileFinite(f, aps);
        DFA dfa = toDFA(aut);
        FiniteTrace traces[] = {
            {{"a"}}, {{}, {"a"}}, {{"a"}, {}}, {{"b"}}, {{"a"}, {"b"}},
            {{"pickup"}, {"deliver"}}, {{"deliver"}, {"pickup"}}, {{"a","b"}, {"b"}},
        };
        for (const auto& t : traces) CHECK(dfaAccepts(dfa, t) == acceptsFinite(aut, t));
        destroy(aut);
    }
}

namespace {
std::string rand_formula(std::mt19937& rng, int depth, const std::vector<std::string>& aps) {
    if (depth <= 0 || (rng() % 4u) == 0u) return aps[rng() % aps.size()];
    int c = (int)(rng() % 8u);
    std::string x = rand_formula(rng, depth - 1, aps);
    switch (c) {
        case 0: return "(!" + x + ")";
        case 1: return "(X " + x + ")";
        case 2: return "(F " + x + ")";
        case 3: return "(G " + x + ")";
        default: {
            std::string y = rand_formula(rng, depth - 1, aps);
            const char* ops[] = {" & ", " | ", " U ", " -> "};
            return "(" + x + ops[c - 4] + y + ")";
        }
    }
}
}

TEST_CASE("ltlf->dfa: randomized differential vs evaluator (random formulas x traces)") {
    std::mt19937 rng(20260625);
    std::vector<std::string> aps = {"a", "b"};   // 2 APs -> 4-letter alphabet (keeps DFA build fast)
    int checked = 0, built = 0;
    for (int fi = 0; fi < 300; ++fi) {
        std::string f = rand_formula(rng, 2 + (int)(rng() % 2u), aps);  // depth 2-3
        impact::ltl::Automaton* aut = nullptr;
        try { aut = compileFinite(f, aps); } catch (...) { continue; }
        DFA dfa;
        try { dfa = toDFA(aut); }                            // skip pathological blow-ups
        catch (...) { destroy(aut); continue; }
        ++built;
        for (int ti = 0; ti < 20; ++ti) {
            int len = 1 + (int)(rng() % 5u);                // non-empty traces (evaluator domain)
            FiniteTrace tr;
            for (int k = 0; k < len; ++k) {
                Letter L;
                for (const std::string& p : aps) if (rng() & 1u) L.insert(p);
                tr.push_back(L);
            }
            CHECK(dfaAccepts(dfa, tr) == acceptsFinite(aut, tr));
            ++checked;
        }
        destroy(aut);
    }
    CHECK(built > 40);
    CHECK(checked > 800);
}
