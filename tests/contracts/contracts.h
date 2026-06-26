#ifndef IMPACT_TEST_CONTRACTS_H
#define IMPACT_TEST_CONTRACTS_H

// ============================================================================
// IMPaCT v2.0 — TEST CONTRACTS (interfaces the implementation must satisfy)
// ----------------------------------------------------------------------------
// These declarations are the *target API* for the new IMPaCT v2.0 components.
// The TDD test suite is written against THESE signatures. Implementation work
// fills in the corresponding src/*.cpp (replacing tests/contracts/stubs.cpp).
//
// RULE: tests are immutable contracts. We change the implementation to satisfy
// the tests, never the other way around. If an interface here must change, it
// is a deliberate design change reviewed against the test plan — not a hack to
// make a failing test pass.
// ============================================================================

#include <vector>
#include <set>
#include <string>
#include <cstddef>

// Phase 1a — O-maximization. Now implemented (verified extraction of IMPaCT's
// sorted approach); the production interface lives in src/. The behavioural
// contract (feasibility, optimality, throwing) is documented there and in
// tests/unit/test_omaximization.cpp.
#include "../../src/omaximization.h"

// Phase 1b — SCC + MEC decomposition. Implemented; production interface in src/.
#include "../../src/graph_utils.h"

// Phase 1c — sound robust interval iteration. Implemented; interface in src/.
#include "../../src/solve.h"

namespace impact {

// ---------------------------------------------------------------------------
// Phase 2 — LTL/LTLf -> automaton front-end. Tested via language membership so
// the contract is independent of the chosen back-end library (Spot/Owl/Lydia).
// Refs: Kupferman-Vardi (FMSD 2001); De Giacomo-Vardi (IJCAI 2013/2015).
// ---------------------------------------------------------------------------
namespace ltl {

    // A finite trace: each element is the set of atomic propositions true at that
    // step (subset of the formula's APs), given by name.
    using Letter = std::set<std::string>;
    using FiniteTrace = std::vector<Letter>;

    // Opaque handle to a compiled (deterministic, finite) automaton for an
    // LTLf / co-safe-LTL formula over the given alphabet of AP names.
    struct Automaton;  // defined by the implementation

    // Compile an LTLf / co-safe LTL formula to a DFA. THROWS std::invalid_argument
    // on parse error or if the formula is not co-safe (when require_cosafe).
    Automaton* compileFinite(const std::string& formula,
                             const std::vector<std::string>& aps);

    // Does the compiled automaton accept this finite trace? (language membership)
    bool acceptsFinite(const Automaton* a, const FiniteTrace& trace);

    void destroy(Automaton* a);

} // namespace ltl

} // namespace impact

#endif // IMPACT_TEST_CONTRACTS_H
