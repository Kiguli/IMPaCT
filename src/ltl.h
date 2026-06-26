#ifndef IMPACT_LTL_H
#define IMPACT_LTL_H

// ============================================================================
// LTLf / co-safe-LTL front-end (Phase 2).
//
// This first increment provides PARSING + finite-trace MEMBERSHIP (LTLf
// semantics, De Giacomo-Vardi IJCAI 2013/2015). The public contract is
// membership-based so it is independent of any back-end. NOTE: the DFA
// construction needed for the IMDP×automaton PRODUCT (Phase 2 reachability) is a
// separate step tracked in issues/ — this evaluator establishes and validates
// the semantics first (and can serve as the differential oracle for the DFA).
//
// Contracts: tests/unit/test_ltl_dfa.cpp.
// ============================================================================

#include <vector>
#include <set>
#include <string>

namespace impact {
namespace ltl {

    // A finite trace: each element is the set of atomic propositions true at that
    // step (subset of the formula's APs), given by name.
    using Letter = std::set<std::string>;
    using FiniteTrace = std::vector<Letter>;

    // Opaque compiled formula (AST + alphabet); defined in ltl.cpp.
    struct Automaton;

    // Parse an LTLf / co-safe-LTL formula over the given AP names.
    // Supported operators: ! (not), & (and), | (or), -> (implies), <-> (iff),
    // X (strong next), F (eventually), G (globally), U (until), R (release),
    // plus true / false and parentheses.
    // THROWS std::invalid_argument on a parse error or an unknown atom.
    Automaton* compileFinite(const std::string& formula,
                             const std::vector<std::string>& aps);

    // Finite-trace membership (LTLf semantics): does the formula hold at position 0?
    bool acceptsFinite(const Automaton* a, const FiniteTrace& trace);

    void destroy(Automaton* a);

} // namespace ltl
} // namespace impact

#endif // IMPACT_LTL_H
