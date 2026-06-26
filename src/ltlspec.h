#ifndef IMPACT_LTLSPEC_H
#define IMPACT_LTLSPEC_H

// ============================================================================
// LTL specification front-end (dispatcher) — one entry point that parses an LTL
// formula and routes the SUPPORTED FRAGMENT to the already-verified solvers, giving
// the robust (controller-Pmax) satisfaction probability per state. No external
// dependency. Arbitrary LTL outside this fragment needs an LDBA translation
// (Spot/Owl) and is reported as out-of-fragment (ISSUE-0016).
//
// Supported fragment (operands of temporal operators are STATE formulas = boolean
// combinations of atoms):
//   boolean atoms / ! & | -> over atoms        -> state set (indicator)
//   X phi                                       -> pctl::next
//   F phi                                       -> solve::maxReach
//   G phi                                       -> solve::maxSafety (stay in phi)
//   phi U psi                                   -> pctl::until
//   G F phi      (recurrence / Büchi)           -> omega::maxBuchi
//   F G phi      (persistence / co-Büchi)       -> omega::maxPersistence
//   (G F a1) & (G F a2) & ...  (patrol)         -> omega::maxGenBuchi
//
// Operators: ! (not) & (and) | (or) -> (implies), X F G U (reserved); atoms are any
// other identifiers; true/false; parentheses.
//
// Refs: Pnueli (FOCS 1977, LTL); the ω-regular back-ends carry their own citations.
// Contracts: tests/unit/test_ltlspec.cpp. Out-of-fragment -> ISSUE-0016 (LDBA).
// ============================================================================

#include <map>
#include <set>
#include <string>
#include "solve.h"

namespace impact {
namespace ltlspec {

    using Labels = std::map<std::string, std::set<int>>;   // atom -> states where it holds

    // Robust satisfaction probability of `formula` over the IMDP (controller-Pmax).
    // pessimistic = nature adversarial (robust lower bound); else optimistic.
    // THROWS std::runtime_error on a parse error or an out-of-fragment formula
    // (message names ISSUE-0016 for the LDBA route).
    solve::IntervalResult synthesize(const solve::IMDPModel& m, const Labels& labels,
                                     const std::string& formula, bool pessimistic, double eps = 1e-6);

} // namespace ltlspec
} // namespace impact

#endif // IMPACT_LTLSPEC_H
