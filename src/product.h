#ifndef IMPACT_PRODUCT_H
#define IMPACT_PRODUCT_H

// ============================================================================
// IMDP x DFA product for co-safe-LTL / LTLf reachability (Phase 2 part 3).
//
// Given an IMDP, a labelling (the set of atomic propositions true in each IMDP
// state) and an LTLf DFA, build the product IMDP whose max-reachability of the
// "accepting" product states equals the probability that the labelled path
// satisfies the spec (good-prefix / co-safe semantics: accepting DFA states are
// targets). The product reuses solve::maxReach* unchanged.
//
// Contracts: tests/unit/test_product.cpp (incl. "F goal" == plain reachability).
// ============================================================================

#include <vector>
#include <set>
#include "solve.h"
#include "ltl.h"

namespace impact {
namespace product {

    using Label = ltl::Letter;     // set of AP names true in a state

    struct Product {
        solve::IMDPModel model;    // product IMDP, state index = s*nQ + q
        std::set<int> targets;     // accepting product states (DFA state accepting)
        int start;                 // product start state (after reading L(s0))
        int nProd;
    };

    // m       : the IMDP (solve model)
    // labels  : labels[s] = APs true in IMDP state s (size == m.size())
    // dfa     : LTLf DFA (ltl::toDFA)
    // s0      : the IMDP initial state
    Product build(const solve::IMDPModel& m,
                  const std::vector<Label>& labels,
                  const ltl::DFA& dfa,
                  int s0);

} // namespace product
} // namespace impact

#endif // IMPACT_PRODUCT_H
