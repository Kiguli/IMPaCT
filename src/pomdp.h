#ifndef IMPACT_POMDP_H
#define IMPACT_POMDP_H

// ============================================================================
// Partially Observable MDPs — exact finite-horizon reachability via belief-state
// value iteration (stretch: other robust-MDP models). The controller observes only
// an observation o (not the state); it acts on the BELIEF (posterior state
// distribution). Finite-horizon Pmax of reaching a target set is computed by exact
// backward induction over the belief tree:
//   V_0(b)   = b(target)
//   V_t(b)   = max_a  sum_o  P(o | b,a) * V_{t-1}( belief-update(b,a,o) )
// with target states made absorbing, so "in target at horizon H" == "reached within
// H". Beliefs are continuous; the reachable belief tree (branching |A|x|O|) is finite
// per horizon, and exact (no alpha-vector pruning needed for finite-horizon value).
//
// Refs: Smallwood-Sondik (Operations Research 1973, finite-horizon POMDP control);
// Kaelbling-Littman-Cassandra (Artificial Intelligence 1998, POMDP planning).
// The ROBUST / interval-POMDP extension (adversarial nature over transition/obs
// intervals; robust belief update) is future work (e.g. Osogami, NeurIPS 2015).
// Contracts: tests/unit/test_pomdp.cpp.
// ============================================================================

#include <vector>
#include <set>

namespace impact {
namespace pomdp {

    struct POMDP {
        int nStates = 0, nActions = 0, nObs = 0;
        std::vector<std::vector<std::vector<double>>> T;   // T[a][s][s'] transition prob
        std::vector<std::vector<std::vector<double>>> O;   // O[a][s'][o] obs prob (in s' after a)
        std::vector<double> b0;                            // initial belief over states
    };

    // belief-update: posterior after taking action a and seeing observation o.
    // Returns the (unnormalized prob of o) in *probO, and the normalized posterior
    // (empty if probO == 0). target states are treated as absorbing if `absorb`.
    std::vector<double> beliefUpdate(const POMDP& p, const std::vector<double>& b,
                                     int a, int o, const std::set<int>& target,
                                     bool absorb, double* probO);

    // Max probability of reaching `target` within `horizon` steps from b0 (Pmax).
    double maxReachFiniteHorizon(const POMDP& p, const std::set<int>& target, int horizon);

    // Same, from an explicit belief (for testing / sub-problems).
    double maxReachFromBelief(const POMDP& p, const std::vector<double>& belief,
                              const std::set<int>& target, int horizon);

} // namespace pomdp
} // namespace impact

#endif // IMPACT_POMDP_H
