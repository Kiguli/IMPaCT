#ifndef IMPACT_OMEGA_H
#define IMPACT_OMEGA_H

// ============================================================================
// omega-regular synthesis on Interval MDPs (Phase 3).
//
// Büchi objective ("visit the accepting set infinitely often") is reduced to
// REACHABILITY of accepting end components: a maximal end component that contains
// an accepting state is "good" (the controller can cycle within it forever and so
// visit an accepting state infinitely often); the Büchi value is the max
// probability of reaching the union of good MECs. This is the classical reduction
// (de Alfaro, PhD 1997; Baier & Katoen, Principles of Model Checking, Ch. 10) and
// it underlies the LDBA / good-for-MDP product route (Sickert et al., CAV 2016;
// Hahn et al., TACAS 2020): an LDBA product turns an LTL objective into Büchi, and
// the accepting frontier is exactly such an accepting set.
//
// Robustness: the OPTIMISTIC value (nature cooperative) uses the support-graph
// accepting-MEC structure, which is exact for that sense and for point MDPs. The
// PESSIMISTIC (robust) value needs more than a support-MEC: even inside a support
// end component (which nature cannot LEAVE, since all leaving edges have hi=0),
// nature can ROUTE AROUND the accepting state using lo=0 edges, so "EC contains an
// accepting state => value 1" is unsound (ISSUE-0009). The robust value uses the
// robust almost-sure-Büchi winning region (robustBuchiWinningStates): a nested
// fixpoint of (i) robust EC closure and (ii) robust (controller-vs-nature)
// almost-sure reachability of the accepting set. This is the qualitative
// 2.5-player Büchi computation (Chatterjee-Henzinger graph games; the IMDP/robust
// reading follows Dutreix-Coogan permanent components and Asadi et al. force
// attractors); the quantitative value is robust reachability of that region.
//
// Contracts: tests/unit/test_omega.cpp.
// ============================================================================

#include <vector>
#include <set>
#include "solve.h"

namespace impact {
namespace omega {

    // States lying in some maximal end component that contains an accepting state
    // (the "good"/accepting MECs). Reaching any of these => Büchi is satisfiable
    // from there with probability 1 (optimistic / point-MDP semantics).
    std::vector<int> acceptingMECStates(const solve::IMDPModel& m, const std::set<int>& accepting);

    // Robust (pessimistic) almost-sure-Büchi winning region: the states from which
    // the controller can force visiting `accepting` infinitely often for ALL nature
    // resolutions within the intervals. Sound for the robust sense (no nature route
    // can defeat it); for point MDPs it coincides with the support-MEC region.
    // maxBuchiPessimistic = robust reachability of this set. (ISSUE-0009.)
    std::vector<int> robustBuchiWinningStates(const solve::IMDPModel& m,
                                              const std::set<int>& accepting);

    // Max probability of the Büchi objective (visit `accepting` infinitely often),
    // = max reachability of the accepting MECs. Optimistic (nature cooperative) and
    // pessimistic (robust) variants; see the robustness note above for pessimistic.
    solve::IntervalResult maxBuchiOptimistic(const solve::IMDPModel& m,
                                             const std::set<int>& accepting, double eps);
    solve::IntervalResult maxBuchiPessimistic(const solve::IMDPModel& m,
                                              const std::set<int>& accepting, double eps);

    // Generalized Büchi ("patrol"): visit EACH set in `accSets` infinitely often
    // (the conjunction of GF over the sets). Degeneralized to a single Büchi
    // objective via a round-robin counter product (state = s*k + c), then solved
    // with the Büchi solver above. The returned IntervalResult is indexed by the
    // ORIGINAL IMDP states (value taken at counter 0). With one set this equals
    // maxBuchi*; with zero sets the objective is vacuously true (value 1).
    solve::IntervalResult maxGenBuchiOptimistic(const solve::IMDPModel& m,
                                  const std::vector<std::set<int>>& accSets, double eps);
    solve::IntervalResult maxGenBuchiPessimistic(const solve::IMDPModel& m,
                                  const std::vector<std::set<int>>& accSets, double eps);

    // Persistence / co-Büchi: "F G p" — eventually always inside `pStates` (the
    // reach-then-stay objective). The value is robust/optimistic reachability of the
    // largest sub-region of `pStates` the controller can remain in forever: the
    // greatest W ⊆ pStates where every state has an action that keeps the play in W
    // for all nature (pessimistic, may-support ⊆ W) or for some nature (optimistic,
    // nature can contain all mass in W). Once W is reached the controller stays in p
    // forever, so F G p = reach W. The robust analogue of robust safety/invariance.
    solve::IntervalResult maxPersistenceOptimistic(const solve::IMDPModel& m,
                                  const std::set<int>& pStates, double eps);
    solve::IntervalResult maxPersistencePessimistic(const solve::IMDPModel& m,
                                  const std::set<int>& pStates, double eps);

} // namespace omega
} // namespace impact

#endif // IMPACT_OMEGA_H
