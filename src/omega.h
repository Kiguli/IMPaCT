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
// Robustness: accepting MECs here use the support-graph MEC structure, which is
// EXACT for point MDPs and for the OPTIMISTIC sense (nature cooperates). The
// PESSIMISTIC (robust) accepting-EC notion — where nature may eject the play from
// a candidate EC — needs the robust-EC machinery (Dutreix-Coogan permanent winning
// components / Weininger et al. game reduction); tracked in ISSUE-0009.
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

    // Max probability of the Büchi objective (visit `accepting` infinitely often),
    // = max reachability of the accepting MECs. Optimistic (nature cooperative) and
    // pessimistic (robust) variants; see the robustness note above for pessimistic.
    solve::IntervalResult maxBuchiOptimistic(const solve::IMDPModel& m,
                                             const std::set<int>& accepting, double eps);
    solve::IntervalResult maxBuchiPessimistic(const solve::IMDPModel& m,
                                              const std::set<int>& accepting, double eps);

} // namespace omega
} // namespace impact

#endif // IMPACT_OMEGA_H
