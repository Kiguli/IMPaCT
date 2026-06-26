#ifndef IMPACT_PCTL_H
#define IMPACT_PCTL_H

// ============================================================================
// PCTL / CTL model checking on Interval MDPs (stretch: logics beyond LTL).
//
// State subformulas evaluate to STATE SETS (compose bottom-up with set algebra);
// each PCTL path operator returns a sound robust probability interval per state via
// the verified solver core. We expose the controller-MAXimizing ("Pmax") flavour
// with nature pessimistic (adversarial, robust lower bound) or optimistic
// (cooperative, upper bound) — the natural "best controller, robust to the interval
// uncertainty" semantics matching solve::maxReach/maxSafety. (Pmin / adversarial
// controller is future work.)
//
// Path operators:
//   Next   X psi            -> next* (one robust Bellman step to psi-states)
//   Until  phi U psi        -> until* (reach psi while remaining in phi)
//   BUntil phi U^<=k psi     -> boundedUntil* (finite-horizon, exact)
//   Finally  F psi == true U psi      (== solve::maxReach*)
//   Globally G phi          (== solve::maxSafety* over the safe set phi)
//
// CTL (qualitative branching time) is sugar over these with {>0, =1} thresholds:
//   EF=Pmax[F]>0, AF=Pmin... (here AF approximated as Pmax-pessimistic[F]=1),
//   EG=Pmax[G]>0, AG=Pmax-pessimistic[G]=1, EX/AX via next. See ctl* helpers; the
//   qualitative-exact (graph positive/almost-sure) versions are a documented refinement.
//
// Refs: Hansson-Jonsson (FAC 1994, PCTL); Baier-Katoen Ch.10; robust/interval-MDP
// PCTL — Puggelli-Li-Sangiovanni-Vincentelli-Seshia (CAV 2013). Contracts:
// tests/unit/test_pctl.cpp (differential vs reach/safety + explicit finite-horizon DP).
// ============================================================================

#include <set>
#include "solve.h"

namespace impact {
namespace pctl {

    // X psi : robust probability that the next state satisfies psi.
    solve::IntervalResult nextPessimistic(const solve::IMDPModel& m, const std::set<int>& psi, double eps);
    solve::IntervalResult nextOptimistic (const solve::IMDPModel& m, const std::set<int>& psi, double eps);

    // phi U psi : reach psi while remaining in phi (unbounded).
    solve::IntervalResult untilPessimistic(const solve::IMDPModel& m,
                                           const std::set<int>& phi, const std::set<int>& psi, double eps);
    solve::IntervalResult untilOptimistic (const solve::IMDPModel& m,
                                           const std::set<int>& phi, const std::set<int>& psi, double eps);

    // phi U^{<=k} psi : reach psi within k steps while remaining in phi (finite horizon, exact).
    solve::IntervalResult boundedUntilPessimistic(const solve::IMDPModel& m,
                            const std::set<int>& phi, const std::set<int>& psi, int k, double eps);
    solve::IntervalResult boundedUntilOptimistic (const solve::IMDPModel& m,
                            const std::set<int>& phi, const std::set<int>& psi, int k, double eps);

    // Threshold verdict for P_{~ p}[path]: with a sound interval [lower,upper] the
    // formula is definitely true / false / undetermined at a state.
    enum class Cmp { Ge, Gt, Le, Lt };
    enum class Verdict { Sat, Unsat, Unknown };
    Verdict check(double lower, double upper, Cmp op, double p);

    // States where P_{~p}[path] holds for sure (Sat) given the interval result.
    std::set<int> satStates(const solve::IntervalResult& r, Cmp op, double p);

    // ---- CTL sugar (qualitative; uses the Pmax interval + {>0,=1} thresholds) ----
    // EF psi: exists a strategy reaching psi with positive probability.
    std::set<int> EF(const solve::IMDPModel& m, const std::set<int>& psi, double eps);
    // AG phi: for all (robust) the system stays in phi almost surely.
    std::set<int> AG(const solve::IMDPModel& m, const std::set<int>& phi, double eps);

} // namespace pctl
} // namespace impact

#endif // IMPACT_PCTL_H
