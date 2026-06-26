#ifndef IMPACT_STL_H
#define IMPACT_STL_H

// ============================================================================
// Bounded Signal Temporal Logic (STL) on Interval-MDP abstractions of continuous
// stochastic systems (stretch: logics beyond LTL). We compute the robust
// PROBABILITY that the system's (discretized) signal satisfies a bounded STL
// formula — "probabilistic STL" — rather than the continuous spatial-robustness
// degree (a distinct single-trace semantics, noted below).
//
// STL's defining pieces vs LTL/PCTL:
//   * ATOMIC PREDICATES over the real state, mu(x) >= 0 — mapped here to the grid
//     cells whose centre satisfies the predicate (predicateCells), bridging the
//     continuous signal to the abstraction.
//   * TIME-BOUNDED temporal operators with discrete-step windows [a,b]:
//       F_[a,b] psi   eventually-in-window      (>= some step in [a,b])
//       G_[a,b] psi   always-in-window          (all steps in [a,b])
//       phi U_[0,b] psi  bounded until
//     evaluated by EXACT finite-horizon value iteration (no convergence needed),
//     reusing pctl::boundedUntil for F/U and a finite-horizon safety DP for G.
//     Windowed [a,b] = free (unconstrained) robust evolution for a steps composed
//     with the [0,b-a] operator as the terminal value.
//
// Robust controller-Pmax with nature pessimistic (adversarial; sound lower bound)
// or optimistic. Refs: Maler-Nickovic (FORMATS 2004, STL); Sadigh-Kapoor
// (RSS 2016, probabilistic STL for control). Spatial-robustness STL is future work.
// Contracts: tests/unit/test_stl.cpp.
// ============================================================================

#include <set>
#include <vector>
#include <functional>
#include "solve.h"

namespace impact {
namespace stl {

    // Atomic predicate mu(x) (>=0) -> set of grid cells whose CENTRE satisfies it.
    // Per-dimension grid {lb, eta, count}; linear index lin = sum_d j_d * stride_d
    // with stride_0 = 1 (row-major over dimensions), matching the abstraction.
    std::set<int> predicateCells(const std::vector<double>& lb,
                                 const std::vector<double>& eta,
                                 const std::vector<int>& count,
                                 const std::function<bool(const std::vector<double>&)>& holds);

    // F_[0,b] psi : eventually psi within b steps (robust controller-Pmax).
    solve::IntervalResult eventuallyBounded(const solve::IMDPModel& m, const std::set<int>& psi, int b, bool pessimistic);
    // G_[0,b] psi : psi holds at every step 0..b.
    solve::IntervalResult globallyBounded(const solve::IMDPModel& m, const std::set<int>& psi, int b, bool pessimistic);
    // phi U_[0,b] psi : bounded until.
    solve::IntervalResult untilBounded(const solve::IMDPModel& m, const std::set<int>& phi,
                                       const std::set<int>& psi, int b, bool pessimistic);
    // F_[a,b] psi and G_[a,b] psi : the windowed forms (0 <= a <= b).
    solve::IntervalResult eventuallyWindow(const solve::IMDPModel& m, const std::set<int>& psi, int a, int b, bool pessimistic);
    solve::IntervalResult globallyWindow(const solve::IMDPModel& m, const std::set<int>& psi, int a, int b, bool pessimistic);

} // namespace stl
} // namespace impact

#endif // IMPACT_STL_H
