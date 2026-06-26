#ifndef IMPACT_SOLVE_H
#define IMPACT_SOLVE_H

// ============================================================================
// Sound robust interval iteration for reachability on Interval MDPs (Phase 1c).
//
// Controller maximizes the probability of reaching a target set; nature picks
// transition probabilities within the per-successor intervals either
// adversarially (pessimistic / robust) or cooperatively (optimistic). Returns a
// SOUND sandwich [lower, upper] per state with gap <= 2*eps.
//
// Soundness needs a unique fixpoint, so end components are collapsed
// (graph::mecs) before interval iteration (Baier et al., CAV 2017;
// Haddad & Monmege, TCS 2018). The inner robust Bellman solve is O-maximization
// (omax::optimize). Contracts: tests/unit/test_interval_iteration.cpp.
// ============================================================================

#include <vector>
#include <set>

namespace impact {
namespace solve {

    struct Interval { int to; double lo; double hi; };
    using ActionDist   = std::vector<Interval>;      // one prob interval per successor
    using StateActions = std::vector<ActionDist>;    // actions available at a state
    using IMDPModel    = std::vector<StateActions>;  // per-state action lists

    struct IntervalResult {
        std::vector<double> lower;  // sound lower bound on the value, per state
        std::vector<double> upper;  // sound upper bound on the value, per state
        int iterations;
    };

    // Robust max-reachability: controller maximizes, nature MINIMIZES within the
    // intervals (worst-case / pessimistic). lower[s] <= V*(s) <= upper[s], gap <= 2*eps.
    //
    // SCOPE (see issues/0003): sound AND convergent for point-probability MDPs and
    // for interval MDPs without nature-confinable end components. For interval MDPs
    // where nature can confine the play via lo=0 leaving edges, the upper bound is
    // still sound but may not converge (gap may stay open) — robust-EC handling for
    // that case is planned with Phase 3. maxReachOptimistic is convergent on intervals.
    IntervalResult maxReachPessimistic(const IMDPModel& m,
                                       const std::set<int>& targets,
                                       double eps);

    // Best-case: controller maximizes, nature MAXIMIZES too (optimistic).
    IntervalResult maxReachOptimistic(const IMDPModel& m,
                                      const std::set<int>& targets,
                                      double eps);

} // namespace solve
} // namespace impact

#endif // IMPACT_SOLVE_H
