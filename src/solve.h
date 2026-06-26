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

    // Selectable solver (toolbox of literature methods):
    //  - OptimisticVI: optimistic value iteration (Hartmanns & Kaminski, CAV 2020) —
    //    VI from below for the lower bound + a verified inductive (pre-fixpoint)
    //    upper bound (F(U) <= U => V* <= U, Knaster-Tarski). Needs no MEC handling;
    //    sound and convergent including nature-confinable ECs (resolves ISSUE-0003).
    //  - MECCollapse: interval iteration with end-component collapse (Haddad-Monmege
    //    TCS 2018; Baier et al. CAV 2017). Faster on controller end components, but
    //    its support-graph collapse does NOT converge on pessimistic interval
    //    nature-traps (ISSUE-0003) — valid for point MDPs and the optimistic sense.
    enum class Method { OptimisticVI, MECCollapse };

    // Robust max-reachability. Controller maximizes; nature MINIMIZES within the
    // intervals (pessimistic) or MAXIMIZES (optimistic). Returns sound
    // lower[s] <= V*(s) <= upper[s] with gap <= 2*eps. The 3-arg forms use the
    // default (OptimisticVI); the 4-arg forms select the method.
    IntervalResult maxReachPessimistic(const IMDPModel& m, const std::set<int>& targets, double eps);
    IntervalResult maxReachOptimistic (const IMDPModel& m, const std::set<int>& targets, double eps);
    IntervalResult maxReachPessimistic(const IMDPModel& m, const std::set<int>& targets, double eps, Method method);
    IntervalResult maxReachOptimistic (const IMDPModel& m, const std::set<int>& targets, double eps, Method method);

    // Robust safety: max over controller of P(never reach `avoid`) = 1 - min-reach to
    // avoid. Pessimistic = nature adversarial; optimistic = nature cooperative.
    // Returns sound [lower,upper] on the safety probability (gap <= 2*eps).
    IntervalResult maxSafetyPessimistic(const IMDPModel& m, const std::set<int>& avoid, double eps);
    IntervalResult maxSafetyOptimistic (const IMDPModel& m, const std::set<int>& avoid, double eps);

} // namespace solve
} // namespace impact

#endif // IMPACT_SOLVE_H
