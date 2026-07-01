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

    // ---- Robust expected reward on interval MDPs (robust dynamic programming;
    // Iyengar, "Robust Dynamic Programming", Math. OR 2005; Nilim & El Ghaoui,
    // Oper. Res. 2005; the interval ambiguity set is resolved in closed form by the
    // same O-maximization). Two objectives:
    //
    //  * REACHABILITY reward: expected total state reward accumulated until the target
    //    set is first reached (target absorbing, reward stops). Robust Bellman:
    //      V(s) = reward[s] + max_a  opt_{p in [lo,hi]} sum_s' p(s') V(s'),   V(target)=0
    //    The controller MAXIMIZES; nature resolves the intervals adversarially
    //    (pessimistic, opt = Sense::Min on the continuation) or cooperatively (optimistic,
    //    Sense::Max). If the target is not reached almost surely and rewards are positive
    //    the value is +infinity (reported as HUGE_VAL, matching PRISM/Storm "Infinity").
    //
    //  * DISCOUNTED reward (gamma in (0,1)): V(s) = reward[s] + gamma * max_a opt_p sum p V,
    //    a contraction, so it always converges (no target needed). Used by IntervalMDP.jl.
    //
    // `reward` is a per-state vector (size == model size). `natureAdversarial` = true is
    // the robust/pessimistic resolution (nature worst-case for the controller); false is
    // cooperative/optimistic. controllerMax selects Max/Min over actions (max/min reward).
    // Returns lower==upper (single VI value); a divergent (+inf) value is HUGE_VAL.
    IntervalResult expReachReward(const IMDPModel& m, const std::set<int>& targets,
                                  const std::vector<double>& reward, double eps,
                                  bool natureAdversarial, bool controllerMax);
    IntervalResult expDiscountedReward(const IMDPModel& m, const std::vector<double>& reward,
                                       double gamma, double eps, bool natureAdversarial, bool controllerMax);

    // Convenience wrappers: MAX expected reward, robust (nature adversarial = Sense::Min)
    // or optimistic (nature cooperative = Sense::Max).
    IntervalResult maxReachRewardPessimistic(const IMDPModel& m, const std::set<int>& targets,
                                             const std::vector<double>& reward, double eps);
    IntervalResult maxReachRewardOptimistic (const IMDPModel& m, const std::set<int>& targets,
                                             const std::vector<double>& reward, double eps);

    // ---- Robust LONG-RUN AVERAGE (mean-payoff) reward on interval MDPs ------------------
    // Two-phase value iteration (Ashok, Chatterjee, Daca, Kretinsky, Meggendorfer,
    // "Value Iteration for Long-Run Average Reward in MDPs", CAV 2017): (1) the robust
    // average reward (gain) of each maximal end component is computed by relative value
    // iteration with Puterman's (1994) aperiodicity transform, the interval ambiguity set
    // resolved by O-maximization (Iyengar 2005); (2) a max-reachability "cash-out" VI funnels
    // each state to the best robustly-reachable MEC gain. For robust/interval MDPs
    // specifically see Chatterjee, Goharshady, Karrabi, Novotny, Zikelic, "Solving Long-run
    // Average Reward Robust MDPs via Stochastic Games", IJCAI 2024 (arXiv:2312.13912) — that
    // work uses policy iteration / games; here we use the VI route.
    // `natureAdversarial` = robust (worst-case gain); controllerMax = LRAmax vs LRAmin.
    IntervalResult longRunAverage(const IMDPModel& m, const std::vector<double>& reward,
                                  double eps, bool natureAdversarial, bool controllerMax);
    IntervalResult maxLRAPessimistic(const IMDPModel& m, const std::vector<double>& reward, double eps);
    IntervalResult maxLRAOptimistic (const IMDPModel& m, const std::vector<double>& reward, double eps);

} // namespace solve
} // namespace impact

#endif // IMPACT_SOLVE_H
