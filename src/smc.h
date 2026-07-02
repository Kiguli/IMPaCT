#ifndef IMPACT_SMC_H
#define IMPACT_SMC_H

// ============================================================================
// Statistical model checking (simulation) for POINT Markov chains: Monte-Carlo
// estimation of P(F<=k target) with confidence intervals, and Wald's sequential
// probability ratio test (SPRT) for threshold queries P >= theta.
//
// Refs: H. L. S. Younes, R. G. Simmons, "Probabilistic Verification of Discrete
// Event Systems Using Acceptance Sampling", CAV 2002 (SPRT-based statistical MC);
// T. Herault, R. Lassaigne, F. Magniette, S. Peyronnet, "Approximate Probabilistic
// Model Checking", VMCAI 2004 (APMC: Chernoff-Hoeffding sample bounds);
// A. Wald, "Sequential Tests of Statistical Hypotheses", Ann. Math. Stat. 16(2),
// 1945 (the SPRT itself).
//
// Scope: the model must be a point DTMC (one action per state, lo==hi); MDP
// scheduler sampling is future work (as in PRISM, whose simulator is DTMC-only).
// ============================================================================

#include <vector>
#include <set>
#include <cstdint>
#include "solve.h"

namespace impact {
namespace smc {

    struct Estimate {
        double mean;            // fraction of sampled paths that hit the target
        double ciLo, ciHi;      // Wilson 95% confidence interval
        double chernoffEps;     // APMC half-width: P(|mean - p| >= eps) <= delta (delta = 0.05)
        long long successes, samples;
    };

    // Monte-Carlo estimate of P(F<=horizon target) from `init` (horizon steps max;
    // paths stop early in absorbing states). Throws if the model is not a point chain.
    Estimate estimateReach(const solve::IMDPModel& m, const std::set<int>& target,
                           int init, int horizon, long long samples, std::uint64_t seed);

    // Wald SPRT for H0: p >= theta+delta vs H1: p <= theta-delta with error bounds
    // alpha=beta=0.01 (indifference region 2*delta). Returns +1 (accept p>=theta),
    // -1 (reject), 0 (max samples hit, undecided) and the samples used.
    int sprt(const solve::IMDPModel& m, const std::set<int>& target, int init, int horizon,
             double theta, double delta, long long maxSamples, std::uint64_t seed,
             long long* samplesUsed);

} // namespace smc
} // namespace impact

#endif // IMPACT_SMC_H
