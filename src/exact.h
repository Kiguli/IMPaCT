#ifndef IMPACT_EXACT_H
#define IMPACT_EXACT_H

// ============================================================================
// EXACT (arbitrary-precision-free, rational) robust reachability on (interval)
// MDPs. Values are rationals (long long num/den, __int128 intermediates, gcd
// normalisation; overflow throws rather than losing exactness).
//
// Algorithm: policy iteration in the Hoffman-Karp style over the controller
// policy and nature's O-maximization vertex (which depends only on the ORDERING
// of the value vector, so both spaces are finite):
//   1. qualitative preprocessing: compute the exact robust prob-0 set P0 (the
//      largest target-free set nature can confine the play to: every action has
//      sum of upper bounds inside >= 1 and zero lower bounds outside), pinned to 0;
//   2. evaluate the induced point chain exactly (rational Gaussian elimination,
//      with chain-level unreachability pinned to 0);
//   3. improve controller action and nature vertex by exact O-maximization;
//      repeat until stable;
//   4. certify: verify the result is an exact fixpoint of the robust Bellman
//      operator. Fixpoints that vanish on P0 are unique for reachability, so the
//      certified fixpoint IS the robust value (Baier-Katoen, Thm 10.100ff;
//      Haddad-Monmege 2018 for the interval/MDP setting).
//
// O-maximization is division-free (sort + additive mass assignment), so nature's
// exact vertex is computed exactly (Givan-Leach-Dean 2000). Robust policy
// iteration: Iyengar 2005 (Sect. 3.3); classical PI: Puterman 1994, Ch. 6-7.
// PRISM's -exact and Storm's --exact support POINT models only (verified
// empirically) — exact robust interval solving is unique to IMPaCT.
// ============================================================================

#include <string>
#include <vector>

namespace impact {
namespace exact {

    struct Result {
        std::string fraction;   // exact value at the queried state, e.g. "1/4"
        double approx;          // decimal approximation
        int iterations;         // policy-improvement rounds
        bool certified;         // exact robust-Bellman fixpoint check passed
    };

    // Exact robust max-reachability P(F target) at `state` (or init if -1) for the
    // .imdp model at `path` (point or interval; decimals parsed exactly as fractions).
    // pessimistic = adversarial nature. Throws on rational overflow or non-convergence.
    Result maxReach(const std::string& path, const std::string& targetLabel,
                    int state, bool pessimistic);

} // namespace exact
} // namespace impact

#endif // IMPACT_EXACT_H
