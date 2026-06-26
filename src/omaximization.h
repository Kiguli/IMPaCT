#ifndef IMPACT_OMAXIMIZATION_H
#define IMPACT_OMAXIMIZATION_H

// ============================================================================
// O-maximization: the robust inner Bellman solve over a box-interval ambiguity
// set, in closed form (sort-and-assign). This is the standalone, unit-tested
// extraction of the algorithm IMPaCT already runs in its "sorted" synthesis
// (src/GPU_synthesis.cpp): start every successor at its lower bound, then push
// the residual mass (1 - sum of lowers) to successors in value order — lowest
// value first to MINIMIZE, highest first to MAXIMIZE — capping each at its upper
// bound, with a single pivot taking the remainder.
//
// Refs: Givan-Leach-Dean, "Bounded-parameter MDPs", AIJ 2000 (pessimistic/
// optimistic orderings); Lahijanian-Andersson-Belta, IEEE TAC 2015.
//
// Verified against an independent brute-force vertex enumeration in
// tests/unit/test_omaximization.cpp and tests/oracles/oracles.py.
// ============================================================================

#include <vector>

namespace impact {
namespace omax {

    enum class Sense { Min, Max };

    struct Result {
        std::vector<double> p;  // optimal feasible distribution over successors
        double value;           // dot(p, V)
    };

    // See tests/contracts/contracts.h for the full behavioural contract.
    // THROWS std::invalid_argument on size mismatch, n==0, any lower>upper, or an
    // infeasible box (sum(lower) > 1 or sum(upper) < 1, beyond fp tolerance).
    Result optimize(const std::vector<double>& lower,
                    const std::vector<double>& upper,
                    const std::vector<double>& V,
                    Sense sense);

} // namespace omax
} // namespace impact

#endif // IMPACT_OMAXIMIZATION_H
