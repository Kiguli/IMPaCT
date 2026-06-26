#ifndef IMPACT_PTA_H
#define IMPACT_PTA_H

// ============================================================================
// Probabilistic Timed Automata (PTA) -> finite MDP, then reuse the verified
// reachability solver. A PTA is a timed automaton whose edges carry, instead of a
// single (reset,target), a PROBABILITY DISTRIBUTION over (reset,target) branches.
//
// The forward zone graph induces a finite MDP whose states are symbolic states
// (location, canonical zone): from (l,Z), each edge with guard g enabled in Z
// (Z ∩ g nonempty) is an ACTION; its successor distribution sends probability p_k to
// the symbolic state reached by branch k (reset_k applied to Z∩g, delayed within the
// target invariant, extrapolated). Maximum reachability of a target LOCATION is then
// exactly solve::maxReach on this MDP. (This zone-MDP gives the exact MAXIMUM
// probability; minimum probability needs the backward/game construction — noted as a
// limitation, ISSUE-0014.) Symbolic states are identified by canonical-zone EQUALITY
// (not inclusion) so branch probabilities stay exact; extrapolation keeps the graph
// finite.
//
// Refs: Kwiatkowska-Norman-Segala-Sproston (TCS 2002, PTA); Kwiatkowska-Norman-
// Sproston-Wang (Information and Computation 2007, symbolic/zone PTA model checking);
// Behrmann et al. (2006, extrapolation). Contracts: tests/unit/test_pta.cpp.
// ============================================================================

#include <vector>
#include <set>
#include "ta.h"        // reuse Constraint + clk* helpers + invariant/guard model
#include "solve.h"

namespace impact {
namespace pta {

    using ta::Constraint;

    struct Branch { double prob; std::vector<int> reset; int to; };   // sum of prob over a distribution = 1
    struct Edge   { int from; std::vector<Constraint> guard; std::vector<Branch> dist; };

    struct PTA {
        int nLoc = 0, nClocks = 0, init = 0;
        std::vector<std::vector<Constraint>> invariant;   // size nLoc
        std::vector<Edge> edges;
        std::vector<long long> kmax;                       // size nClocks+1 (kmax[0]=0)
    };

    // The induced symbolic MDP (point distributions: lo==hi).
    struct SymbolicMDP {
        solve::IMDPModel model;     // one state per reachable symbolic state (+ a deadlock sink)
        std::set<int> targets;      // symbolic states whose location == targetLoc
        int init = 0;
        int nSym = 0;
        std::vector<int> locOf;     // location of each symbolic state (sink = -1)
        std::vector<std::string> descr;  // human-readable "L<loc>: x_i∈[lo,hi]" per state
        bool hitCap = false;
    };

    SymbolicMDP build(const PTA& p, int targetLoc, int maxStates = 200000);

    // Maximum probability of reaching `targetLoc` (exact for Pmax).
    double maxReachLocation(const PTA& p, int targetLoc, double eps = 1e-7, int maxStates = 200000);

    // --- Digital-clocks engine (resolves the Pmin gap, ISSUE-0014) ---------------
    // For CLOSED, diagonal-free PTAs (non-strict guards/invariants, x-vs-constant),
    // replacing dense clocks with bounded integers and making time-elapse an explicit
    // "tick" action yields a finite MDP that is EXACT for BOTH minimum and maximum
    // reachability (Kwiatkowska-Norman-Sproston, FMSD 2006). REQUIREMENT: kmax[i] must
    // be >= every constant clock i is compared against in ANY guard OR invariant (so
    // saturation happens strictly above all constants). This is a second, independent
    // engine — its Pmax must agree with the zone engine (used as a cross-check).
    SymbolicMDP buildDigital(const PTA& p, int targetLoc, int maxStates = 200000);
    double maxReachLocationDigital(const PTA& p, int targetLoc, double eps = 1e-7, int maxStates = 200000);
    double minReachLocationDigital(const PTA& p, int targetLoc, double eps = 1e-7, int maxStates = 200000);

} // namespace pta
} // namespace impact

#endif // IMPACT_PTA_H
