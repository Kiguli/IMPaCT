#ifndef IMPACT_TA_H
#define IMPACT_TA_H

// ============================================================================
// Timed automata: model + zone-graph forward reachability (stretch model, built on
// the DBM zone abstraction). Locations with clock invariants; edges with clock
// guards and resets. Reachability is decided by exploring symbolic states
// (location, zone): delay within the invariant, take an edge (intersect guard,
// reset clocks), delay again within the target invariant; terminate via maximal-
// constant extrapolation + zone inclusion (a finite zone graph).
//
// This is the non-probabilistic core; a probabilistic-timed-automaton (PTA) front
// end adds probability distributions to edges, so the zone graph becomes a finite
// (I)MDP that the verified solve::maxReach / pctl solvers handle — the integration
// point that ties timed models into the rest of IMPaCT.
//
// Refs: Alur-Dill (TCS 1994); Bengtsson-Yi (2004, zone-graph algorithms);
// Behrmann-Bouyer-Larsen-Pelanek (2006, extrapolation); PTA target
// Kwiatkowska-Norman-Segala-Sproston (TCS 2002). Contracts: tests/unit/test_ta.cpp.
// ============================================================================

#include <vector>
#include "dbm.h"

namespace impact {
namespace ta {

    // A clock constraint as a DBM difference x_i - x_j (</<=) c. Use the helpers
    // below to build the usual clock comparisons (clocks are 1..n; 0 is the zero clock).
    using Constraint = struct { int i; int j; dbm::Bound b; };

    inline Constraint clkLe(int x, long long v) { return { x, 0, dbm::Bound::leq(v) }; }   // x <= v
    inline Constraint clkLt(int x, long long v) { return { x, 0, dbm::Bound::lt(v)  }; }   // x <  v
    inline Constraint clkGe(int x, long long v) { return { 0, x, dbm::Bound::leq(-v) }; }  // x >= v
    inline Constraint clkGt(int x, long long v) { return { 0, x, dbm::Bound::lt(-v)  }; }  // x >  v

    struct Edge {
        int from;
        std::vector<Constraint> guard;   // must hold to take the edge
        std::vector<int> reset;          // clocks set to 0 on taking the edge
        int to;
    };

    struct TA {
        int nLoc = 0;
        int nClocks = 0;
        int init = 0;
        std::vector<std::vector<Constraint>> invariant;  // per location (size nLoc)
        std::vector<Edge> edges;
        std::vector<long long> kmax;                     // size nClocks+1 (kmax[0]=0)
    };

    // Is `target` location reachable from the initial symbolic state? `maxStates` is
    // a safety valve on the symbolic-state count (the zone graph is finite under
    // extrapolation; the cap guards against malformed input). Returns false if the
    // cap is hit without reaching `target` (reported via `hitCap`).
    bool reachable(const TA& ta, int target, int maxStates = 200000, bool* hitCap = nullptr);

} // namespace ta
} // namespace impact

#endif // IMPACT_TA_H
