#ifndef IMPACT_DBM_H
#define IMPACT_DBM_H

// ============================================================================
// Difference Bound Matrices (DBMs) — the zone abstraction for timed automata
// (stretch: timed-automaton models, after STL/CTL). A "zone" is a convex set of
// clock valuations defined by difference constraints x_i - x_j < / <= c. The DBM is
// the canonical data structure for zone-based reachability of timed automata; here
// it is the foundation a (probabilistic) timed-automaton front-end builds on (the
// zone graph yields a finite MDP/IMDP that the verified reach/PCTL solvers handle).
//
// Clocks are indexed 1..n; index 0 is the constant-zero reference clock, so a bound
// D[i][j] encodes  x_i - x_j  <(strict) / <=  c   (with x_0 = 0). Operations
// implemented (Bengtsson-Yi, "Timed Automata: Semantics, Algorithms and Tools",
// 2004): canonicalize (Floyd-Warshall tightening), emptiness (negative cycle),
// intersection / single constraint (guards), delay/up (let time elapse), clock
// reset, and zone inclusion (for fixpoint termination). A pointwise `contains`
// membership test supports brute-force verification.
//
// Refs: Alur-Dill, "A theory of timed automata" (TCS 1994); Bengtsson-Yi (2004);
// the probabilistic-timed-automaton target is Kwiatkowska-Norman-Segala-Sproston
// (TCS 2002). Contracts: tests/unit/test_dbm.cpp.
// ============================================================================

#include <vector>
#include <limits>

namespace impact {
namespace dbm {

    // A bound "< c" (strict) or "<= c" (non-strict). INF = no constraint.
    struct Bound {
        long long c;
        bool strict;
        static constexpr long long INFC = (std::numeric_limits<long long>::max)() / 4;
        static Bound inf()   { return { INFC, false }; }
        static Bound leq(long long v) { return { v, false }; }
        static Bound lt(long long v)  { return { v, true  }; }
        bool isInf() const { return c >= INFC; }
        bool operator==(const Bound& o) const { return c == o.c && strict == o.strict; }
    };

    // (x_i - x_j) <(=) c. add: compose two bounds along a path. tighter: the smaller.
    Bound addB(const Bound& a, const Bound& b);
    bool  tighterB(const Bound& a, const Bound& b);   // a strictly tighter than b
    const Bound& minB(const Bound& a, const Bound& b);

    // A DBM over n clocks: (n+1) x (n+1) matrix, m[i][j] = bound on x_i - x_j.
    struct Zone {
        int n;                                   // number of clocks (1..n); index 0 = zero
        std::vector<std::vector<Bound>> m;
        explicit Zone(int clocks);               // the universe x_i>=0 (canonical)
    };

    void canonicalize(Zone& z);                  // Floyd-Warshall tightening
    bool isEmpty(const Zone& z);                 // negative cycle => empty zone

    // Add the constraint x_i - x_j (<=/<) c (a guard), then canonicalize.
    void constrain(Zone& z, int i, int j, Bound b);
    // Intersection of two zones (same clock count), canonicalized.
    Zone intersect(const Zone& a, const Zone& b);
    // Delay / future: let time elapse (drop upper bounds on each clock).
    void up(Zone& z);
    // Reset clock r (1..n) to 0.
    void reset(Zone& z, int r);
    // Inclusion: is `inner` a subset of `outer`? (both should be canonical / nonempty)
    bool includes(const Zone& outer, const Zone& inner);

    // Membership: does the clock valuation (size n, for clocks 1..n) lie in the zone?
    bool contains(const Zone& z, const std::vector<double>& val);

    // Classic maximal-constant extrapolation Extra_M (Behrmann-Bouyer-Larsen-Pelanek,
    // "Lower and upper bounds in zone-based abstractions of timed automata", 2006):
    // bounds above a clock's maximal constant are forgotten, guaranteeing finitely
    // many zones (termination of zone-graph reachability). kmax has size n+1 with
    // kmax[0]=0 and kmax[i] = the largest constant clock i is compared against.
    void extrapolate(Zone& z, const std::vector<long long>& kmax);

} // namespace dbm
} // namespace impact

#endif // IMPACT_DBM_H
