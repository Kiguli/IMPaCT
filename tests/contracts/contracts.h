#ifndef IMPACT_TEST_CONTRACTS_H
#define IMPACT_TEST_CONTRACTS_H

// ============================================================================
// IMPaCT v2.0 — TEST CONTRACTS (interfaces the implementation must satisfy)
// ----------------------------------------------------------------------------
// These declarations are the *target API* for the new IMPaCT v2.0 components.
// The TDD test suite is written against THESE signatures. Implementation work
// fills in the corresponding src/*.cpp (replacing tests/contracts/stubs.cpp).
//
// RULE: tests are immutable contracts. We change the implementation to satisfy
// the tests, never the other way around. If an interface here must change, it
// is a deliberate design change reviewed against the test plan — not a hack to
// make a failing test pass.
// ============================================================================

#include <vector>
#include <set>
#include <string>
#include <cstddef>

namespace impact {

// ---------------------------------------------------------------------------
// Phase 1a — O-maximization (robust inner Bellman solve over a box-interval
// ambiguity set). Replaces the per-state GLPK LP.
// Ref: Givan-Leach-Dean (AIJ 2000); Lahijanian-Andersson-Belta (IEEE TAC 2015).
// ---------------------------------------------------------------------------
namespace omax {

    enum class Sense { Min, Max };

    struct Result {
        std::vector<double> p;  // optimal feasible distribution over successors
        double value;           // dot(p, V)
    };

    // Given per-successor lower bounds `lower`, upper bounds `upper`, and a value
    // vector `V` (all equal length n >= 1), return the distribution p in the box
    // [lower,upper] with sum(p)=1 that minimizes (Sense::Min) or maximizes
    // (Sense::Max) dot(p,V), and that optimal value.
    //
    // CONTRACT:
    //   * p[i] in [lower[i], upper[i]] for all i (within 1e-9)
    //   * sum(p) == 1 (within 1e-9)
    //   * value == dot(p,V) is the true optimum (matches brute-force vertex enum)
    //   * THROWS std::invalid_argument if sizes differ, n==0, any lower>upper,
    //     sum(lower) > 1 + 1e-12, or sum(upper) < 1 - 1e-12 (infeasible box).
    Result optimize(const std::vector<double>& lower,
                    const std::vector<double>& upper,
                    const std::vector<double>& V,
                    Sense sense);

} // namespace omax

// ---------------------------------------------------------------------------
// Phase 1b — Graph algorithms: SCC + Maximal End Component decomposition.
// Refs: Tarjan (1972) SCC; de Alfaro (1997), Chatterjee-Henzinger (JACM 2014) MEC.
// ---------------------------------------------------------------------------
namespace graph {

    // Plain directed graph as adjacency list: succ[u] = list of v with edge u->v.
    using AdjList = std::vector<std::vector<int>>;

    // Return the strongly connected components. Each component is a node list,
    // sorted ascending; the outer vector is sorted by each component's min node.
    // (Canonical ordering so tests can compare deterministically.)
    std::vector<std::vector<int>> sccs(const AdjList& succ);

    // MDP graph: g[s][a] = list of possible successor states of action a in s
    // (the "support" graph — a transition is present iff its upper bound > 0).
    using MDPGraph = std::vector<std::vector<std::vector<int>>>;

    // Return the maximal end components. Each MEC is a sorted state list; outer
    // vector sorted by min state. Singletons with no self-sustaining action are
    // NOT returned (only genuine end components: a sub-MDP closed under some
    // action choice whose induced graph is strongly connected).
    std::vector<std::vector<int>> mecs(const MDPGraph& g);

} // namespace graph

// ---------------------------------------------------------------------------
// Phase 1c — Sound robust interval iteration for reachability on an IMDP.
// Refs: Haddad-Monmege (TCS 2018, IMDP soundness); Baier et al. (CAV 2017).
// ---------------------------------------------------------------------------
namespace solve {

    // One successor of an action, with a probability interval [lo,hi].
    struct Interval { int to; double lo; double hi; };
    using ActionDist  = std::vector<Interval>;       // one prob interval per successor
    using StateActions = std::vector<ActionDist>;    // actions available at a state
    using IMDPModel   = std::vector<StateActions>;   // per-state action lists

    struct IntervalResult {
        std::vector<double> lower;  // sound lower bound on the value, per state
        std::vector<double> upper;  // sound upper bound on the value, per state
        int iterations;
    };

    // Robust max-reachability of `targets`: controller MAXIMIZES, nature picks
    // transition probabilities within the intervals adversarially to MINIMIZE
    // (worst-case / pessimistic). Returns sound [lower,upper] with
    // upper[s]-lower[s] <= 2*eps for every state s.
    //
    // CONTRACT:
    //   * lower[s] <= V*(s) <= upper[s] for the true robust value V* (soundness)
    //   * target states have value 1; states that cannot reach a target under
    //     ALL adversary choices have value 0 (requires Prob0/Prob1 + MEC handling)
    //   * gap <= 2*eps at termination
    IntervalResult maxReachPessimistic(const IMDPModel& m,
                                       const std::set<int>& targets,
                                       double eps);

    // Best-case variant: controller maximizes, nature maximizes too (optimistic).
    IntervalResult maxReachOptimistic(const IMDPModel& m,
                                      const std::set<int>& targets,
                                      double eps);

} // namespace solve

// ---------------------------------------------------------------------------
// Phase 2 — LTL/LTLf -> automaton front-end. Tested via language membership so
// the contract is independent of the chosen back-end library (Spot/Owl/Lydia).
// Refs: Kupferman-Vardi (FMSD 2001); De Giacomo-Vardi (IJCAI 2013/2015).
// ---------------------------------------------------------------------------
namespace ltl {

    // A finite trace: each element is the set of atomic propositions true at that
    // step (subset of the formula's APs), given by name.
    using Letter = std::set<std::string>;
    using FiniteTrace = std::vector<Letter>;

    // Opaque handle to a compiled (deterministic, finite) automaton for an
    // LTLf / co-safe-LTL formula over the given alphabet of AP names.
    struct Automaton;  // defined by the implementation

    // Compile an LTLf / co-safe LTL formula to a DFA. THROWS std::invalid_argument
    // on parse error or if the formula is not co-safe (when require_cosafe).
    Automaton* compileFinite(const std::string& formula,
                             const std::vector<std::string>& aps);

    // Does the compiled automaton accept this finite trace? (language membership)
    bool acceptsFinite(const Automaton* a, const FiniteTrace& trace);

    void destroy(Automaton* a);

} // namespace ltl

} // namespace impact

#endif // IMPACT_TEST_CONTRACTS_H
