#ifndef IMPACT_GRAPH_UTILS_H
#define IMPACT_GRAPH_UTILS_H

// ============================================================================
// Graph algorithms for IMPaCT v2.0: strongly connected components (Tarjan) and
// maximal end-component (MEC) decomposition. Needed for sound interval
// iteration (Prob0/Prob1 + MEC collapse, Phase 1c) and reused by the ω-regular
// accepting-EC analysis (Phase 3).
//
// Refs: Tarjan (1972) SCC; de Alfaro (1997), Chatterjee-Henzinger (JACM 2014) MEC.
// Contracts: tests/unit/test_graph.cpp.
// ============================================================================

#include <vector>

namespace impact {
namespace graph {

    // Directed graph as adjacency list: succ[u] = list of v with edge u->v.
    using AdjList = std::vector<std::vector<int>>;

    // Strongly connected components. Each component is sorted ascending; the
    // outer vector is sorted by each component's minimum node (canonical form).
    std::vector<std::vector<int>> sccs(const AdjList& succ);

    // MDP graph: g[s][a] = list of possible successor states of action a in s
    // (the support graph — a transition is present iff its upper bound > 0).
    using MDPGraph = std::vector<std::vector<std::vector<int>>>;

    // Maximal end components. Each MEC is a sorted state list; outer vector
    // sorted by min state. A singleton {s} is returned only if s has an action
    // whose successors are exactly {s} (a self-sustaining loop); transient
    // states are excluded.
    std::vector<std::vector<int>> mecs(const MDPGraph& g);

} // namespace graph
} // namespace impact

#endif // IMPACT_GRAPH_UTILS_H
