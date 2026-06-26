// ============================================================================
// CONTRACT TESTS — Phase 1b: SCC + Maximal End Component decomposition.
// IMMUTABLE hand-derived expectations. See tests/TEST_PLAN.md §1b for the
// derivation of each expected component set.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>
#include <algorithm>

using impact::graph::sccs;
using impact::graph::mecs;
using impact::graph::AdjList;
using impact::graph::MDPGraph;
using std::vector;

// Canonicalize: sort each component, sort components by their first element.
static vector<vector<int>> canon(vector<vector<int>> comps) {
    for (auto& c : comps) std::sort(c.begin(), c.end());
    std::sort(comps.begin(), comps.end(),
              [](const vector<int>& a, const vector<int>& b) {
                  return a.front() < b.front();
              });
    return comps;
}

TEST_CASE("scc: two cycles linked by a cross edge") {
    // 0->1->2->0  ; 3->4->3 ; 2->3 ; 4->5 ; 5 sink
    AdjList g = {
        /*0*/ {1},
        /*1*/ {2},
        /*2*/ {0, 3},
        /*3*/ {4},
        /*4*/ {3, 5},
        /*5*/ {},
    };
    auto got = canon(sccs(g));
    vector<vector<int>> want = {{0, 1, 2}, {3, 4}, {5}};
    CHECK(got == want);
}

TEST_CASE("scc: a DAG has only singleton components") {
    AdjList g = { {1}, {2}, {} };
    auto got = canon(sccs(g));
    vector<vector<int>> want = {{0}, {1}, {2}};
    CHECK(got == want);
}

TEST_CASE("scc: self-loop is its own component") {
    AdjList g = { {0} };
    auto got = canon(sccs(g));
    vector<vector<int>> want = {{0}};
    CHECK(got == want);
}

TEST_CASE("mec: choose the staying action to form end component {1,2}, plus self-loop {4}") {
    // g[s][a] = successors of action a in state s
    // 0: a0->1                 (transient: only leaves)
    // 1: a0->2                 (stays toward {1,2})
    // 2: a0->1 ; a1->3         (can loop back to 1, or leave to 3)
    // 3: a0->4                 (transient)
    // 4: a0->4                 (self-loop end component)
    MDPGraph g = {
        /*0*/ {{1}},
        /*1*/ {{2}},
        /*2*/ {{1}, {3}},
        /*3*/ {{4}},
        /*4*/ {{4}},
    };
    auto got = canon(mecs(g));
    vector<vector<int>> want = {{1, 2}, {4}};
    CHECK(got == want);
}

TEST_CASE("mec: a deterministic 2-cycle is a single MEC") {
    // 0: a0->1 ; 1: a0->0   => end component {0,1}
    MDPGraph g = { {{1}}, {{0}} };
    auto got = canon(mecs(g));
    vector<vector<int>> want = {{0, 1}};
    CHECK(got == want);
}

TEST_CASE("mec: a state whose only action leaves is not an end component on its own") {
    // 0: a0->1 ; 1: a0->1(self-loop)  => only MEC is {1}; 0 is transient.
    MDPGraph g = { {{1}}, {{1}} };
    auto got = canon(mecs(g));
    vector<vector<int>> want = {{1}};
    CHECK(got == want);
}
