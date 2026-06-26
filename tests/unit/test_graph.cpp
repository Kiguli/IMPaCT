// ============================================================================
// CONTRACT TESTS — Phase 1b: SCC + Maximal End Component decomposition.
// IMMUTABLE hand-derived expectations. See tests/TEST_PLAN.md §1b for the
// derivation of each expected component set.
// ============================================================================
#include "../doctest.h"
#include "../contracts/contracts.h"

#include <vector>
#include <algorithm>
#include <random>

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

TEST_CASE("mec: the leaking-action example that breaks the NAIVE rule (true MECs {0},{2})") {
    // 0: a->{1,2} (leaks), b->{0} (self) ; 1: a->{0} ; 2: a->{2} (self).
    // {0,1} is an SCC of the full staying-action graph (edge 0->1 via a) and every
    // state of {0,1} has *some* action staying in {0,1} (0 has b, 1 has a). A naive
    // "accept SCC if no state lacks a staying action" would WRONGLY return {0,1}.
    // {0,1} is NOT an end component: 0's only {0,1}-staying action is the self-loop b,
    // so 0 cannot reach 1 inside {0,1}. Correct MECs: {0} and {2}.
    MDPGraph g = { {{1,2},{0}}, {{0}}, {{2}} };
    auto got = canon(mecs(g));
    vector<vector<int>> want = {{0}, {2}};
    CHECK(got == want);
}

// ---------------------------------------------------------------------------
// Definition-based brute-force MEC oracle (independent of impact::graph::mecs).
// T is an end component iff (a) every state in T has at least one action whose
// successors are all in T, and (b) the digraph on T using edges from ALL such
// T-staying actions is strongly connected. MECs are the inclusion-maximal ECs.
// This is the EC definition transcribed directly; enumerating subsets gives a
// ground-truth oracle for small MDPs, used for randomized differential testing.
// ---------------------------------------------------------------------------
namespace {

using impact::graph::MDPGraph;

bool succ_subset_of(const std::vector<int>& succ, unsigned Tmask) {
    for (int t : succ) if (!((Tmask >> t) & 1u)) return false;
    return true;
}

bool strongly_connected_within(unsigned Tmask, int n,
                               const std::vector<std::vector<int>>& adj) {
    std::vector<int> nodes;
    for (int i = 0; i < n; ++i) if ((Tmask >> i) & 1u) nodes.push_back(i);
    if (nodes.size() <= 1) return true;
    auto reach_count = [&](int start, const std::vector<std::vector<int>>& gg) {
        std::vector<char> vis(n, 0);
        std::vector<int> st{start};
        vis[start] = 1; int c = 1;
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            for (int v : gg[u]) if (((Tmask >> v) & 1u) && !vis[v]) { vis[v] = 1; ++c; st.push_back(v); }
        }
        return c;
    };
    if (reach_count(nodes[0], adj) != (int)nodes.size()) return false;  // fwd
    std::vector<std::vector<int>> radj(n);
    for (int u = 0; u < n; ++u) if ((Tmask >> u) & 1u)
        for (int v : adj[u]) if ((Tmask >> v) & 1u) radj[v].push_back(u);
    return reach_count(nodes[0], radj) == (int)nodes.size();          // bwd
}

bool is_end_component(const MDPGraph& g, unsigned Tmask, int n) {
    std::vector<std::vector<int>> adj(n);
    for (int s = 0; s < n; ++s) {
        if (!((Tmask >> s) & 1u)) continue;
        bool has_staying = false;
        for (const auto& succ : g[s]) {
            if (succ_subset_of(succ, Tmask)) {
                has_staying = true;
                for (int t : succ) adj[s].push_back(t);
            }
        }
        if (!has_staying) return false;                  // (a) closure impossible
    }
    return strongly_connected_within(Tmask, n, adj);      // (b)
}

std::vector<std::vector<int>> brute_mecs(const MDPGraph& g) {
    const int n = (int)g.size();
    std::vector<unsigned> ecs;
    for (unsigned T = 1; T < (1u << n); ++T)
        if (is_end_component(g, T, n)) ecs.push_back(T);
    std::vector<std::vector<int>> res;
    for (unsigned T : ecs) {
        bool maximal = true;
        for (unsigned U : ecs) if (U != T && (T & U) == T) { maximal = false; break; } // T ⊊ U
        if (maximal) {
            std::vector<int> v;
            for (int i = 0; i < n; ++i) if ((T >> i) & 1u) v.push_back(i);
            res.push_back(std::move(v));
        }
    }
    std::sort(res.begin(), res.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) { return a.front() < b.front(); });
    return res;
}

} // namespace

TEST_CASE("mec: randomized differential vs brute-force definition-based oracle") {
    std::mt19937 rng(2024);                         // fixed seed -> deterministic
    std::uniform_int_distribution<int> ndist(1, 6);
    std::uniform_int_distribution<int> adist(1, 3); // actions per state
    int checked = 0;
    for (int trial = 0; trial < 6000 && checked < 1500; ++trial) {
        int n = ndist(rng);
        MDPGraph g(n);
        for (int s = 0; s < n; ++s) {
            int na = adist(rng);
            for (int a = 0; a < na; ++a) {
                std::vector<int> succ;
                for (int t = 0; t < n; ++t) if (rng() & 1u) succ.push_back(t);
                if (succ.empty()) succ.push_back((int)(rng() % n));  // nonempty support
                g[s].push_back(succ);
            }
        }
        ++checked;
        CHECK(canon(mecs(g)) == brute_mecs(g));
    }
    CHECK(checked > 500);
}
