#include "graph_utils.h"

#include <algorithm>
#include <numeric>

namespace impact {
namespace graph {

namespace {

// Iterative Tarjan SCC over the nodes for which active[v] is true. Edges to
// inactive nodes are skipped. Returns components, each sorted ascending, the
// outer vector sorted by component minimum. Iterative (explicit stack) so it is
// safe on large graphs.
std::vector<std::vector<int>> tarjan(const AdjList& succ,
                                     const std::vector<char>& active) {
    const int n = static_cast<int>(succ.size());
    std::vector<int> index(n, -1), low(n, 0);
    std::vector<char> onstack(n, 0);
    std::vector<std::size_t> it(n, 0);   // per-node child pointer
    std::vector<int> tstack;             // Tarjan stack
    std::vector<int> call;               // explicit DFS stack
    int idx = 0;
    std::vector<std::vector<int>> comps;

    for (int root = 0; root < n; ++root) {
        if (!active[root] || index[root] != -1) continue;
        index[root] = low[root] = idx++;
        onstack[root] = 1;
        tstack.push_back(root);
        call.push_back(root);

        while (!call.empty()) {
            const int v = call.back();
            if (it[v] < succ[v].size()) {
                const int w = succ[v][it[v]++];
                if (!active[w]) continue;
                if (index[w] == -1) {
                    index[w] = low[w] = idx++;
                    onstack[w] = 1;
                    tstack.push_back(w);
                    call.push_back(w);
                } else if (onstack[w]) {
                    low[v] = std::min(low[v], index[w]);
                }
            } else {
                if (low[v] == index[v]) {       // v is an SCC root
                    std::vector<int> comp;
                    while (true) {
                        const int w = tstack.back();
                        tstack.pop_back();
                        onstack[w] = 0;
                        comp.push_back(w);
                        if (w == v) break;
                    }
                    std::sort(comp.begin(), comp.end());
                    comps.push_back(std::move(comp));
                }
                call.pop_back();
                if (!call.empty()) {
                    const int parent = call.back();
                    low[parent] = std::min(low[parent], low[v]);
                }
            }
        }
    }

    std::sort(comps.begin(), comps.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
                  return a.front() < b.front();
              });
    return comps;
}

} // namespace

std::vector<std::vector<int>> sccs(const AdjList& succ) {
    std::vector<char> active(succ.size(), 1);
    return tarjan(succ, active);
}

std::vector<std::vector<int>> mecs(const MDPGraph& g) {
    const int n = static_cast<int>(g.size());
    std::vector<std::vector<int>> result;

    // Worklist of candidate state-sets. Start from the full state space.
    std::vector<std::vector<int>> work;
    {
        std::vector<int> full(n);
        std::iota(full.begin(), full.end(), 0);
        work.push_back(std::move(full));
    }

    std::vector<char> inW(n, 0);
    while (!work.empty()) {
        const std::vector<int> W = std::move(work.back());
        work.pop_back();

        for (int s : W) inW[s] = 1;

        // Build the induced graph on W using only actions that STAY in W
        // (every successor of the action is in W).
        AdjList adj(n);
        for (int s : W) {
            for (const std::vector<int>& action_succ : g[s]) {
                bool stays = true;
                for (int t : action_succ) if (!inW[t]) { stays = false; break; }
                if (!stays) continue;
                for (int t : action_succ) adj[s].push_back(t);
            }
        }

        const std::vector<std::vector<int>> comps = tarjan(adj, inW);

        if (comps.size() == 1) {
            // W is a single SCC under its own staying actions.
            const std::vector<int>& C = comps.front();
            if (C.size() > 1) {
                result.push_back(C);                 // genuine multi-state EC
            } else {
                const int s = C.front();             // singleton: EC iff self-loop
                bool self_loop = std::find(adj[s].begin(), adj[s].end(), s) != adj[s].end();
                if (self_loop) result.push_back(C);
            }
        } else {
            // Not strongly connected under staying actions: refine into SCCs.
            for (const std::vector<int>& C : comps) work.push_back(C);
        }

        for (int s : W) inW[s] = 0;
    }

    std::sort(result.begin(), result.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
                  return a.front() < b.front();
              });
    return result;
}

} // namespace graph
} // namespace impact
