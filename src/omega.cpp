#include "omega.h"
#include "graph_utils.h"

#include <algorithm>

namespace impact {
namespace omega {

namespace {

// Support graph of the IMDP as an MDPGraph: g[s][a] = successors with hi > 0.
graph::MDPGraph supportGraph(const solve::IMDPModel& m) {
    graph::MDPGraph g(m.size());
    for (std::size_t s = 0; s < m.size(); ++s) {
        for (const solve::ActionDist& act : m[s]) {
            std::vector<int> succ;
            for (const solve::Interval& iv : act) if (iv.hi > 0.0) succ.push_back(iv.to);
            if (!succ.empty()) g[s].push_back(std::move(succ));
        }
    }
    return g;
}

} // namespace

std::vector<int> acceptingMECStates(const solve::IMDPModel& m, const std::set<int>& accepting) {
    const graph::MDPGraph g = supportGraph(m);
    const std::vector<std::vector<int>> mecs = graph::mecs(g);
    std::vector<int> out;
    for (const std::vector<int>& C : mecs) {
        bool good = false;
        for (int s : C) if (accepting.count(s)) { good = true; break; }
        if (good) out.insert(out.end(), C.begin(), C.end());
    }
    std::sort(out.begin(), out.end());
    return out;
}

solve::IntervalResult maxBuchiOptimistic(const solve::IMDPModel& m,
                                         const std::set<int>& accepting, double eps) {
    std::vector<int> good = acceptingMECStates(m, accepting);
    std::set<int> tgt(good.begin(), good.end());
    return solve::maxReachOptimistic(m, tgt, eps);
}

solve::IntervalResult maxBuchiPessimistic(const solve::IMDPModel& m,
                                          const std::set<int>& accepting, double eps) {
    std::vector<int> good = acceptingMECStates(m, accepting);
    std::set<int> tgt(good.begin(), good.end());
    return solve::maxReachPessimistic(m, tgt, eps);
}

} // namespace omega
} // namespace impact
