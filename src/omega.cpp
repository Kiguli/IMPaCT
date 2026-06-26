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

// ---- Robust almost-sure Büchi (pessimistic; ISSUE-0009) --------------------
// All qualitative computations work on the "may" graph: a transition (s,a)->t is
// possible iff hi>0 (nature can put positive mass on t). Inside a support EC every
// leaving edge has hi=0, so nature cannot LEAVE; the danger is nature ROUTING
// AROUND the accepting set via lo=0 edges. The robust value therefore needs the
// 2.5-player almost-sure-Büchi region, not just a support MEC.

// robustClosure(X): largest subset of X in which every state has at least one
// action whose may-support is fully inside X (controller can keep the play in X
// for ALL nature resolutions).
std::set<int> robustClosure(const solve::IMDPModel& m, std::set<int> X) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = X.begin(); it != X.end(); ) {
            bool hasStaying = false;
            for (const solve::ActionDist& act : m[*it]) {
                bool allIn = true;
                for (const solve::Interval& iv : act)
                    if (iv.hi > 0.0 && !X.count(iv.to)) { allIn = false; break; }
                if (allIn) { hasStaying = true; break; }
            }
            if (!hasStaying) { it = X.erase(it); changed = true; }
            else ++it;
        }
    }
    return X;
}

constexpr double TOL = 1e-9;

// natureCanContain(act, Z): can nature resolve this action's interval distribution
// with ALL probability mass inside Z? Feasible iff the in-Z successors can carry
// mass 1 (sum of their hi >= 1) and nature is not FORCED outside Z (sum of lo over
// out-of-Z successors == 0). If so, nature can keep the play in Z for sure via this
// action (lo=0 escaping edges are zeroed; the freed mass goes onto in-Z successors).
bool natureCanContain(const solve::ActionDist& act, const std::set<int>& Z) {
    double inHi = 0.0, outLo = 0.0;
    for (const solve::Interval& iv : act) {
        if (Z.count(iv.to)) inHi += iv.hi;
        else                outLo += iv.lo;
    }
    return inHi >= 1.0 - TOL && outLo <= TOL;
}

// withinX(act, X): all may-successors (hi>0) of the action lie in X.
bool withinX(const solve::ActionDist& act, const std::set<int>& X) {
    for (const solve::Interval& iv : act)
        if (iv.hi > 0.0 && !X.count(iv.to)) return false;
    return true;
}

// sureAvoid(X, T): greatest set Z ⊆ X\T such that for every z in Z and every
// within-X action of z, nature can contain the play in Z (natureCanContain). From
// Z, nature can keep the play off T FOREVER for sure, no matter what staying action
// the controller picks. (The robust analogue of the "sure-losing" region.)
std::set<int> sureAvoid(const solve::IMDPModel& m,
                        const std::set<int>& X, const std::set<int>& T) {
    std::set<int> Z;
    for (int s : X) if (!T.count(s)) Z.insert(s);
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = Z.begin(); it != Z.end(); ) {
            bool natureContainsAll = true;   // nature contains in Z for EVERY staying action
            for (const solve::ActionDist& act : m[*it]) {
                if (!withinX(act, X)) continue;          // controller wouldn't use it to stay
                if (!natureCanContain(act, Z)) { natureContainsAll = false; break; }
            }
            if (!natureContainsAll) { it = Z.erase(it); changed = true; }
            else ++it;
        }
    }
    return Z;
}

// natureAttractor(X, D): least set A ⊇ D such that for every s in X\D, if EVERY
// within-X action of s has a may-successor in A, then s ∈ A. From A, nature can
// force the play into D with positive probability against ANY controller strategy,
// so the controller cannot almost-surely avoid D.
std::set<int> natureAttractor(const solve::IMDPModel& m,
                              const std::set<int>& X, std::set<int> A) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int s : X) {
            if (A.count(s)) continue;
            bool natureForces = true;     // every staying action steps toward A
            bool anyStaying = false;
            for (const solve::ActionDist& act : m[s]) {
                if (!withinX(act, X)) continue;
                anyStaying = true;
                bool hasSuccInA = false;
                for (const solve::Interval& iv : act)
                    if (iv.hi > 0.0 && A.count(iv.to)) { hasSuccInA = true; break; }
                if (!hasSuccInA) { natureForces = false; break; }
            }
            if (anyStaying && natureForces) { A.insert(s); changed = true; }
        }
    }
    return A;
}

// optimisticClosure(X): greatest subset of X in which every state has an action
// whose interval distribution can be resolved with ALL mass inside the subset
// (natureCanContain) — i.e. cooperative nature can keep the play inside forever.
// (The optimistic analogue of robustClosure, which instead needs may-support ⊆ X.)
std::set<int> optimisticClosure(const solve::IMDPModel& m, std::set<int> X) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = X.begin(); it != X.end(); ) {
            bool hasStaying = false;
            for (const solve::ActionDist& act : m[*it])
                if (natureCanContain(act, X)) { hasStaying = true; break; }
            if (!hasStaying) { it = X.erase(it); changed = true; }
            else ++it;
        }
    }
    return X;
}

} // namespace

std::vector<int> robustBuchiWinningStates(const solve::IMDPModel& m,
                                          const std::set<int>& accepting) {
    std::set<int> X;
    for (std::size_t s = 0; s < m.size(); ++s) X.insert((int)s);
    while (!X.empty()) {
        X = robustClosure(m, X);                       // controller can stay in X
        std::set<int> T;
        for (int s : X) if (accepting.count(s)) T.insert(s);
        // States from which controller can NOT almost-surely reach T within X:
        // the sure-avoid region and everything nature can force into it.
        std::set<int> avoid = natureAttractor(m, X, sureAvoid(m, X, T));
        if (avoid.empty()) break;                      // all of X robustly a.s.-reaches T i.o.
        for (int s : avoid) X.erase(s);                // drop them, re-close, repeat
    }
    return std::vector<int>(X.begin(), X.end());
}

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
    // Robust value: reach the robust almost-sure-Büchi region (where the controller
    // forces visiting `accepting` i.o. for ALL nature), then it wins a.s. — so the
    // value is robust reachability of that region. (ISSUE-0009; replaces the unsound
    // support-MEC reduction, which over-counts ECs nature can route around.)
    std::vector<int> win = robustBuchiWinningStates(m, accepting);
    std::set<int> tgt(win.begin(), win.end());
    return solve::maxReachPessimistic(m, tgt, eps);
}

// Round-robin degeneralization of generalized Büchi into a single Büchi objective.
// Product state = s*k + c (c = which set we are currently waiting to see). The
// counter advances c -> (c+1) mod k when the current IMDP state is in F_c, and the
// product state (s, k-1) with s in F_{k-1} is the single Büchi-accepting class
// (one full round of all sets completed). With k=1 this is exactly the input model
// and accepting set, so maxGenBuchi* reduces to maxBuchi*.
namespace {
solve::IntervalResult maxGenBuchi(const solve::IMDPModel& m,
                                  const std::vector<std::set<int>>& F,
                                  double eps, bool optimistic) {
    const int n = (int)m.size();
    solve::IntervalResult out;
    out.lower.assign(n, 0.0);
    out.upper.assign(n, 0.0);
    out.iterations = 0;
    if (F.empty()) {                      // vacuously true: every infinite path accepts
        out.lower.assign(n, 1.0);
        out.upper.assign(n, 1.0);
        return out;
    }
    const int k = (int)F.size();
    auto idx = [&](int s, int c) { return s * k + c; };
    solve::IMDPModel pm((std::size_t)n * k);
    std::set<int> acc;
    for (int s = 0; s < n; ++s) {
        for (int c = 0; c < k; ++c) {
            const bool inFc = F[c].count(s) > 0;
            const int cadv = inFc ? (c + 1) % k : c;
            for (const solve::ActionDist& act : m[s]) {
                solve::ActionDist pa;
                pa.reserve(act.size());
                for (const solve::Interval& iv : act)
                    pa.push_back({idx(iv.to, cadv), iv.lo, iv.hi});
                pm[idx(s, c)].push_back(std::move(pa));
            }
            if (inFc && c == k - 1) acc.insert(idx(s, c));   // completed a full round
        }
    }
    solve::IntervalResult r = optimistic ? maxBuchiOptimistic(pm, acc, eps)
                                         : maxBuchiPessimistic(pm, acc, eps);
    out.iterations = r.iterations;
    for (int s = 0; s < n; ++s) { out.lower[s] = r.lower[idx(s, 0)]; out.upper[s] = r.upper[idx(s, 0)]; }
    return out;
}
} // namespace

solve::IntervalResult maxGenBuchiOptimistic(const solve::IMDPModel& m,
                              const std::vector<std::set<int>>& accSets, double eps) {
    return maxGenBuchi(m, accSets, eps, /*optimistic=*/true);
}
solve::IntervalResult maxGenBuchiPessimistic(const solve::IMDPModel& m,
                              const std::vector<std::set<int>>& accSets, double eps) {
    return maxGenBuchi(m, accSets, eps, /*optimistic=*/false);
}

namespace {
solve::IntervalResult zeros(std::size_t n) {
    solve::IntervalResult r; r.lower.assign(n, 0.0); r.upper.assign(n, 0.0); r.iterations = 0;
    return r;
}
} // namespace

solve::IntervalResult maxPersistencePessimistic(const solve::IMDPModel& m,
                              const std::set<int>& pStates, double eps) {
    // Largest sub-region of p the controller can stay in forever for ALL nature,
    // then reach it robustly. (F G p = reach-then-stay.)
    std::set<int> W = robustClosure(m, pStates);
    if (W.empty()) return zeros(m.size());          // F G p unsatisfiable -> 0
    return solve::maxReachPessimistic(m, W, eps);
}

solve::IntervalResult maxPersistenceOptimistic(const solve::IMDPModel& m,
                              const std::set<int>& pStates, double eps) {
    std::set<int> W = optimisticClosure(m, pStates);
    if (W.empty()) return zeros(m.size());
    return solve::maxReachOptimistic(m, W, eps);
}

} // namespace omega
} // namespace impact
