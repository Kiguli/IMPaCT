#include "solve.h"
#include "omaximization.h"
#include "graph_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <set>

namespace impact {
namespace solve {

namespace {

// Robust Bellman backup at one state: max over actions of the omax optimum.
// sense = Min for pessimistic (adversarial nature), Max for optimistic.
// A state with no actions has value 0 (cannot make progress to a target).
// controllerMax: the controller MAXIMIZES over actions (reachability) or MINIMIZES
// (used for safety = 1 - min-reach-to-avoid). A state with no actions has value 0.
double backup(const StateActions& actions, const std::vector<double>& V, omax::Sense sense,
              bool controllerMax = true) {
    double best = 0.0;
    bool any = false;
    std::vector<double> lo, hi, vv;
    for (const ActionDist& act : actions) {
        lo.clear(); hi.clear(); vv.clear();
        lo.reserve(act.size()); hi.reserve(act.size()); vv.reserve(act.size());
        for (const Interval& iv : act) { lo.push_back(iv.lo); hi.push_back(iv.hi); vv.push_back(V[iv.to]); }
        const double val = omax::optimize(lo, hi, vv, sense).value;
        if (!any) { best = val; any = true; }
        else best = controllerMax ? std::max(best, val) : std::min(best, val);
    }
    return any ? best : 0.0;
}

struct Collapsed {
    IMDPModel model;
    std::vector<int> origToCollapsed;
    std::set<int> targets;
};

// Collapse maximal end components so the Bellman operator has a unique fixpoint.
// Within a MEC every state has the same max-reach value; we merge each MEC into one
// representative, drop within-MEC "staying" actions, and keep/leaving actions with
// successors remapped to representatives. A pure-trap MEC (no leaving action) becomes
// an actionless state (value 0). Targets are kept as-is (value pinned to 1 elsewhere).
Collapsed collapseMECs(const IMDPModel& m, const std::set<int>& targets) {
    const int n = (int)m.size();

    // Support MDP graph (edge present iff some action puts hi>0 on the successor);
    // targets get no actions so they never sit inside a non-trivial MEC.
    graph::MDPGraph g(n);
    for (int s = 0; s < n; ++s) {
        if (targets.count(s)) continue;
        for (const ActionDist& act : m[s]) {
            std::vector<int> succ;
            for (const Interval& iv : act) if (iv.hi > 0.0) succ.push_back(iv.to);
            if (!succ.empty()) g[s].push_back(std::move(succ));
        }
    }

    const std::vector<std::vector<int>> mecList = graph::mecs(g);

    std::vector<int> repOf(n);
    for (int s = 0; s < n; ++s) repOf[s] = s;
    std::vector<char> inMec(n, 0);
    for (const auto& C : mecList) {
        const int r = C.front();                 // canonical rep = min state (C is sorted)
        for (int s : C) { repOf[s] = r; inMec[s] = 1; }
    }

    std::vector<int> compact(n, -1);
    int nc = 0;
    for (int s = 0; s < n; ++s) if (repOf[s] == s) compact[s] = nc++;
    std::vector<int> origToCollapsed(n);
    for (int s = 0; s < n; ++s) origToCollapsed[s] = compact[repOf[s]];

    IMDPModel M(nc);
    std::set<int> ctargets;
    for (int t : targets) ctargets.insert(origToCollapsed[t]);

    for (int s = 0; s < n; ++s) {
        if (targets.count(s)) continue;
        const int c = origToCollapsed[s];
        for (const ActionDist& act : m[s]) {
            ActionDist remapped;
            remapped.reserve(act.size());
            bool leaves = false;
            for (const Interval& iv : act) {
                const int ct = origToCollapsed[iv.to];
                remapped.push_back(Interval{ct, iv.lo, iv.hi});
                if (ct != c) leaves = true;
            }
            if (inMec[s] && !leaves) continue;   // within-MEC staying action: drop
            M[c].push_back(std::move(remapped));
        }
    }

    return Collapsed{std::move(M), std::move(origToCollapsed), std::move(ctargets)};
}

// Two-sided interval iteration on an (end-component-free) model: lower from 0,
// upper from 1, robust Bellman applied to both, stop when the max gap <= 2*eps.
IntervalResult intervalIterate(const IMDPModel& M, const std::set<int>& targets,
                               double eps, omax::Sense sense) {
    const int n = (int)M.size();
    std::vector<double> L(n, 0.0), U(n, 1.0);
    for (int t : targets) { L[t] = 1.0; U[t] = 1.0; }

    const int MAXIT = 2000000;
    int iters = 0;
    std::vector<double> nL(n), nU(n);
    while (iters < MAXIT) {
        ++iters;
        for (int s = 0; s < n; ++s) {
            if (targets.count(s)) { nL[s] = 1.0; nU[s] = 1.0; continue; }
            nL[s] = backup(M[s], L, sense);
            nU[s] = backup(M[s], U, sense);
        }
        L.swap(nL); U.swap(nU);
        double gap = 0.0;
        for (int s = 0; s < n; ++s) gap = std::max(gap, U[s] - L[s]);
        if (gap <= 2.0 * eps) break;
    }
    return IntervalResult{L, U, iters};
}

IntervalResult solveReach(const IMDPModel& m, const std::set<int>& targets,
                          double eps, omax::Sense sense) {
    const Collapsed col = collapseMECs(m, targets);
    const IntervalResult cr = intervalIterate(col.model, col.targets, eps, sense);
    const int n = (int)m.size();
    IntervalResult r;
    r.lower.resize(n); r.upper.resize(n); r.iterations = cr.iterations;
    for (int s = 0; s < n; ++s) {
        const int c = col.origToCollapsed[s];
        r.lower[s] = cr.lower[c];
        r.upper[s] = cr.upper[c];
    }
    return r;
}

// Optimistic value iteration (Hartmanns & Kaminski, CAV 2020): VI-from-below gives
// a sound lower bound L (<= V*); then guess U = min(1, L+eps) and VERIFY it is a
// pre-fixpoint (F(U) <= U). By Knaster-Tarski, F(U) <= U implies V* (the least
// fixpoint) <= U, so [L,U] is sound with gap <= eps. No end-component handling
// needed, so it converges on nature-confinable ECs (ISSUE-0003). If the guess is
// not yet inductive, refine L (smaller delta) and retry.
IntervalResult solveOVI(const IMDPModel& m, const std::set<int>& targets,
                        double eps, omax::Sense sense, bool controllerMax = true) {
    const int n = (int)m.size();
    std::vector<double> L(n, 0.0), tmp(n), U(n), FU(n);
    for (int t : targets) L[t] = 1.0;
    int iters = 0;
    double delta = eps;
    const int MAXROUND = 100, MAXINNER = 2000000;
    for (int round = 0; round < MAXROUND; ++round) {
        for (int it = 0; it < MAXINNER; ++it) {              // refine L from below
            ++iters;
            for (int s = 0; s < n; ++s)
                tmp[s] = targets.count(s) ? 1.0 : backup(m[s], L, sense, controllerMax);
            double ch = 0.0;
            for (int s = 0; s < n; ++s) ch = std::max(ch, std::fabs(tmp[s] - L[s]));
            L.swap(tmp);
            if (ch < delta) break;
        }
        for (int s = 0; s < n; ++s)
            U[s] = targets.count(s) ? 1.0 : std::min(1.0, L[s] + eps);
        bool inductive = true;                                // verify F(U) <= U
        for (int s = 0; s < n; ++s) {
            FU[s] = targets.count(s) ? 1.0 : backup(m[s], U, sense, controllerMax);
            if (FU[s] > U[s] + 1e-12) inductive = false;
        }
        if (inductive) return IntervalResult{L, U, iters};    // L <= V* <= U, gap <= eps
        delta *= 0.5;
    }
    for (int s = 0; s < n; ++s) U[s] = targets.count(s) ? 1.0 : std::min(1.0, L[s] + eps);
    return IntervalResult{L, U, iters};                       // best-effort fallback
}

// Min-reach: controller MINIMIZES reach to `targets`. Used for safety.
IntervalResult minReach(const IMDPModel& m, const std::set<int>& targets,
                        double eps, omax::Sense sense) {
    return solveOVI(m, targets, eps, sense, /*controllerMax=*/false);
}

// Safety = 1 - min-reach-to-avoid (controller maximizes staying out of `avoid`).
IntervalResult safetyFromMinReach(const IMDPModel& m, const std::set<int>& avoid,
                                  double eps, omax::Sense sense) {
    IntervalResult mr = minReach(m, avoid, eps, sense);
    IntervalResult r;
    r.iterations = mr.iterations;
    r.lower.resize(mr.lower.size()); r.upper.resize(mr.upper.size());
    for (size_t s = 0; s < mr.lower.size(); ++s) {
        r.lower[s] = 1.0 - mr.upper[s];    // safety lower = 1 - max possible reach
        r.upper[s] = 1.0 - mr.lower[s];
    }
    return r;
}

IntervalResult dispatch(const IMDPModel& m, const std::set<int>& targets,
                        double eps, omax::Sense sense, Method method) {
    return (method == Method::MECCollapse) ? solveReach(m, targets, eps, sense)
                                           : solveOVI(m, targets, eps, sense);
}

} // namespace

IntervalResult maxReachPessimistic(const IMDPModel& m, const std::set<int>& targets, double eps) {
    return dispatch(m, targets, eps, omax::Sense::Min, Method::OptimisticVI);
}
IntervalResult maxReachOptimistic(const IMDPModel& m, const std::set<int>& targets, double eps) {
    return dispatch(m, targets, eps, omax::Sense::Max, Method::OptimisticVI);
}
IntervalResult maxReachPessimistic(const IMDPModel& m, const std::set<int>& targets, double eps, Method method) {
    return dispatch(m, targets, eps, omax::Sense::Min, method);
}
IntervalResult maxReachOptimistic(const IMDPModel& m, const std::set<int>& targets, double eps, Method method) {
    return dispatch(m, targets, eps, omax::Sense::Max, method);
}

// Robust safety: max over controller of P(never reach `avoid`); pessimistic =
// nature adversarial (maximizes reach to avoid, omax Sense::Max).
IntervalResult maxSafetyPessimistic(const IMDPModel& m, const std::set<int>& avoid, double eps) {
    return safetyFromMinReach(m, avoid, eps, omax::Sense::Max);
}
IntervalResult maxSafetyOptimistic(const IMDPModel& m, const std::set<int>& avoid, double eps) {
    return safetyFromMinReach(m, avoid, eps, omax::Sense::Min);
}

} // namespace solve
} // namespace impact
