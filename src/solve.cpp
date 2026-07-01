#include "solve.h"
#include "omaximization.h"
#include "graph_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <unordered_map>

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

// ---- Robust expected reward (robust dynamic programming; Iyengar 2005) -------------
// One VI on the reward Bellman operator T V (s) = reward[s] + gamma * opt_a opt_p sum p V,
// with the interval ambiguity set resolved in closed form by O-maximization. For the
// reachability objective gamma=1 and the target set is absorbing (T V(target)=0). For the
// discounted objective gamma in (0,1) and there is no target. Rewards are assumed >= 0 so
// the from-0 iterate is monotone non-decreasing; if it fails to converge (target not
// reached a.s., gamma==1) the value diverges and we report HUGE_VAL (= PRISM/Storm "inf").
static IntervalResult rewardVI(const IMDPModel& m, const std::set<int>& targets,
                               const std::vector<double>& reward, double gamma, double eps,
                               omax::Sense nature, bool controllerMax) {
    const int n = (int)m.size();
    std::vector<double> V(n, 0.0), Vn(n);
    const int MAXIT = 2000000;
    // Finite cap: states that never robustly reach the target diverge (Puterman SSP:
    // only proper policies give a finite value). We clamp each iterate at CAP so the
    // improper states pin there (monotone, non-decreasing for reward>=0) instead of
    // overflowing to +inf — a FINITE cap keeps omax's 0*CAP = 0 (no 0*inf = NaN), while
    // proper states converge well below CAP. Values >= CAP*0.9 are reported as +inf.
    const double CAP = 1e18;
    int iters = 0;
    while (iters < MAXIT) {
        ++iters;
        double change = 0.0;
        for (int s = 0; s < n; ++s) {
            if (targets.count(s)) { Vn[s] = 0.0; continue; }   // absorbing target (reach reward)
            const double r = (s < (int)reward.size()) ? reward[s] : 0.0;
            double v = r + gamma * backup(m[s], V, nature, controllerMax);
            if (v > CAP) v = CAP;
            Vn[s] = v;
            change = std::max(change, std::fabs(Vn[s] - V[s]));
        }
        V.swap(Vn);
        if (change < eps) break;
    }
    IntervalResult r;
    r.iterations = iters;
    r.lower.resize(n); r.upper.resize(n);
    for (int s = 0; s < n; ++s) {
        const double v = (V[s] >= CAP * 0.9) ? HUGE_VAL : V[s];   // pinned at cap => +inf
        r.lower[s] = v; r.upper[s] = v;
    }
    return r;
}

IntervalResult expReachReward(const IMDPModel& m, const std::set<int>& targets,
                              const std::vector<double>& reward, double eps,
                              bool natureAdversarial, bool controllerMax) {
    return rewardVI(m, targets, reward, /*gamma=*/1.0, eps,
                    natureAdversarial ? omax::Sense::Min : omax::Sense::Max, controllerMax);
}
IntervalResult expDiscountedReward(const IMDPModel& m, const std::vector<double>& reward,
                                   double gamma, double eps, bool natureAdversarial, bool controllerMax) {
    return rewardVI(m, /*targets=*/{}, reward, gamma, eps,
                    natureAdversarial ? omax::Sense::Min : omax::Sense::Max, controllerMax);
}
IntervalResult maxReachRewardPessimistic(const IMDPModel& m, const std::set<int>& targets,
                                         const std::vector<double>& reward, double eps) {
    return expReachReward(m, targets, reward, eps, /*natureAdversarial=*/true, /*controllerMax=*/true);
}
IntervalResult maxReachRewardOptimistic(const IMDPModel& m, const std::set<int>& targets,
                                        const std::vector<double>& reward, double eps) {
    return expReachReward(m, targets, reward, eps, /*natureAdversarial=*/false, /*controllerMax=*/true);
}

// ---- Robust long-run average reward (mean-payoff); see solve.h for the references -------

// Phase 1: robust average reward (gain) of one MEC, using only its staying actions (support
// inside the MEC). Relative value iteration with Puterman's aperiodicity transform
// L_tau h = (1-tau) h + tau (r + opt_a opt_p sum_{s' in M} p(s') h(s')). At the fixpoint
// (L_tau h)(s) - h(s) = tau * g, so g = (L_tau h - h)/tau; the self-loop weight (1-tau) makes
// the iteration aperiodic so the relative values converge (Puterman 1994, section 8.5).
static double mecGain(const IMDPModel& m, const std::vector<int>& mec,
                      const std::vector<double>& reward, omax::Sense nature,
                      bool controllerMax, double eps) {
    const int k = (int)mec.size();
    std::unordered_map<int,int> idx;
    for (int i = 0; i < k; ++i) idx[mec[i]] = i;
    std::vector<std::vector<ActionDist>> loc(k);          // staying actions, local indices
    for (int i = 0; i < k; ++i)
        for (const ActionDist& act : m[mec[i]]) {
            bool stays = true;
            for (const Interval& iv : act) if (iv.hi > 0.0 && !idx.count(iv.to)) { stays = false; break; }
            if (!stays) continue;
            ActionDist la;
            for (const Interval& iv : act) { auto it = idx.find(iv.to); if (it != idx.end()) la.push_back({it->second, iv.lo, iv.hi}); }
            if (!la.empty()) loc[i].push_back(std::move(la));
        }
    std::vector<double> h(k, 0.0), Lh(k), Ltau(k), lo, hi, vv;
    const double tau = 0.5;
    const int MAXIT = 2000000;
    double g = 0.0, gPrev = 1e300;
    for (int it = 0; it < MAXIT; ++it) {
        for (int i = 0; i < k; ++i) {
            double best = 0.0; bool any = false;
            for (const ActionDist& a : loc[i]) {
                lo.clear(); hi.clear(); vv.clear();
                for (const Interval& iv : a) { lo.push_back(iv.lo); hi.push_back(iv.hi); vv.push_back(h[iv.to]); }
                const double q = lo.empty() ? 0.0 : omax::optimize(lo, hi, vv, nature).value;
                if (!any) { best = q; any = true; } else best = controllerMax ? std::max(best,q) : std::min(best,q);
            }
            Lh[i] = reward[mec[i]] + best;
        }
        for (int i = 0; i < k; ++i) Ltau[i] = (1.0 - tau) * h[i] + tau * Lh[i];
        g = (Ltau[0] - h[0]) / tau;                       // per-step gain (ref state = local 0)
        const double ref = Ltau[0];
        for (int i = 0; i < k; ++i) h[i] = Ltau[i] - ref; // relative values, h[0]=0
        if (std::fabs(g - gPrev) < eps) break;
        gPrev = g;
    }
    return g;
}

IntervalResult longRunAverage(const IMDPModel& m, const std::vector<double>& reward,
                              double eps, bool natureAdversarial, bool controllerMax) {
    const int n = (int)m.size();
    const omax::Sense nature = natureAdversarial ? omax::Sense::Min : omax::Sense::Max;
    std::vector<double> rew = reward; rew.resize(n, 0.0);

    // support graph -> MECs
    graph::MDPGraph g(n);
    for (int s = 0; s < n; ++s)
        for (const ActionDist& act : m[s]) {
            std::vector<int> succ;
            for (const Interval& iv : act) if (iv.hi > 0.0) succ.push_back(iv.to);
            if (!succ.empty()) g[s].push_back(std::move(succ));
        }
    const std::vector<std::vector<int>> mecList = graph::mecs(g);

    // phase 1: gain per MEC
    std::vector<double> gainOf(n, 0.0);
    std::vector<char> inMec(n, 0);
    double minGain = 1e300, maxGain = -1e300;
    for (const auto& M : mecList) {
        const double gain = mecGain(m, M, rew, nature, controllerMax, eps);
        for (int s : M) { gainOf[s] = gain; inMec[s] = 1; }
        minGain = std::min(minGain, gain); maxGain = std::max(maxGain, gain);
    }
    if (mecList.empty()) { minGain = maxGain = 0.0; }

    // phase 2: "cash-out" max-reachability VI to the best robustly-reachable MEC gain.
    //   V(s) = max( inMec ? gain(M) : -inf ,  opt_a opt_p sum p V )
    std::vector<double> V(n), Vn(n);
    for (int s = 0; s < n; ++s) V[s] = inMec[s] ? gainOf[s] : minGain;   // finite seed (no -inf)
    const int MAXIT = 2000000;
    int iters = 0;
    for (; iters < MAXIT; ++iters) {
        double change = 0.0;
        for (int s = 0; s < n; ++s) {
            double v = backup(m[s], V, nature, controllerMax);
            if (inMec[s]) v = std::max(v, gainOf[s]);          // cash-out option
            if (v > maxGain) v = maxGain;                      // clamp to the best gain (sound)
            Vn[s] = v;
            change = std::max(change, std::fabs(Vn[s] - V[s]));
        }
        V.swap(Vn);
        if (change < eps) break;
    }
    IntervalResult r; r.iterations = iters; r.lower = V; r.upper = V;
    return r;
}
IntervalResult maxLRAPessimistic(const IMDPModel& m, const std::vector<double>& reward, double eps) {
    return longRunAverage(m, reward, eps, /*natureAdversarial=*/true, /*controllerMax=*/true);
}
IntervalResult maxLRAOptimistic(const IMDPModel& m, const std::vector<double>& reward, double eps) {
    return longRunAverage(m, reward, eps, /*natureAdversarial=*/false, /*controllerMax=*/true);
}

} // namespace solve
} // namespace impact
