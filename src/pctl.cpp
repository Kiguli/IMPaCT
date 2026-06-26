#include "pctl.h"
#include "omaximization.h"

#include <vector>
#include <limits>
#include <algorithm>

namespace impact {
namespace pctl {

namespace {

// One robust Bellman backup at state s over value V: controller maximizes, nature
// minimizes (pessimistic) or maximizes (optimistic) within the intervals.
double backup(const solve::StateActions& acts, const std::vector<double>& V, bool natureMin) {
    const omax::Sense sense = natureMin ? omax::Sense::Min : omax::Sense::Max;
    double best = -std::numeric_limits<double>::infinity();
    std::vector<double> lo, hi, v;
    for (const solve::ActionDist& a : acts) {
        lo.clear(); hi.clear(); v.clear();
        for (const solve::Interval& iv : a) { lo.push_back(iv.lo); hi.push_back(iv.hi); v.push_back(V[iv.to]); }
        double q = lo.empty() ? 0.0 : omax::optimize(lo, hi, v, sense).value;
        if (q > best) best = q;
    }
    return (best == -std::numeric_limits<double>::infinity()) ? 0.0 : best;
}

solve::IntervalResult exact(std::vector<double> v, int iters) {
    solve::IntervalResult r; r.lower = v; r.upper = std::move(v); r.iterations = iters; return r;
}

// Build the model used for `phi U psi`: states outside phi∪psi become absorbing
// (dead, value 0); psi states are the reach target; phi states keep their dynamics.
solve::IMDPModel restrictUntil(const solve::IMDPModel& m,
                               const std::set<int>& phi, const std::set<int>& psi) {
    solve::IMDPModel out = m;
    for (int s = 0; s < (int)m.size(); ++s) {
        if (psi.count(s)) continue;            // target — left as is (reach treats it absorbing)
        if (!phi.count(s)) out[s] = { { {s, 1.0, 1.0} } };   // ¬phi ∧ ¬psi -> dead self-loop
    }
    return out;
}

} // namespace

solve::IntervalResult nextPessimistic(const solve::IMDPModel& m, const std::set<int>& psi, double) {
    std::vector<double> ind(m.size(), 0.0);
    for (int s : psi) if (s >= 0 && s < (int)m.size()) ind[s] = 1.0;
    std::vector<double> out(m.size(), 0.0);
    for (int s = 0; s < (int)m.size(); ++s) out[s] = backup(m[s], ind, /*natureMin=*/true);
    return exact(std::move(out), 1);
}
solve::IntervalResult nextOptimistic(const solve::IMDPModel& m, const std::set<int>& psi, double) {
    std::vector<double> ind(m.size(), 0.0);
    for (int s : psi) if (s >= 0 && s < (int)m.size()) ind[s] = 1.0;
    std::vector<double> out(m.size(), 0.0);
    for (int s = 0; s < (int)m.size(); ++s) out[s] = backup(m[s], ind, /*natureMin=*/false);
    return exact(std::move(out), 1);
}

solve::IntervalResult untilPessimistic(const solve::IMDPModel& m,
                                       const std::set<int>& phi, const std::set<int>& psi, double eps) {
    return solve::maxReachPessimistic(restrictUntil(m, phi, psi), psi, eps);
}
solve::IntervalResult untilOptimistic(const solve::IMDPModel& m,
                                      const std::set<int>& phi, const std::set<int>& psi, double eps) {
    return solve::maxReachOptimistic(restrictUntil(m, phi, psi), psi, eps);
}

static solve::IntervalResult boundedUntil(const solve::IMDPModel& m, const std::set<int>& phi,
                                          const std::set<int>& psi, int k, bool natureMin) {
    const int n = (int)m.size();
    std::vector<double> V(n, 0.0);
    for (int s : psi) if (s >= 0 && s < n) V[s] = 1.0;
    for (int t = 0; t < k; ++t) {
        std::vector<double> Vn(n, 0.0);
        for (int s = 0; s < n; ++s) {
            if (psi.count(s)) { Vn[s] = 1.0; continue; }     // already satisfied
            if (!phi.count(s)) { Vn[s] = 0.0; continue; }    // left phi before psi -> fail
            Vn[s] = backup(m[s], V, natureMin);
        }
        V = std::move(Vn);
    }
    return exact(std::move(V), k);
}

solve::IntervalResult boundedUntilPessimistic(const solve::IMDPModel& m,
                        const std::set<int>& phi, const std::set<int>& psi, int k, double) {
    return boundedUntil(m, phi, psi, k, /*natureMin=*/true);
}
solve::IntervalResult boundedUntilOptimistic(const solve::IMDPModel& m,
                        const std::set<int>& phi, const std::set<int>& psi, int k, double) {
    return boundedUntil(m, phi, psi, k, /*natureMin=*/false);
}

Verdict check(double lower, double upper, Cmp op, double p) {
    // Sat iff the threshold holds for the WHOLE interval; Unsat iff it fails for all of it.
    switch (op) {
        case Cmp::Ge: if (lower >= p) return Verdict::Sat; if (upper <  p) return Verdict::Unsat; break;
        case Cmp::Gt: if (lower >  p) return Verdict::Sat; if (upper <= p) return Verdict::Unsat; break;
        case Cmp::Le: if (upper <= p) return Verdict::Sat; if (lower >  p) return Verdict::Unsat; break;
        case Cmp::Lt: if (upper <  p) return Verdict::Sat; if (lower >= p) return Verdict::Unsat; break;
    }
    return Verdict::Unknown;
}

std::set<int> satStates(const solve::IntervalResult& r, Cmp op, double p) {
    std::set<int> out;
    for (int s = 0; s < (int)r.lower.size(); ++s)
        if (check(r.lower[s], r.upper[s], op, p) == Verdict::Sat) out.insert(s);
    return out;
}

std::set<int> EF(const solve::IMDPModel& m, const std::set<int>& psi, double eps) {
    // exists strategy reaching psi with positive probability: optimistic max-reach > 0.
    auto r = solve::maxReachOptimistic(m, psi, eps);
    return satStates(r, Cmp::Gt, 0.0);
}

std::set<int> AG(const solve::IMDPModel& m, const std::set<int>& phi, double eps) {
    // robustly stay in phi almost surely: pessimistic safety == 1.
    std::set<int> avoid;
    for (int s = 0; s < (int)m.size(); ++s) if (!phi.count(s)) avoid.insert(s);
    auto r = solve::maxSafetyPessimistic(m, avoid, eps);
    return satStates(r, Cmp::Ge, 1.0 - 1e-6);
}

} // namespace pctl
} // namespace impact
