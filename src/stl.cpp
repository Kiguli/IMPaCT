#include "stl.h"
#include "pctl.h"          // reuse the verified bounded-until finite-horizon DP for F / U
#include "omaximization.h"

#include <limits>
#include <algorithm>

namespace impact {
namespace stl {

namespace {

// One robust Bellman backup (controller max; nature min if pessimistic, else max).
// Same finite-horizon backup verified in pctl (Baier-Katoen Ch.10 bounded operators).
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

solve::IntervalResult wrap(std::vector<double> v, int iters) {
    solve::IntervalResult r; r.lower = v; r.upper = std::move(v); r.iterations = iters; return r;
}

std::set<int> allStates(const solve::IMDPModel& m) {
    std::set<int> a; for (int s = 0; s < (int)m.size(); ++s) a.insert(s); return a;
}

// a steps of free (unconstrained) robust evolution with terminal value g.
std::vector<double> freeEvolve(const solve::IMDPModel& m, std::vector<double> g, int a, bool natureMin) {
    for (int t = 0; t < a; ++t) {
        std::vector<double> u(m.size(), 0.0);
        for (int s = 0; s < (int)m.size(); ++s) u[s] = backup(m[s], g, natureMin);
        g = std::move(u);
    }
    return g;
}

} // namespace

std::set<int> predicateCells(const std::vector<double>& lb,
                             const std::vector<double>& eta,
                             const std::vector<int>& count,
                             const std::function<bool(const std::vector<double>&)>& holds) {
    const int d = (int)lb.size();
    long long total = 1;
    for (int i = 0; i < d; ++i) total *= count[i];
    std::set<int> out;
    std::vector<int> idx(d, 0);
    std::vector<double> centre(d, 0.0);
    for (long long lin = 0; lin < total; ++lin) {
        long long rem = lin;
        for (int i = 0; i < d; ++i) { idx[i] = (int)(rem % count[i]); rem /= count[i]; }   // stride_0 = 1
        for (int i = 0; i < d; ++i) centre[i] = lb[i] + (idx[i] + 0.5) * eta[i];
        if (holds(centre)) out.insert((int)lin);
    }
    return out;
}

solve::IntervalResult eventuallyBounded(const solve::IMDPModel& m, const std::set<int>& psi, int b, bool pess) {
    return pess ? pctl::boundedUntilPessimistic(m, allStates(m), psi, b, 1e-7)
                : pctl::boundedUntilOptimistic (m, allStates(m), psi, b, 1e-7);
}

solve::IntervalResult untilBounded(const solve::IMDPModel& m, const std::set<int>& phi,
                                   const std::set<int>& psi, int b, bool pess) {
    return pess ? pctl::boundedUntilPessimistic(m, phi, psi, b, 1e-7)
                : pctl::boundedUntilOptimistic (m, phi, psi, b, 1e-7);
}

solve::IntervalResult globallyBounded(const solve::IMDPModel& m, const std::set<int>& psi, int b, bool pess) {
    const int n = (int)m.size();
    std::vector<double> W(n, 0.0);
    for (int s = 0; s < n; ++s) W[s] = psi.count(s) ? 1.0 : 0.0;   // G_[0,0] = psi now
    for (int t = 0; t < b; ++t) {
        std::vector<double> Wn(n, 0.0);
        for (int s = 0; s < n; ++s)
            Wn[s] = psi.count(s) ? backup(m[s], W, pess) : 0.0;     // psi now AND stay psi next t steps
        W = std::move(Wn);
    }
    return wrap(std::move(W), b);
}

solve::IntervalResult eventuallyWindow(const solve::IMDPModel& m, const std::set<int>& psi, int a, int b, bool pess) {
    auto inner = eventuallyBounded(m, psi, b - a, pess);            // F_[0,b-a]
    return wrap(freeEvolve(m, inner.lower, a, pess), b);            // then free-evolve a steps
}

solve::IntervalResult globallyWindow(const solve::IMDPModel& m, const std::set<int>& psi, int a, int b, bool pess) {
    auto inner = globallyBounded(m, psi, b - a, pess);             // G_[0,b-a]
    return wrap(freeEvolve(m, inner.lower, a, pess), b);
}

} // namespace stl
} // namespace impact
