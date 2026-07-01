#include "ctmc.h"
#include "omaximization.h"

#include <cmath>
#include <algorithm>

namespace impact {
namespace ctmc {

Uniformized uniformize(const solve::IMDPModel& rateModel) {
    const int n = (int)rateModel.size();
    std::vector<double> exitHi(n, 0.0);            // upper exit rate (for Λ) — rates may be intervals
    for (int s = 0; s < n; ++s)
        if (!rateModel[s].empty())
            for (const solve::Interval& iv : rateModel[s][0])
                if (iv.to != s) exitHi[s] += iv.hi;
    double lambda = 0.0;
    for (int s = 0; s < n; ++s) lambda = std::max(lambda, exitHi[s]);
    if (lambda <= 0.0) lambda = 1.0;               // all-absorbing: any Λ>0 works

    solve::IMDPModel P(n);
    for (int s = 0; s < n; ++s) {
        solve::ActionDist row; double sumLo = 0.0, sumHi = 0.0;
        if (!rateModel[s].empty())
            for (const solve::Interval& iv : rateModel[s][0])
                if (iv.to != s) { double lo = iv.lo / lambda, hi = iv.hi / lambda;
                    row.push_back({iv.to, lo, hi}); sumLo += lo; sumHi += hi; }
        // uniformisation self-loop I + Q/Λ; interval self prob = [1-sum_hi, 1-sum_lo] (sound
        // interval row: total sum_lo<=1<=total sum_hi, so O-maximisation is feasible).
        double slo = 1.0 - sumHi, shi = 1.0 - sumLo;
        if (slo < 0.0) slo = 0.0; if (shi < 0.0) shi = 0.0; if (shi > 1.0) shi = 1.0;
        row.push_back({s, slo, shi});
        P[s].push_back(std::move(row));
    }
    return { std::move(P), lambda };
}

std::vector<double> timeBoundedReach(const Uniformized& u, const std::set<int>& goal,
                                     double t, double eps, bool robust) {
    const int n = (int)u.dtmc.size();
    std::vector<double> ind(n, 0.0);
    for (int g : goal) if (g >= 0 && g < n) ind[g] = 1.0;

    solve::IMDPModel P = u.dtmc;                    // goal absorbing (already reached by time t)
    for (int g : goal) if (g >= 0 && g < n) P[g] = { { solve::Interval{g, 1.0, 1.0} } };

    const double qt = u.lambda * t;
    if (qt <= 0.0) return ind;                      // t == 0: P(F<=0 goal) = ind_goal

    // Fox-Glynn Poisson weights e^{-qt}(qt)^k/k!, in log space relative to the mode (avoids
    // under/overflow; the e^{-qt} constant cancels in the normalisation W). Fox & Glynn 1988.
    const int mode = (int)qt;
    const int R = mode + (int)(10.0 * std::sqrt(qt + 1.0)) + 20;
    std::vector<double> w(R + 1); double maxlw = -1e300;
    for (int k = 0; k <= R; ++k) { double lw = k * std::log(qt) - std::lgamma(k + 1.0); w[k] = lw; maxlw = std::max(maxlw, lw); }
    double W = 0.0;
    for (int k = 0; k <= R; ++k) { w[k] = std::exp(w[k] - maxlw); W += w[k]; }

    // result = (1/W) sum_k w_k (P^k ind); each step is a ROBUST expectation (O-maximisation,
    // Sense::Min = worst-case / min reach for `robust`, Max = best-case). For a point CTMC
    // the interval collapses and this equals the ordinary uniformisation sum.
    const omax::Sense sense = robust ? omax::Sense::Min : omax::Sense::Max;
    std::vector<double> res(n, 0.0), v = ind, vn(n), lo, hi, vv;
    for (int k = 0; k <= R; ++k) {
        const double wk = w[k] / W;
        for (int s = 0; s < n; ++s) res[s] += wk * v[s];
        if (k == R) break;
        for (int s = 0; s < n; ++s) {
            lo.clear(); hi.clear(); vv.clear();
            for (const solve::Interval& iv : P[s][0]) { lo.push_back(iv.lo); hi.push_back(iv.hi); vv.push_back(v[iv.to]); }
            vn[s] = lo.empty() ? v[s] : omax::optimize(lo, hi, vv, sense).value;
        }
        v.swap(vn);
    }
    return res;
}

} // namespace ctmc
} // namespace impact
