#include "ctmc.h"

#include <cmath>
#include <algorithm>

namespace impact {
namespace ctmc {

Uniformized uniformize(const solve::IMDPModel& rateModel) {
    const int n = (int)rateModel.size();
    std::vector<double> exit(n, 0.0);
    for (int s = 0; s < n; ++s)
        if (!rateModel[s].empty())
            for (const solve::Interval& iv : rateModel[s][0])
                if (iv.to != s) exit[s] += iv.lo;             // rate (point: lo==hi)
    double lambda = 0.0;
    for (int s = 0; s < n; ++s) lambda = std::max(lambda, exit[s]);
    if (lambda <= 0.0) lambda = 1.0;                          // all-absorbing: any Λ>0 works

    solve::IMDPModel P(n);
    for (int s = 0; s < n; ++s) {
        solve::ActionDist row;
        double selfp = 1.0;
        if (!rateModel[s].empty())
            for (const solve::Interval& iv : rateModel[s][0])
                if (iv.to != s) { double p = iv.lo / lambda; row.push_back({iv.to, p, p}); selfp -= p; }
        if (selfp < 0.0) selfp = 0.0;                         // Λ >= exit so selfp >= 0
        row.push_back({s, selfp, selfp});                     // uniformisation self-loop I + Q/Λ
        P[s].push_back(std::move(row));
    }
    return { std::move(P), lambda };
}

std::vector<double> timeBoundedReach(const Uniformized& u, const std::set<int>& goal,
                                     double t, double eps) {
    const int n = (int)u.dtmc.size();
    std::vector<double> ind(n, 0.0);
    for (int g : goal) if (g >= 0 && g < n) ind[g] = 1.0;

    // P with `goal` absorbing (self-loop): once in goal we have already reached it by time t.
    solve::IMDPModel P = u.dtmc;
    for (int g : goal) if (g >= 0 && g < n) P[g] = { { solve::Interval{g, 1.0, 1.0} } };

    const double qt = u.lambda * t;
    if (qt <= 0.0) return ind;                                // t == 0: P(F<=0 goal) = ind_goal

    // Fox-Glynn Poisson weights e^{-qt}(qt)^k/k!, computed in log space relative to the mode
    // (avoids under/overflow of e^{-qt}); truncate the right tail; the e^{-qt} constant cancels
    // in the normalisation W. (Fox & Glynn, CACM 1988.)
    const int mode = (int)qt;
    const int R = mode + (int)(10.0 * std::sqrt(qt + 1.0)) + 20;
    std::vector<double> w(R + 1);
    double maxlw = -1e300;
    for (int k = 0; k <= R; ++k) { double lw = k * std::log(qt) - std::lgamma(k + 1.0); w[k] = lw; maxlw = std::max(maxlw, lw); }
    double W = 0.0;
    for (int k = 0; k <= R; ++k) { w[k] = std::exp(w[k] - maxlw); W += w[k]; }

    // result = (1/W) sum_k w_k (P^k ind); iterate v = P^k ind and accumulate.
    std::vector<double> res(n, 0.0), v = ind, vn(n);
    for (int k = 0; k <= R; ++k) {
        const double wk = w[k] / W;
        if (wk > 0.0) for (int s = 0; s < n; ++s) res[s] += wk * v[s];
        if (k == R) break;
        for (int s = 0; s < n; ++s) {                        // vn = P v
            double x = 0.0; for (const solve::Interval& iv : P[s][0]) x += iv.lo * v[iv.to];
            vn[s] = x;
        }
        v.swap(vn);
    }
    return res;
}

} // namespace ctmc
} // namespace impact
