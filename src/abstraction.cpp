#include "abstraction.h"

#include <cmath>
#include <algorithm>

namespace impact {
namespace abstraction {

namespace { constexpr double INV_SQRT2 = 0.70710678118654752440; }

double normalCdf(double z) { return 0.5 * std::erfc(-z * INV_SQRT2); }

double massInInterval(double mu, double sigma, double a, double b) {
    return normalCdf((b - mu) / sigma) - normalCdf((a - mu) / sigma);
}

Bound transitionInterval1D(double muLo, double muHi, double sigma, double a, double b) {
    if (muHi < muLo) std::swap(muLo, muHi);
    const double center = 0.5 * (a + b);
    const double muMax = std::min(std::max(center, muLo), muHi);   // closest to centre
    double hi = massInInterval(muMax, sigma, a, b);
    double lo = std::min(massInInterval(muLo, sigma, a, b),
                         massInInterval(muHi, sigma, a, b));         // farther endpoint
    lo = std::max(0.0, std::min(1.0, lo));
    hi = std::max(0.0, std::min(1.0, hi));
    if (lo > hi) lo = hi;
    return {lo, hi};
}

Bound transitionIntervalBox(const std::vector<double>& muLo,
                            const std::vector<double>& muHi,
                            const std::vector<double>& sigma,
                            const std::vector<double>& aLo,
                            const std::vector<double>& aHi) {
    double lo = 1.0, hi = 1.0;
    for (size_t d = 0; d < sigma.size(); ++d) {
        Bound b = transitionInterval1D(muLo[d], muHi[d], sigma[d], aLo[d], aHi[d]);
        lo *= b.lo; hi *= b.hi;
    }
    return {lo, hi};
}

SparseReach buildSparseReach1D(const System1D& sys, double prune) {
    const int N = std::max(1, (int)std::llround((sys.xub - sys.xlb) / sys.eta));
    const int TARGET = N, SINK = N + 1;
    const int M = std::max(0, (int)std::llround((sys.uub - sys.ulb) / sys.ueta));  // input grid points = M+1

    SparseReach out;
    out.nCells = N;
    out.nnz = 0;
    out.model.assign(N + 2, {});
    out.targets.insert(TARGET);

    auto cellLo = [&](int j) { return sys.xlb + j * sys.eta; };
    auto isTargetCell = [&](int j) {
        double lo = cellLo(j), hi = lo + sys.eta;
        return lo >= sys.tlo - 1e-12 && hi <= sys.thi + 1e-12;
    };

    const double W = 6.0 * sys.sigma;   // truncation window (~all mass within 6 sigma)

    for (int i = 0; i < N; ++i) {
        if (isTargetCell(i)) {                              // absorbing target cell
            out.model[i].push_back({ {TARGET, 1.0, 1.0} });
            out.nnz += 1;
            continue;
        }
        const double xl = cellLo(i), xr = xl + sys.eta;
        for (int k = 0; k <= M; ++k) {
            const double u = sys.ulb + k * sys.ueta;
            // affine mean range over the source cell (handle sign of a)
            double m1 = sys.a * xl + sys.b * u, m2 = sys.a * xr + sys.b * u;
            double muLo = std::min(m1, m2), muHi = std::max(m1, m2);

            solve::ActionDist row;
            // mass into the target region [tlo,thi]
            Bound bT = transitionInterval1D(muLo, muHi, sys.sigma, sys.tlo, sys.thi);
            if (bT.hi > prune) row.push_back({TARGET, bT.lo, bT.hi});

            // mass into non-target grid cells within the support window
            int jmin = (int)std::floor((muLo - W - sys.xlb) / sys.eta);
            int jmax = (int)std::floor((muHi + W - sys.xlb) / sys.eta);
            jmin = std::max(0, jmin); jmax = std::min(N - 1, jmax);
            for (int j = jmin; j <= jmax; ++j) {
                if (isTargetCell(j)) continue;              // counted in the TARGET aggregate
                double cl = cellLo(j), cr = cl + sys.eta;
                Bound b = transitionInterval1D(muLo, muHi, sys.sigma, cl, cr);
                if (b.hi > prune) row.push_back({j, b.lo, b.hi});
            }

            // remaining mass (outside grid / pruned) -> SINK, with feasible bounds
            double sumLo = 0, sumHi = 0;
            for (const auto& iv : row) { sumLo += iv.lo; sumHi += iv.hi; }
            double sinkLo = std::max(0.0, 1.0 - sumHi);
            double sinkHi = std::min(1.0, 1.0 - sumLo);
            row.push_back({SINK, sinkLo, sinkHi});

            out.nnz += (long long)row.size();
            out.model[i].push_back(std::move(row));
        }
    }
    out.model[TARGET].push_back({ {TARGET, 1.0, 1.0} });    // absorbing
    out.model[SINK].push_back({ {SINK, 1.0, 1.0} });
    return out;
}

} // namespace abstraction
} // namespace impact
