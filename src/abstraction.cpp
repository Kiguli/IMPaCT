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

SparseReach buildSparseReachND(const SystemND& sys, double prune) {
    const int dx = sys.dim_x, du = sys.dim_u;
    std::vector<int> Nd(dx);
    std::vector<long long> stride(dx);
    long long N = 1;
    for (int i = 0; i < dx; ++i) {
        Nd[i] = std::max(1, (int)std::llround((sys.xub[i] - sys.xlb[i]) / sys.eta[i]));
        stride[i] = N; N *= Nd[i];
    }
    const int TARGET = (int)N, SINK = (int)N + 1;

    // input actions = Cartesian product of per-dimension input grids
    std::vector<std::vector<double>> upts(du);
    for (int k = 0; k < du; ++k) {
        int Mk = std::max(0, (int)std::llround((sys.uub[k] - sys.ulb[k]) / sys.ueta[k]));
        for (int t = 0; t <= Mk; ++t) upts[k].push_back(sys.ulb[k] + t * sys.ueta[k]);
    }
    std::vector<std::vector<double>> actions;
    if (du == 0) actions.push_back({});
    else {
        std::vector<int> idx(du, 0);
        while (true) {
            std::vector<double> u(du);
            for (int k = 0; k < du; ++k) u[k] = upts[k][idx[k]];
            actions.push_back(std::move(u));
            int k = 0; for (; k < du; ++k) { if (++idx[k] < (int)upts[k].size()) break; idx[k] = 0; }
            if (k == du) break;
        }
    }

    SparseReach out; out.nCells = (int)N; out.nnz = 0;
    out.model.assign((size_t)N + 2, {}); out.targets.insert(TARGET);

    auto cellLoDim = [&](int i, int j) { return sys.xlb[i] + j * sys.eta[i]; };
    auto isTargetMi = [&](const std::vector<int>& mi) {
        for (int i = 0; i < dx; ++i) {
            double lo = cellLoDim(i, mi[i]), hi = lo + sys.eta[i];
            if (!(lo >= sys.tlo[i] - 1e-12 && hi <= sys.thi[i] + 1e-12)) return false;
        }
        return true;
    };

    std::vector<int> mi(dx), wlo(dx), whi(dx), jt(dx);
    std::vector<double> muLo(dx), muHi(dx);
    for (long long lin = 0; lin < N; ++lin) {
        for (int i = 0; i < dx; ++i) mi[i] = (int)((lin / stride[i]) % Nd[i]);
        if (isTargetMi(mi)) { out.model[lin].push_back({ {TARGET, 1.0, 1.0} }); out.nnz += 1; continue; }

        for (const auto& u : actions) {
            for (int i = 0; i < dx; ++i) {                 // affine interval arithmetic for mean range
                double lo = sys.c.empty() ? 0.0 : sys.c[i], hi = lo;
                for (int j = 0; j < dx; ++j) {
                    double a = sys.A[i][j], xl = cellLoDim(j, mi[j]), xr = xl + sys.eta[j];
                    if (a >= 0) { lo += a * xl; hi += a * xr; } else { lo += a * xr; hi += a * xl; }
                }
                for (int k = 0; k < du; ++k) { lo += sys.B[i][k] * u[k]; hi += sys.B[i][k] * u[k]; }
                muLo[i] = lo; muHi[i] = hi;
            }

            solve::ActionDist row;
            { double tl = 1.0, th = 1.0;                   // target box aggregate
              for (int i = 0; i < dx; ++i) { Bound b = transitionInterval1D(muLo[i], muHi[i], sys.sigma[i], sys.tlo[i], sys.thi[i]); tl *= b.lo; th *= b.hi; }
              if (th > prune) row.push_back({TARGET, tl, th}); }

            bool any = true;                                // per-dim kernel window
            for (int i = 0; i < dx; ++i) {
                double W = 6.0 * sys.sigma[i];
                wlo[i] = std::max(0, (int)std::floor((muLo[i] - W - sys.xlb[i]) / sys.eta[i]));
                whi[i] = std::min(Nd[i] - 1, (int)std::floor((muHi[i] + W - sys.xlb[i]) / sys.eta[i]));
                if (wlo[i] > whi[i]) any = false;
                jt[i] = wlo[i];
            }
            if (any) while (true) {                         // Cartesian product over windows
                if (!isTargetMi(jt)) {
                    double pl = 1.0, ph = 1.0;
                    for (int i = 0; i < dx; ++i) { double cl = cellLoDim(i, jt[i]), cr = cl + sys.eta[i];
                        Bound b = transitionInterval1D(muLo[i], muHi[i], sys.sigma[i], cl, cr); pl *= b.lo; ph *= b.hi; }
                    if (ph > prune) { long long lj = 0; for (int i = 0; i < dx; ++i) lj += (long long)jt[i] * stride[i];
                        row.push_back({(int)lj, pl, ph}); }
                }
                int i = 0; for (; i < dx; ++i) { if (++jt[i] <= whi[i]) break; jt[i] = wlo[i]; }
                if (i == dx) break;
            }

            double sumLo = 0, sumHi = 0;
            for (const auto& iv : row) { sumLo += iv.lo; sumHi += iv.hi; }
            row.push_back({SINK, std::max(0.0, 1.0 - sumHi), std::min(1.0, 1.0 - sumLo)});
            out.nnz += (long long)row.size();
            out.model[lin].push_back(std::move(row));
        }
    }
    out.model[TARGET].push_back({ {TARGET, 1.0, 1.0} });
    out.model[SINK].push_back({ {SINK, 1.0, 1.0} });
    return out;
}

} // namespace abstraction
} // namespace impact
