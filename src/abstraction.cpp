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
    for (int k = 0; k <= M; ++k) out.actions.push_back({sys.ulb + k * sys.ueta});

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
            // Transitions to the (disjoint) grid cells in the support window; a target
            // cell routes its mass to the absorbing TARGET. No separate target-region
            // aggregate (which would double-count cells partially overlapping a
            // grid-unaligned target). See ISSUE-0008.
            int jmin = (int)std::floor((muLo - W - sys.xlb) / sys.eta);
            int jmax = (int)std::floor((muHi + W - sys.xlb) / sys.eta);
            jmin = std::max(0, jmin); jmax = std::min(N - 1, jmax);
            for (int j = jmin; j <= jmax; ++j) {
                double cl = cellLo(j), cr = cl + sys.eta;
                Bound b = transitionInterval1D(muLo, muHi, sys.sigma, cl, cr);
                if (b.hi > prune) row.push_back({isTargetCell(j) ? TARGET : j, b.lo, b.hi});
            }

            // mass leaving the grid -> SINK (value 0), bounded TIGHTLY as the
            // complement of the whole-grid-box probability (not 1 - sum of loose
            // per-cell lower bounds, which would let nature drain all mass to sink).
            Bound g = transitionInterval1D(muLo, muHi, sys.sigma, sys.xlb, sys.xub);
            row.push_back({SINK, std::max(0.0, 1.0 - g.hi), std::min(1.0, 1.0 - g.lo)});

            out.nnz += (long long)row.size();
            out.model[i].push_back(std::move(row));
        }
    }
    out.model[TARGET].push_back({ {TARGET, 1.0, 1.0} });    // absorbing
    out.model[SINK].push_back({ {SINK, 1.0, 1.0} });
    return out;
}

// --- interval arithmetic (sound natural inclusion) for nonlinear mean bounds ----
Ival operator+(const Ival& a, const Ival& b) { return Ival(a.lo + b.lo, a.hi + b.hi); }
Ival operator-(const Ival& a, const Ival& b) { return Ival(a.lo - b.hi, a.hi - b.lo); }
Ival operator*(const Ival& a, const Ival& b) {
    double p1 = a.lo*b.lo, p2 = a.lo*b.hi, p3 = a.hi*b.lo, p4 = a.hi*b.hi;
    return Ival(std::min({p1,p2,p3,p4}), std::max({p1,p2,p3,p4}));
}
Ival operator+(const Ival& a, double s) { return Ival(a.lo + s, a.hi + s); }
Ival operator*(double s, const Ival& a) { return s >= 0 ? Ival(s*a.lo, s*a.hi) : Ival(s*a.hi, s*a.lo); }
Ival isquare(const Ival& a) {
    if (a.lo >= 0) return Ival(a.lo*a.lo, a.hi*a.hi);
    if (a.hi <= 0) return Ival(a.hi*a.hi, a.lo*a.lo);
    double m = std::max(-a.lo, a.hi);
    return Ival(0.0, m*m);
}

SparseReach buildSparseReachGeneral(const GridSpec& g, const MeanBoundFn& mean, double prune) {
    const int dx = g.dim_x, du = g.dim_u;
    std::vector<int> Nd(dx);
    std::vector<long long> stride(dx);
    long long N = 1;
    for (int i = 0; i < dx; ++i) {
        Nd[i] = std::max(1, (int)std::llround((g.xub[i] - g.xlb[i]) / g.eta[i]));
        stride[i] = N; N *= Nd[i];
    }
    const int TARGET = (int)N, SINK = (int)N + 1;

    std::vector<std::vector<double>> upts(du);
    for (int k = 0; k < du; ++k) {
        int Mk = std::max(0, (int)std::llround((g.uub[k] - g.ulb[k]) / g.ueta[k]));
        for (int t = 0; t <= Mk; ++t) upts[k].push_back(g.ulb[k] + t * g.ueta[k]);
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
    out.actions = actions;

    auto cellLoDim = [&](int i, int j) { return g.xlb[i] + j * g.eta[i]; };
    auto isTargetMi = [&](const std::vector<int>& mi) {
        for (int i = 0; i < dx; ++i) {
            double lo = cellLoDim(i, mi[i]), hi = lo + g.eta[i];
            if (!(lo >= g.tlo[i] - 1e-12 && hi <= g.thi[i] + 1e-12)) return false;
        }
        return true;
    };

    std::vector<int> mi(dx), wlo(dx), whi(dx), jt(dx);
    std::vector<double> muLo(dx), muHi(dx), cellLo(dx), cellHi(dx);
    for (long long lin = 0; lin < N; ++lin) {
        for (int i = 0; i < dx; ++i) mi[i] = (int)((lin / stride[i]) % Nd[i]);
        if (isTargetMi(mi)) { out.model[lin].push_back({ {TARGET, 1.0, 1.0} }); out.nnz += 1; continue; }
        for (int i = 0; i < dx; ++i) { cellLo[i] = cellLoDim(i, mi[i]); cellHi[i] = cellLo[i] + g.eta[i]; }

        for (const auto& u : actions) {
            mean(cellLo, cellHi, u, muLo, muHi);            // SOUND per-dim mean enclosure

            // Transitions to each grid cell in the per-dimension kernel window. Cells
            // are DISJOINT boxes, so there is no double counting. A window cell that is
            // a target cell routes its mass to the absorbing TARGET (no separate target
            // aggregate, which would double-count cells that partially overlap a
            // grid-unaligned target region).
            solve::ActionDist row;
            bool any = true;
            for (int i = 0; i < dx; ++i) {
                double W = 6.0 * g.sigma[i];
                wlo[i] = std::max(0, (int)std::floor((muLo[i] - W - g.xlb[i]) / g.eta[i]));
                whi[i] = std::min(Nd[i] - 1, (int)std::floor((muHi[i] + W - g.xlb[i]) / g.eta[i]));
                if (wlo[i] > whi[i]) any = false;
                jt[i] = wlo[i];
            }
            if (any) while (true) {
                double pl = 1.0, ph = 1.0;
                for (int i = 0; i < dx; ++i) { double cl = cellLoDim(i, jt[i]), cr = cl + g.eta[i];
                    Bound b = transitionInterval1D(muLo[i], muHi[i], g.sigma[i], cl, cr); pl *= b.lo; ph *= b.hi; }
                if (ph > prune) {
                    if (isTargetMi(jt)) { row.push_back({TARGET, pl, ph}); }
                    else { long long lj = 0; for (int i = 0; i < dx; ++i) lj += (long long)jt[i] * stride[i];
                        row.push_back({(int)lj, pl, ph}); }
                }
                int i = 0; for (; i < dx; ++i) { if (++jt[i] <= whi[i]) break; jt[i] = wlo[i]; }
                if (i == dx) break;
            }

            double gl = 1.0, gh = 1.0;                        // outside-grid via grid-box complement
            for (int i = 0; i < dx; ++i) { Bound gg = transitionInterval1D(muLo[i], muHi[i], g.sigma[i], g.xlb[i], g.xub[i]); gl *= gg.lo; gh *= gg.hi; }
            row.push_back({SINK, std::max(0.0, 1.0 - gh), std::min(1.0, 1.0 - gl)});
            out.nnz += (long long)row.size();
            out.model[lin].push_back(std::move(row));
        }
    }
    out.model[TARGET].push_back({ {TARGET, 1.0, 1.0} });
    out.model[SINK].push_back({ {SINK, 1.0, 1.0} });
    return out;
}

SparseReach buildSparseReachND(const SystemND& sys, double prune) {
    GridSpec g;
    g.dim_x = sys.dim_x; g.dim_u = sys.dim_u;
    g.xlb = sys.xlb; g.xub = sys.xub; g.eta = sys.eta;
    g.ulb = sys.ulb; g.uub = sys.uub; g.ueta = sys.ueta;
    g.sigma = sys.sigma; g.tlo = sys.tlo; g.thi = sys.thi;
    const auto A = sys.A; const auto B = sys.B; const auto c = sys.c;
    const int dx = sys.dim_x, du = sys.dim_u;
    MeanBoundFn affine = [A, B, c, dx, du](const std::vector<double>& cl, const std::vector<double>& ch,
                                           const std::vector<double>& u,
                                           std::vector<double>& muLo, std::vector<double>& muHi) {
        muLo.assign(dx, 0.0); muHi.assign(dx, 0.0);
        for (int i = 0; i < dx; ++i) {
            double lo = c.empty() ? 0.0 : c[i], hi = lo;
            for (int j = 0; j < dx; ++j) { double a = A[i][j];
                if (a >= 0) { lo += a * cl[j]; hi += a * ch[j]; } else { lo += a * ch[j]; hi += a * cl[j]; } }
            for (int k = 0; k < du; ++k) { lo += B[i][k] * u[k]; hi += B[i][k] * u[k]; }
            muLo[i] = lo; muHi[i] = hi;
        }
    };
    return buildSparseReachGeneral(g, affine, prune);
}

} // namespace abstraction
} // namespace impact
