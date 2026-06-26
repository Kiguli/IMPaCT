// End-to-end verification of ROBUST OMEGA-REGULAR synthesis on a continuous system
// with BOUNDED disturbance (the abstraction that makes it non-trivial — ISSUE-0011).
//
// System (1-D): x' = 0.9 x + 0.5 u + w,  w ~ Uniform[-W, W],  x in [-3,3], u in [-1,1].
// Spec: recurrence  G F region  (visit the central region [-0.5,0.5] infinitely often).
//
// Because the disturbance is BOUNDED, a cell well inside the domain has its entire
// next-state window inside the domain — NO leak to the absorbing sink — so genuine
// robust end components exist and the robust recurrence value is non-trivial (unlike
// the unbounded-Gaussian case, ISSUE-0011, where every infinite-horizon value is 0).
//
// We synthesize the robust recurrence controller (greedy on robust reach-to-region),
// simulate the CONTINUOUS closed loop under real bounded noise, and check the
// empirical recurrence frequency is at/above the synthesized robust lower bound.
//
// Build: c++ -std=c++17 -O2 benchmarks/validate_omega_bounded.cpp \
//   src/abstraction.cpp src/omega.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp -o /tmp/omgb
#include "../src/abstraction.h"
#include "../src/omega.h"
#include "../src/solve.h"
#include "../src/omaximization.h"
#include <cstdio>
#include <vector>
#include <set>
#include <random>
#include <cmath>

using namespace impact;

int main() {
    const double xlb = -3, xub = 3, eta = 0.5;       // 12 cells
    const double ulb = -1, uub = 1, ueta = 0.5;       // 5 inputs
    const double A = 0.9, B = 0.5, W = 0.3, prune = 1e-9;
    const int NC = (int)std::llround((xub - xlb) / eta);
    const int SINK = NC;                              // single absorbing off-domain sink
    auto cellLo = [&](int c){ return xlb + c * eta; };

    std::vector<double> us;
    for (double u = ulb; u <= uub + 1e-9; u += ueta) us.push_back(u);

    // Build the cell IMDP with the bounded-uniform kernel.
    solve::IMDPModel m(NC + 1);
    long long nnz = 0;
    for (int c = 0; c < NC; ++c) {
        double cl = cellLo(c), cr = cl + eta;
        for (double u : us) {
            double muLo = A * cl + B * u, muHi = A * cr + B * u;
            solve::ActionDist act;
            for (int j = 0; j < NC; ++j) {
                double a = cellLo(j), b = a + eta;
                abstraction::Bound bd = abstraction::transitionInterval1DUniform(muLo, muHi, W, a, b);
                if (bd.hi > prune) { act.push_back({j, bd.lo, bd.hi}); ++nnz; }
            }
            // off-domain mass -> SINK (left and right tails); 0 for interior cells.
            abstraction::Bound lft = abstraction::transitionInterval1DUniform(muLo, muHi, W, xlb - 10, xlb);
            abstraction::Bound rgt = abstraction::transitionInterval1DUniform(muLo, muHi, W, xub, xub + 10);
            double slo = lft.lo + rgt.lo, shi = lft.hi + rgt.hi;
            if (shi > prune) { act.push_back({SINK, slo, shi}); ++nnz; }
            m[c].push_back(std::move(act));
        }
    }
    m[SINK].push_back({{SINK, 1.0, 1.0}});           // absorbing
    printf("bounded-noise 1-D abstraction: %d cells (+sink), nnz=%lld, %zu inputs\n", NC, nnz, us.size());

    // region [-0.5, 0.5]
    std::set<int> region;
    for (int c = 0; c < NC; ++c) { double cl = cellLo(c); if (cl >= -0.5 - 1e-9 && cl + eta <= 0.5 + 1e-9) region.insert(c); }

    auto buc    = omega::maxBuchiPessimistic(m, region, 1e-7);     // robust recurrence value
    auto vacc   = solve::maxReachPessimistic(m, region, 1e-7);     // for the greedy recurrence policy
    int nWin = 0; for (int c = 0; c < NC; ++c) if (buc.lower[c] > 0.5) ++nWin;
    printf("robust recurrence (G F region): %d/%d cells with value > 0.5\n", nWin, NC);

    // greedy "always head to region" policy (achieves recurrence within the winning region).
    std::vector<int> policy(NC, 0);
    for (int c = 0; c < NC; ++c) {
        double best = -1; int bi = 0;
        for (size_t a = 0; a < m[c].size(); ++a) {
            std::vector<double> lo, hi, V;
            for (const auto& iv : m[c][a]) { lo.push_back(iv.lo); hi.push_back(iv.hi); V.push_back(vacc.lower[iv.to]); }
            double q = lo.empty() ? 0.0 : omax::optimize(lo, hi, V, omax::Sense::Min).value;
            if (q > best) { best = q; bi = (int)a; }
        }
        policy[c] = bi;
    }
    auto locate = [&](double x, int& c){ int j = (int)std::floor((x - xlb)/eta); if (j<0||j>=NC) return false; c=j; return true; };
    auto inRegion = [&](double x){ return x >= -0.5 && x <= 0.5; };

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> wdist(-W, W);
    const int TRIALS = 4000, HORIZON = 400, TAILFROM = 200; const int NEED = 5;  // i.o. proxy

    printf("\nG F region — robust lower bound vs empirical recurrence:\n");
    printf("%-8s %10s %12s %6s\n", "start", "robust_lo", "empirical", "in?");
    double tests[] = { 0.0, 1.0, -1.5, 2.0, -2.5 };
    int okN = 0, chk = 0;
    for (double x0 : tests) {
        int lin; if (!locate(x0, lin)) continue;
        int succ = 0;
        for (int t = 0; t < TRIALS; ++t) {
            double x = x0; int visitsTail = 0; bool alive = true;
            for (int k = 0; k < HORIZON; ++k) {
                int c; if (!locate(x, c)) { alive = false; break; }
                if (k >= TAILFROM && inRegion(x)) ++visitsTail;
                x = A * x + B * us[policy[c]] + wdist(rng);
            }
            if (alive && visitsTail >= NEED) ++succ;          // visited region many times late -> recurring
        }
        double emp = (double)succ / TRIALS, lo = buc.lower[lin];
        bool ok = emp >= lo - 0.03;
        printf("%-8.1f %10.3f %12.3f %6s\n", x0, lo, emp, ok ? "yes" : "NO");
        ++chk; if (ok) ++okN;
    }
    printf("\n%d/%d starts: empirical recurrence >= robust lower bound.\n", okN, chk);
    return (okN == chk && nWin > 0) ? 0 : 1;
}
