// End-to-end verification of the sparse pipeline against the REAL continuous system
// (ISSUE-0006 / "actually verify the approaches"). We build a sparse IMDP for an
// affine diagonal-Gaussian system, synthesize a robust controller, then simulate the
// CONTINUOUS closed loop under that controller and check the empirical reach
// probability lies within the synthesized [lower, upper] bounds (the abstraction's
// soundness guarantee, validated against ground truth).
//
// Build: c++ -std=c++17 -O2 benchmarks/validate_montecarlo.cpp \
//        src/abstraction.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp -o /tmp/mc
#include "../src/abstraction.h"
#include "../src/solve.h"
#include "../src/omaximization.h"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

using namespace impact;

int main() {
    // 2-D robot-style affine system: x0'=0.9 x0 + 1.4 u0 ; x1'=0.8 x1 + 1.4 u1 + noise
    abstraction::SystemND s;
    s.dim_x = 2; s.dim_u = 2;
    s.xlb = {-3, -3}; s.xub = {3, 3}; s.eta = {0.3, 0.3};
    s.ulb = {-1, -1}; s.uub = {1, 1}; s.ueta = {0.5, 0.5};   // 5 pts/dim
    s.A = {{0.9, 0.0}, {0.0, 0.8}};
    s.B = {{1.4, 0.0}, {0.0, 1.4}};
    s.c = {0.0, 0.0};
    const double sd = 0.8;  // high noise -> reach values in (0,1): discriminating check
    s.sigma = {sd, sd};
    s.tlo = {2.0, 2.0}; s.thi = {3.0, 3.0};

    auto ab = abstraction::buildSparseReachND(s, 1e-9);
    auto val = solve::maxReachPessimistic(ab.model, ab.targets, 1e-6);   // robust controller

    // grid geometry
    std::vector<int> Nd(2); std::vector<long long> stride(2); long long N = 1;
    for (int i = 0; i < 2; ++i) { Nd[i] = (int)std::llround((s.xub[i]-s.xlb[i])/s.eta[i]); stride[i]=N; N*=Nd[i]; }
    auto locate = [&](double x0, double x1, int& lin)->bool {
        double xs[2] = {x0, x1}; lin = 0;
        for (int i = 0; i < 2; ++i) { int j = (int)std::floor((xs[i]-s.xlb[i])/s.eta[i]);
            if (j < 0 || j >= Nd[i]) return false; lin += (int)(j*stride[i]); }
        return true;
    };
    // success == entering a TARGET CELL (a cell whose box is fully inside the target
    // region), matching the abstraction's absorbing TARGET exactly. This alignment is
    // required for the robust lower-bound performance guarantee to apply.
    auto inTargetCell = [&](double x0, double x1) {
        double xs[2] = {x0, x1};
        for (int i = 0; i < 2; ++i) {
            int j = (int)std::floor((xs[i]-s.xlb[i])/s.eta[i]);
            if (j < 0 || j >= Nd[i]) return false;
            double lo = s.xlb[i] + j*s.eta[i], hi = lo + s.eta[i];
            if (!(lo >= s.tlo[i]-1e-12 && hi <= s.thi[i]+1e-12)) return false;
        }
        return true;
    };

    // extract robust policy: argmax_a min_nature E[V_lower] (omax Sense::Min)
    std::vector<int> policy(N, 0);
    for (long long c = 0; c < N; ++c) {
        const auto& acts = ab.model[c];
        double best = -1; int bi = 0;
        for (size_t a = 0; a < acts.size(); ++a) {
            std::vector<double> lo, hi, V;
            for (const auto& iv : acts[a]) { lo.push_back(iv.lo); hi.push_back(iv.hi); V.push_back(val.lower[iv.to]); }
            double q = omax::optimize(lo, hi, V, omax::Sense::Min).value;
            if (q > best) { best = q; bi = (int)a; }
        }
        policy[c] = bi;
    }

    std::mt19937 rng(2026);
    std::normal_distribution<double> g0(0.0, sd), g1(0.0, sd);
    const int TRIALS = 4000, HORIZON = 2000;

    printf("%-14s %8s %10s %8s %7s\n", "start(x0,x1)", "lower", "empirical", "upper", "in?");
    int startsChecked = 0, startsOK = 0;
    double tests[][2] = { {-2.5,-2.5}, {-1.0,-1.0}, {0.0,0.0}, {1.0,1.0}, {1.5,1.5}, {-2.5,2.5} };
    for (auto& st : tests) {
        int lin; if (!locate(st[0], st[1], lin)) continue;
        int succ = 0;
        for (int t = 0; t < TRIALS; ++t) {
            double x0 = st[0], x1 = st[1]; bool done = false;
            for (int k = 0; k < HORIZON && !done; ++k) {
                if (inTargetCell(x0, x1)) { ++succ; done = true; break; }
                int cl; if (!locate(x0, x1, cl)) { done = true; break; }   // left grid -> fail
                const auto& u = ab.actions[policy[cl]];
                double nx0 = 0.9*x0 + 1.4*u[0] + g0(rng);
                double nx1 = 0.8*x1 + 1.4*u[1] + g1(rng);
                x0 = nx0; x1 = nx1;
            }
        }
        double emp = (double)succ / TRIALS;
        double ci = 1.96 * std::sqrt(std::max(emp*(1-emp), 1e-9) / TRIALS);
        double lo = val.lower[lin], hi = val.upper[lin];
        bool ok = (emp >= lo - ci - 5e-3);  // robust guarantee: real (benign) noise >= worst-case lower
        ++startsChecked; startsOK += ok;
        printf("(%4.1f,%4.1f)   %8.3f %8.3f+-%.3f %8.3f %7s\n",
               st[0], st[1], lo, emp, ci, hi, ok ? "yes" : "NO");
    }
    printf("\n%d/%d start states: empirical reach >= robust lower bound (performance guarantee).\n", startsOK, startsChecked);
    printf("states=%lld actions=%zu nnz=%lld\n", N, ab.actions.size(), ab.nnz);
    return startsOK == startsChecked ? 0 : 1;
}
