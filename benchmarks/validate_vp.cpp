// End-to-end verification on a NONLINEAR ARCH benchmark (Van der Pol, verification /
// no input) through the sparse pipeline. Build the sparse interval-MC via interval-
// arithmetic mean bounds, compute the reach interval [pessimistic, optimistic], then
// simulate the real nonlinear stochastic system and check the empirical reach lies
// within the synthesized bounds (abstraction soundness vs ground truth).
//
// Build: c++ -std=c++17 -O2 benchmarks/validate_vp.cpp \
//        src/abstraction.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp -o /tmp/vp
#include "../src/abstraction.h"
#include "../src/solve.h"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

using namespace impact;
using abstraction::Ival;
using abstraction::isquare;

static void vpMean(const std::vector<double>& cl, const std::vector<double>& ch,
                   const std::vector<double>&, std::vector<double>& muLo, std::vector<double>& muHi) {
    Ival X0(cl[0], ch[0]), X1(cl[1], ch[1]);
    Ival f0 = X0 + 0.1 * X1;
    Ival f1 = X1 + 0.1 * ((-1.0 * X0) + isquare(Ival(1.0) - X0) * X1);
    muLo = {f0.lo, f1.lo}; muHi = {f0.hi, f1.hi};
}

int main() {
    abstraction::GridSpec g;
    g.dim_x = 2; g.dim_u = 0;
    g.xlb = {-5, -5}; g.xub = {5, 5}; g.eta = {0.2, 0.2};
    const double sd = 0.2;
    g.sigma = {sd, sd};
    g.tlo = {-1.2, -2.9}; g.thi = {-0.9, -2.0};

    auto ab = abstraction::buildSparseReachGeneral(g, vpMean, 1e-7);
    auto lo = solve::maxReachPessimistic(ab.model, ab.targets, 1e-6);
    auto hi = solve::maxReachOptimistic(ab.model, ab.targets, 1e-6);

    std::vector<int> Nd(2); std::vector<long long> stride(2); long long N = 1;
    for (int i = 0; i < 2; ++i) { Nd[i] = (int)std::llround((g.xub[i]-g.xlb[i])/g.eta[i]); stride[i]=N; N*=Nd[i]; }
    auto locate = [&](double x0, double x1, int& lin)->bool {
        double xs[2] = {x0, x1}; lin = 0;
        for (int i = 0; i < 2; ++i) { int j = (int)std::floor((xs[i]-g.xlb[i])/g.eta[i]);
            if (j < 0 || j >= Nd[i]) return false; lin += (int)(j*stride[i]); }
        return true;
    };
    auto inTargetCell = [&](double x0, double x1) {
        double xs[2] = {x0, x1};
        for (int i = 0; i < 2; ++i) { int j = (int)std::floor((xs[i]-g.xlb[i])/g.eta[i]);
            if (j < 0 || j >= Nd[i]) return false;
            double lo2 = g.xlb[i] + j*g.eta[i], hi2 = lo2 + g.eta[i];
            if (!(lo2 >= g.tlo[i]-1e-12 && hi2 <= g.thi[i]+1e-12)) return false; }
        return true;
    };

    std::mt19937 rng(7);
    std::normal_distribution<double> gg(0.0, sd);
    const int TRIALS = 3000, HORIZON = 400;

    printf("Van der Pol (nonlinear, verification): empirical reach vs [lower,upper]\n");
    printf("%-14s %8s %10s %8s %6s\n", "start(x0,x1)", "lower", "empirical", "upper", "in?");
    double tests[][2] = { {0.0,0.0}, {-1.0,-2.0}, {-2.0,-2.0}, {0.0,-3.0}, {-1.0,-1.0}, {2.0,0.0} };
    int ok = 0, n = 0;
    for (auto& st : tests) {
        int lin; if (!locate(st[0], st[1], lin)) continue;
        int succ = 0;
        for (int t = 0; t < TRIALS; ++t) {
            double x0 = st[0], x1 = st[1]; bool done = false;
            for (int k = 0; k < HORIZON && !done; ++k) {
                if (inTargetCell(x0, x1)) { ++succ; done = true; break; }
                int cl; if (!locate(x0, x1, cl)) { done = true; break; }
                double nx0 = x0 + 0.1*x1 + gg(rng);
                double nx1 = x1 + 0.1*(-x0 + (1-x0)*(1-x0)*x1) + gg(rng);
                x0 = nx0; x1 = nx1;
            }
        }
        double emp = (double)succ / TRIALS;
        double ci = 1.96 * std::sqrt(std::max(emp*(1-emp), 1e-9) / TRIALS);
        double L = lo.lower[lin], U = hi.upper[lin];
        bool good = (emp >= L - ci - 1e-2) && (emp <= U + ci + 1e-2);
        ++n; ok += good;
        printf("(%4.1f,%4.1f)   %8.3f %8.3f+-%.3f %8.3f %6s\n",
               st[0], st[1], L, emp, ci, U, good ? "yes" : "NO");
    }
    printf("\n%d/%d start states: empirical reach within [lower,upper].\n", ok, n);
    printf("states=%lld nnz=%lld\n", N, ab.nnz);
    return ok == n ? 0 : 1;
}
