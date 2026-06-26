// End-to-end co-safe-LTL synthesis over a CONTINUOUS system (Package-Delivery
// pattern), tying together: sparse abstraction (full dynamics IMDP) + LTLf->DFA +
// IMDP x DFA product + robust solver. Verified by Monte-Carlo: simulate the
// continuous closed loop, evaluate the realised label trace against the LTLf
// formula, and check the empirical satisfaction probability >= the synthesized
// robust lower bound.
//
// Spec: F(pickup & F deliver) -- reach the pickup region, then later the deliver
// region. 2-D robot x0'=0.9 x0 + 1.4 u0 ; x1'=0.8 x1 + 1.4 u1 + N(0,sigma^2).
//
// Build: c++ -std=c++17 -O2 benchmarks/validate_cosafe.cpp \
//   src/abstraction.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp \
//   src/ltl.cpp src/product.cpp -o /tmp/cosafe
#include "../src/abstraction.h"
#include "../src/ltl.h"
#include "../src/product.h"
#include "../src/solve.h"
#include "../src/omaximization.h"
#include <cstdio>
#include <vector>
#include <set>
#include <random>
#include <cmath>

using namespace impact;

int main() {
    // --- continuous system + grid (full dynamics IMDP: empty target box) ---
    abstraction::SystemND s;
    s.dim_x = 2; s.dim_u = 2;
    s.xlb = {-5, -5}; s.xub = {5, 5}; s.eta = {0.5, 0.5};
    s.ulb = {-1, -1}; s.uub = {1, 1}; s.ueta = {0.5, 0.5};   // 5 pts/dim
    s.A = {{0.9, 0.0}, {0.0, 0.8}};
    s.B = {{1.4, 0.0}, {0.0, 1.4}};
    s.c = {0.0, 0.0};
    const double sd = 0.2;
    s.sigma = {sd, sd};
    s.tlo = {1e18, 1e18}; s.thi = {-1e18, -1e18};            // empty -> full IMDP

    auto ab = abstraction::buildSparseReachND(s, 1e-9);
    const int N = ab.nCells;

    std::vector<int> Nd(2); std::vector<long long> stride(2); long long NN = 1;
    for (int i = 0; i < 2; ++i) { Nd[i] = (int)std::llround((s.xub[i]-s.xlb[i])/s.eta[i]); stride[i]=NN; NN*=Nd[i]; }
    auto centre = [&](int lin, double& x0, double& x1) {
        int j0 = lin % Nd[0], j1 = (lin / Nd[0]) % Nd[1];
        x0 = s.xlb[0] + (j0 + 0.5)*s.eta[0]; x1 = s.xlb[1] + (j1 + 0.5)*s.eta[1];
    };
    auto locate = [&](double x0, double x1, int& lin)->bool {
        int j0 = (int)std::floor((x0-s.xlb[0])/s.eta[0]), j1 = (int)std::floor((x1-s.xlb[1])/s.eta[1]);
        if (j0<0||j0>=Nd[0]||j1<0||j1>=Nd[1]) return false; lin = (int)(j0*stride[0]+j1*stride[1]); return true;
    };

    // --- labels: pickup region [3,5]^2, deliver region [-5,-3]^2 (by cell centre) ---
    auto labelOf = [&](double x0, double x1) {
        ltl::Letter L;
        if (x0>=3 && x0<=5 && x1>=3 && x1<=5) L.insert("pickup");
        if (x0>=-5 && x0<=-3 && x1>=-5 && x1<=-3) L.insert("deliver");
        return L;
    };
    std::vector<ltl::Letter> labels(ab.model.size());   // cells + TARGET + SINK
    for (int c = 0; c < N; ++c) { double x0,x1; centre(c,x0,x1); labels[c] = labelOf(x0,x1); }
    // TARGET (N) and SINK (N+1) carry no labels (already empty).

    // --- LTLf DFA + product + robust synthesis ---
    auto* aut = ltl::compileFinite("F(pickup & F deliver)", {"pickup", "deliver"});
    ltl::DFA dfa = ltl::toDFA(aut);
    double cx0, cx1; (void)cx0;
    // pick a start far from both regions
    int startCell; locate(0.0, 0.0, startCell);
    auto P = product::build(ab.model, labels, dfa, startCell);
    auto val = solve::maxReachPessimistic(P.model, P.targets, 1e-6);

    // --- extract robust policy on the product ---
    std::vector<int> policy(P.model.size(), 0);
    for (size_t p = 0; p < P.model.size(); ++p) {
        double best = -1; int bi = 0;
        for (size_t a = 0; a < P.model[p].size(); ++a) {
            std::vector<double> lo, hi, V;
            for (const auto& iv : P.model[p][a]) { lo.push_back(iv.lo); hi.push_back(iv.hi); V.push_back(val.lower[iv.to]); }
            double q = omax::optimize(lo, hi, V, omax::Sense::Min).value;
            if (q > best) { best = q; bi = (int)a; }
        }
        policy[p] = bi;
    }
    const int nQ = dfa.nStates;

    // --- Monte-Carlo: simulate continuous loop, evaluate the realised trace ---
    std::mt19937 rng(11);
    std::normal_distribution<double> g(0.0, sd);
    const int TRIALS = 3000, HORIZON = 300;
    printf("Co-safe F(pickup & F deliver) over a continuous robot\n");
    printf("%-12s %8s %10s %6s\n", "start", "lower", "empirical", "ok?");
    double tests[][2] = { {0.0,0.0}, {-2.0,2.0}, {2.0,-2.0}, {0.0,3.0} };
    int ok = 0, ntot = 0;
    for (auto& st : tests) {
        int c0; if (!locate(st[0], st[1], c0)) continue;
        int q0 = dfa.trans[dfa.start][ltl::letterIndex(dfa, labels[c0])];
        int pstart = c0 * nQ + q0;
        double lower = val.lower[pstart];
        int succ = 0;
        for (int t = 0; t < TRIALS; ++t) {
            double x0 = st[0], x1 = st[1];
            int q = dfa.trans[dfa.start][ltl::letterIndex(dfa, labels[c0])];
            bool done = false;
            for (int k = 0; k < HORIZON && !done; ++k) {
                if (dfa.accepting[q]) { ++succ; done = true; break; }
                int c; if (!locate(x0, x1, c)) { done = true; break; }   // left grid
                int p = c * nQ + q;
                const auto& u = ab.actions[policy[p] < (int)ab.actions.size() ? policy[p] : 0];
                x0 = 0.9*x0 + 1.4*u[0] + g(rng);
                x1 = 0.8*x1 + 1.4*u[1] + g(rng);
                int c2; if (!locate(x0, x1, c2)) { done = true; break; }
                q = dfa.trans[q][ltl::letterIndex(dfa, labels[c2])];      // DFA reads entered cell
            }
        }
        double emp = (double)succ / TRIALS;
        double ci = 1.96 * std::sqrt(std::max(emp*(1-emp),1e-9)/TRIALS);
        bool good = emp >= lower - ci - 1e-2;
        ok += good; ++ntot;
        printf("(%4.1f,%4.1f) %8.3f %8.3f   %4s\n", st[0], st[1], lower, emp, good ? "yes":"NO");
    }
    printf("\n%d/%d starts: empirical co-safe satisfaction >= robust lower bound.\n", ok, ntot);
    printf("cells=%d product_states=%zu dfa_states=%d nnz=%lld\n", N, P.model.size(), nQ, ab.nnz);
    ltl::destroy(aut);
    return ok == ntot ? 0 : 1;
}
