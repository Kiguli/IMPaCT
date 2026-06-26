// Recorded design experiment for OMEGA-REGULAR synthesis on continuous systems
// (Phase 3) — a NEGATIVE result that motivates a future abstraction (ISSUE-0011).
//
// FINDING: abstracting an UNBOUNDED-noise system (Gaussian) with an absorbing
// off-grid SINK yields a transition system with NO non-trivial end components —
// every support path (hi>0 edges) eventually reaches the SINK, the only recurrent
// class. Consequently EVERY infinite-horizon ω-regular objective is 0, for the
// ROBUST sense (nature drives to the SINK a.s.) AND even the OPTIMISTIC sense (no
// support-closed set other than the SINK exists to cycle in). This is mathematically
// correct, not a solver bug: the robust ω-regular solvers are exact and non-trivial
// on explicit IMDPs (benchmarks/crosstool + the 12k-check oracle differential,
// ISSUE-0009). Non-trivial continuous ω-regular needs a bounded-disturbance /
// reflecting-boundary abstraction (future work).
//
// This program builds such an abstraction and confirms the values are all ~0.
//
// Build: c++ -std=c++17 -O2 benchmarks/validate_omega.cpp \
//   src/abstraction.cpp src/omega.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp -o /tmp/omg
#include "../src/abstraction.h"
#include "../src/omega.h"
#include "../src/solve.h"
#include <cstdio>
#include <vector>
#include <set>
#include <cmath>

using namespace impact;

int main() {
    abstraction::SystemND s;
    s.dim_x = 2; s.dim_u = 2;
    s.xlb = {-3, -3}; s.xub = {3, 3}; s.eta = {0.5, 0.5};
    s.ulb = {-1, -1}; s.uub = {1, 1}; s.ueta = {0.5, 0.5};
    s.A = {{0.9, 0.0}, {0.0, 0.9}}; s.B = {{0.5, 0.0}, {0.0, 0.5}}; s.c = {0.0, 0.0};
    s.sigma = {0.3, 0.3};
    s.tlo = {100, 100}; s.thi = {101, 101};                 // plain abstraction (target off-grid)

    auto ab = abstraction::buildSparseReachND(s, 1e-4);
    printf("abstraction: %d cells, nnz=%lld, %zu actions\n", ab.nCells, ab.nnz, ab.actions.size());

    // central region [-0.5,0.5]^2 as the recurrence / safe set
    auto cellLo = [&](int lin, int d){ long long j=(d==0)?(lin%12):(lin/12); return -3.0 + j*0.5; };
    std::set<int> region;
    for (int l = 0; l < ab.nCells; ++l) {
        double a0=cellLo(l,0), a1=cellLo(l,1);
        if (a0>=-0.5-1e-9 && a0+0.5<=0.5+1e-9 && a1>=-0.5-1e-9 && a1+0.5<=0.5+1e-9) region.insert(l);
    }
    auto maxval = [&](const solve::IntervalResult& r){ double m=0; for(int c=0;c<ab.nCells;c++) m=std::max(m, r.upper[c]); return m; };

    double pPer = maxval(omega::maxPersistencePessimistic(ab.model, region, 1e-6));
    double oPer = maxval(omega::maxPersistenceOptimistic (ab.model, region, 1e-6));
    double pBuc = maxval(omega::maxBuchiPessimistic(ab.model, region, 1e-6));
    double oBuc = maxval(omega::maxBuchiOptimistic (ab.model, region, 1e-6));

    printf("\nmax over cells of the omega-regular value on this abstraction:\n");
    printf("  persistence F G region : robust %.4f   optimistic %.4f\n", pPer, oPer);
    printf("  recurrence  G F region : robust %.4f   optimistic %.4f\n", pBuc, oBuc);

    bool allZero = (pPer < 1e-3 && oPer < 1e-3 && pBuc < 1e-3 && oBuc < 1e-3);
    printf("\nFINDING (ISSUE-0011): all infinite-horizon omega-regular values are %s.\n",
           allZero ? "0 (no non-trivial end components; SINK is the only recurrent class)"
                   : "NON-zero (unexpected for this abstraction)");
    printf("The robust omega-regular SOLVERS are validated non-trivially on explicit\n"
           "IMDPs: see benchmarks/crosstool (buchi/patrol/persist) and test_omega.cpp.\n");
    return allZero ? 0 : 1;
}
