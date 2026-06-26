// Sparse-abstraction scaling demo (ISSUE-0006): grow the DOMAIN with the grid step
// (eta) and noise (sigma) FIXED, so the Gaussian kernel spans a constant number of
// cells. Then stored nonzeros are O(N) while a dense (min+max) matrix is O(N^2):
// large state counts that would OOM densely run in megabytes.
// (Refining eta with fixed sigma instead would widen the kernel-in-cells and is NOT
// asymptotically sparse — the win is fixed kernel width, growing N.)
//
// Build:  c++ -std=c++17 -O2 benchmarks/sparse_scaling.cpp \
//             src/abstraction.cpp src/solve.cpp src/omaximization.cpp src/graph_utils.cpp -o /tmp/scaling
// Run:    /tmp/scaling
#include "../src/abstraction.h"
#include "../src/solve.h"
#include <cstdio>
#include <chrono>

using namespace impact;

int main() {
    printf("%10s %10s %12s %10s %12s %12s %9s\n",
           "domain", "cells", "nnz", "nnz/cell", "sparse(MB)", "dense(MB)", "synth(s)");
    for (double half : {10.0, 50.0, 250.0, 1250.0, 6250.0}) {   // grow domain, fixed eta+sigma
        abstraction::System1D s;
        s.a = 0.9; s.b = 1.0; s.sigma = 0.3;
        s.xlb = -half; s.xub = half; s.eta = 0.1;   // fixed step => kernel ~ constant #cells
        s.ulb = -1; s.uub = 1; s.ueta = 1.0;        // 3 inputs
        s.tlo = half - 2; s.thi = half;

        auto ab = abstraction::buildSparseReach1D(s, 1e-7);
        auto t0 = std::chrono::steady_clock::now();
        auto r = solve::maxReachOptimistic(ab.model, ab.targets, 1e-6);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        double sparseMB = ab.nnz * (double)(sizeof(int) + 2 * sizeof(double)) / 1e6;
        double denseMB  = (double)ab.nCells * ab.nCells * 2 * sizeof(double) / 1e6; // min+max N x N
        printf("%10.0f %10d %12lld %10.1f %12.2f %12.1f %9.2f\n",
               2 * half, ab.nCells, ab.nnz, (double)ab.nnz / (ab.nCells + 2),
               sparseMB, denseMB, secs);
        (void)r;
    }
    return 0;
}
