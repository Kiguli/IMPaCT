#ifndef IMPACT_ABSTRACTION_H
#define IMPACT_ABSTRACTION_H

// ============================================================================
// Sparse interval-MDP abstraction of continuous-state stochastic systems
// (Phase: sparse / scalability — ISSUE-0006). Produces a SPARSE solve::IMDPModel
// (per-state lists of only the nonzero successor probability intervals), the same
// memory model as IntervalMDP.jl, so memory is O(nnz) not O(states^2).
//
// This header provides the correctness-critical kernel: the transition-probability
// INTERVAL [lo,hi] from a SOURCE CELL (a box of states) to a TARGET box, under a
// Gaussian noise model. For affine, axis-decoupled dynamics the per-dimension mean
// ranges are independent, so the box probability is the product of 1-D factors and
// its bounds are the product of the per-dimension bounds.
//
// Refs: interval-MDP abstraction of stochastic systems — Lahijanian-Andersson-Belta
// (IEEE TAC 2015); FAUST^2 (TACAS 2015); AMYTISS (CAV 2020); IntervalMDP.jl (arXiv
// 2401.04068) for the sparse representation. Verified against brute-force numerics
// in tests/unit/test_abstraction.cpp.
// ============================================================================

#include <vector>
#include <functional>
#include "solve.h"

namespace impact {
namespace abstraction {

    // Standard normal CDF.
    double normalCdf(double z);

    // Probability mass of N(mu, sigma^2) in the interval [a, b].
    double massInInterval(double mu, double sigma, double a, double b);

    struct Bound { double lo; double hi; };

    // The 1-D transition probability mass of N(mu, sigma^2) in [a,b], minimized and
    // maximized over the mean mu ranging in [muLo, muHi]. Closed form: the mass is
    // unimodal in mu with peak at the box centre (a+b)/2 — so the max is at mu
    // clamped to the centre and the min at the farther endpoint. Requires muLo<=muHi,
    // a<=b, sigma>0.
    Bound transitionInterval1D(double muLo, double muHi, double sigma, double a, double b);

    // Axis-decoupled n-D box bound = product of per-dimension 1-D bounds.
    Bound transitionIntervalBox(const std::vector<double>& muLo,
                                const std::vector<double>& muHi,
                                const std::vector<double>& sigma,
                                const std::vector<double>& aLo,
                                const std::vector<double>& aHi);

    // --- 1-D sparse reach abstraction (the first end-to-end, verifiable case) -----
    // System: x' = a*x + b*u + N(0, sigma^2), x on grid [xlb,xub] step eta, inputs
    // on grid [ulb,uub] step ueta. Target region [tlo,thi] (absorbing). Mass leaving
    // the grid (or into the target) is handled by absorbing TARGET / SINK states.
    struct System1D {
        double a, b, sigma;
        double xlb, xub, eta;
        double ulb, uub, ueta;
        double tlo, thi;
    };

    struct SparseReach {
        solve::IMDPModel model;     // states: 0..N-1 cells, N = TARGET, N+1 = SINK
        std::set<int> targets;      // { N }
        int nCells;
        long long nnz;              // total stored successor intervals (sparsity metric)
        // Input vector for each action index (same action set for every non-target
        // cell); lets a synthesized policy be simulated on the continuous system.
        std::vector<std::vector<double>> actions;
    };

    // Build the sparse IMDP. `prune` drops successor cells whose upper bound <= prune
    // (mass folded into SINK); prune=0 keeps every cell (a dense-equivalent build).
    SparseReach buildSparseReach1D(const System1D& sys, double prune);

    // --- n-D sparse reach abstraction (affine dynamics, diagonal Gaussian) --------
    // x'_i = sum_j A[i][j] x_j + sum_k B[i][k] u_k + c[i] + N(0, sigma_i^2).
    // The per-dimension mean RANGE over a source cell is computed exactly by interval
    // arithmetic on the affine map (handles COUPLED systems). The box transition
    // bound is the product of per-dimension 1-D bounds: this is EXACT when dynamics
    // are axis-decoupled (A diagonal) and a SOUND over-approximation otherwise (the
    // true probability always lies within [prod lo_i, prod hi_i]). Only cells inside
    // the per-dimension kernel window are stored (sparse).
    struct SystemND {
        int dim_x = 0, dim_u = 0;
        std::vector<double> xlb, xub, eta;       // size dim_x
        std::vector<double> ulb, uub, ueta;      // size dim_u
        std::vector<std::vector<double>> A;      // dim_x x dim_x
        std::vector<std::vector<double>> B;      // dim_x x dim_u
        std::vector<double> c;                   // dim_x (default 0)
        std::vector<double> sigma;               // dim_x
        std::vector<double> tlo, thi;            // dim_x target box (absorbing)
    };

    SparseReach buildSparseReachND(const SystemND& sys, double prune);

    // --- General (nonlinear) sparse reach abstraction ----------------------------
    // Decouples the dynamics from the gridding: the caller supplies a SOUND
    // per-dimension mean enclosure of f_i over a source cell under a fixed input.
    // For nonlinear f this is typically built with interval arithmetic
    // (`abstraction::Ival`) or mixed-monotone bounds (Dutreix-Coogan). Soundness of
    // the resulting IMDP follows from the enclosure being sound (true mean in range).
    struct GridSpec {
        int dim_x = 0, dim_u = 0;
        std::vector<double> xlb, xub, eta;   // dim_x
        std::vector<double> ulb, uub, ueta;  // dim_u
        std::vector<double> sigma;           // dim_x (diagonal Gaussian)
        std::vector<double> tlo, thi;        // dim_x target box
    };

    // (cellLo, cellHi, u) -> per-dimension [muLo, muHi] enclosing f over the cell.
    using MeanBoundFn = std::function<void(const std::vector<double>& cellLo,
                                           const std::vector<double>& cellHi,
                                           const std::vector<double>& u,
                                           std::vector<double>& muLo,
                                           std::vector<double>& muHi)>;

    SparseReach buildSparseReachGeneral(const GridSpec& g, const MeanBoundFn& mean, double prune);

    // Minimal interval-arithmetic type for writing sound nonlinear mean bounds.
    struct Ival {
        double lo, hi;
        Ival(double l, double h) : lo(l), hi(h) {}
        explicit Ival(double v) : lo(v), hi(v) {}
    };
    Ival operator+(const Ival& a, const Ival& b);
    Ival operator-(const Ival& a, const Ival& b);
    Ival operator*(const Ival& a, const Ival& b);
    Ival operator+(const Ival& a, double s);
    Ival operator*(double s, const Ival& a);
    Ival isquare(const Ival& a);     // [a]^2, tight (handles intervals straddling 0)

} // namespace abstraction
} // namespace impact

#endif // IMPACT_ABSTRACTION_H
