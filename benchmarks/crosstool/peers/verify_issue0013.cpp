// ISSUE-0013 toolchain verification: the fixed finiteHorizonSafeControllerSorted
// UPPER-bound kernels, on a NO-INPUT model where the semantics are unambiguous
// (no policy choice, so lower = pessimistic-nature safety and upper = optimistic-
// nature safety EXACTLY = IntervalMDP.jl FiniteTimeSafety Pessimistic/Optimistic).
// 2-D decoupled affine system (sparse==dense exact regime, ISSUE-0020), H=5.
// Outputs controller.h5 ([x | lower | upper]) + issue0013.imdp for the oracle.
#include <iostream>
#include <vector>
#include <functional>
#include "../../../src/IMDP.h"
#include <armadillo>
using namespace std;
using namespace arma;

const int dim_x = 2, dim_u = 0, dim_w = 0;
const vec ss_lb = {0.0, 0.0}, ss_ub = {2.0, 2.0}, ss_eta = {0.1, 0.1};
const vec sigma = {0.1, 0.1};
auto dynamics = [](const vec& x) -> vec { vec xx(dim_x); xx[0]=0.9*x[0]+0.1; xx[1]=0.9*x[1]+0.1; return xx; };

int main(int argc, char** argv) {
    bool lower = !(argc > 1 && string(argv[1]) == "upper-first");
    IMDP mdp(dim_x, dim_u, dim_w);
    mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::NORMAL);
    mdp.setStdDev(sigma);
    mdp.minAvoidTransitionVector();       // safety: avoid = leaving the domain
    mdp.maxAvoidTransitionVector();
    mdp.transitionMatrixBounds();
    mdp.exportIMDP("issue0013.imdp");
    mdp.finiteHorizonSafeControllerSorted(lower, 5);
    mdp.saveController();
    return 0;
}
