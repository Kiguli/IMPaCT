// OVI validation (ISSUE-0003 nature-trap): PD_p3 reach-avoid solved on the dense
// abstraction through IterationMethod::OptimisticVI. Interval iteration does NOT
// converge on this case; OVI must give a certified [lower,upper] with a small gap.
// Build: make PD_ovi_check ; Run: ./PD_ovi_check
#include <iostream>
#include <vector>
#include <functional>
#include "../../../../src/IMDP.h"
#include <armadillo>
using namespace std;
using namespace arma;

const int dim_x = 2, dim_u = 2, dim_w = 0;
const vec ss_lb = {-6, -6}, ss_ub = {6, 6}, ss_eta = {0.5, 0.5};
const vec is_lb = {-1,-1}, is_ub = {1, 1}, is_eta = {0.1, 0.1};
const vec sigma = {sqrt(0.02), sqrt(0.02)};
auto target_condition = [](const vec& ss) { return (ss[0] >= -4 && ss[0] <= -2) && (ss[1] >= -4 && ss[1] <= -3); };
auto avoid_condition  = [](const vec& ss) { return (ss[0] >= 0 && ss[0] <= 1 ) && (ss[1] >= -5 && ss[1] <= 1); };
auto dynamics = [](const vec& x, const vec& u) -> vec { vec xx(dim_x); xx[0]=0.9*x[0]+1.4*u[0]; xx[1]=0.8*x[1]+1.4*u[1]; return xx; };

int main() {
    IMDP mdp(dim_x,dim_u,dim_w);
    mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
    mdp.setInputSpace(is_lb, is_ub, is_eta);
    mdp.setTargetAvoidSpace(target_condition,avoid_condition, true);
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::NORMAL);
    mdp.setStdDev(sigma);
    mdp.targetTransitionVectorBounds();
    mdp.minAvoidTransitionVector();
    mdp.maxAvoidTransitionVector();
    mdp.transitionMatrixBounds();

    cout << "=== OptimisticVI (certified bracket, converges on nature-traps) ===" << endl;
    mdp.setIterationMethod(IterationMethod::OptimisticVI);
    mdp.infiniteHorizonReachControllerSorted(true);   // pessimistic / robust

    cout << "=== ValueIteration (peer-style VI, no certified upper) for cross-check ===" << endl;
    mdp.setIterationMethod(IterationMethod::ValueIteration);
    mdp.infiniteHorizonReachControllerSorted(true);
    return 0;
}
