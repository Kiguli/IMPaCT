// Run: make
//        ./vehicle3D

// Code: Ben Wooding 6 Jan 2024

#include <iostream>
#include <vector>
#include <functional>
#include "../../../../src/IMDP.h"
#include <chrono>
#include <armadillo>

using namespace std;
using namespace arma;

/*
 ################################# PARAMETERS ###############################################
 */

// Set the dimensions
const int dim_x = 3;
const int dim_u = 2;
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {-5.0, -5.0,-3.4};
const vec ss_ub = {5.0, 5.0, 3.4};
const vec ss_eta = {0.5, 0.5,0.4};
// Inputs
const vec is_lb = {-1.0,-0.4};
const vec is_ub = {4.0,0.4};
const vec is_eta = {1, 0.2};

//standard deviation of each dimension
const vec sigma = {sqrt(1/1.5), sqrt(1/1.5),sqrt(1/1.5)};

// logical expression for target region and avoid region
auto target_condition = [](const vec& ss) { return (ss(0) >= -5.75 && ss(0) <= 0.25) && (ss(1) >= -0.25 && ss(1) <= 5.75) && (ss(2) >= -3.45 && ss(2) <= 3.45);};
auto avoid_condition = [](const vec& ss) { return (ss(0) >= -5.75 && ss(0) <= 0.25) && (ss(1) >= -0.75 && ss(1) <= -0.25) && (ss(2) >= -3.45 && ss(2) <= 3.45);};

//dynamics - 2 parameters
const auto dynamics = [](const vec& x, const vec& u) -> vec {
    float Ts = 0.100;
    vec f(3);
    vec xx = x;
    
    f[0] = u[0]*cos(atan((float)(tan(u[1])/2.0))+xx[2])/cos((float)atan((float)(tan(u[1])/2.0)));
    f[1] = u[0]*sin(atan((float)(tan(u[1])/2.0))+xx[2])/cos((float)atan((float)(tan(u[1])/2.0)));
    f[2] = u[0]*tan(u[1]);
    
    xx = xx + Ts*f;
    return xx;
};

/*
 ################################# MAIN FUNCTION ######################################
 */

int main() {
    
    /* ###### create IMDP object ###### */
    IMDP mdp(dim_x,dim_u,dim_w);
    
    /* ###### create finite sets for the different spaces ###### */
    mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
    mdp.setInputSpace(is_lb, is_ub, is_eta);
    
    /* ###### relabel states based on specification ###### */
    mdp.setTargetAvoidSpace(target_condition,avoid_condition, true);
    
    /*###### save the files ######*/
    //mdp.saveStateSpace();
    //mdp.saveInputSpace();
    //mdp.saveTargetSpace();
    
    /*###### set dynamics and noise ######*/
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::NORMAL);
    mdp.setStdDev(sigma);
    
    /* ###### calculate abstraction for target vectors ######*/
    mdp.targetTransitionVectorBounds();
    
    /* ###### save target vectors ######*/
    //mdp.saveMinTargetTransitionVector();
    //mdp.saveMaxTargetTransitionVector();
    
    /* ###### calculate abstraction for avoid vectors ######*/
    mdp.minAvoidTransitionVector();
    mdp.maxAvoidTransitionVector();
    
    /* ###### save avoid vectors ######*/
    //mdp.saveMinAvoidTransitionVector();
    //mdp.saveMaxAvoidTransitionVector();
    
    
    /* ###### calculate abstraction for transition matrices ######*/
    mdp.transitionMatrixBounds();
    
    /* ###### save transition matrices ######*/
    //mdp.saveMinTransitionMatrix();
    //mdp.saveMaxTransitionMatrix();
    
    /* ###### synthesize infinite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.infiniteHorizonReachControllerSorted(true);
    
    /* ###### synthesize finite horizon controller (true = pessimistic, false = optimistic) ######*/
    //mdp.finiteHorizonReachControllerSorted(true,10);
    
    /* ###### save controller ######*/
    //mdp.saveController();
    
    return 0;
}
