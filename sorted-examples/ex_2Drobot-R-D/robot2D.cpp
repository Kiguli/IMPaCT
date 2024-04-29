// Run: make
// Then: ./robot2D

// Code: Ben Wooding 4 Jan 2024

#include <iostream>
#include <vector>
#include <functional>
#include "../../src/IMDP.h"
#include <armadillo>
#include <chrono>

using namespace std;
using namespace arma;

/*
 ################################# PARAMETERS ###############################################
 */

// Set the dimensions
const int dim_x = 2;
const int dim_u = 2;
const int dim_w = 1;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {-10, -10};
const vec ss_ub = {10, 10};
const vec ss_eta = {1, 1};
// Inputs
const vec is_lb = {-1,-1};
const vec is_ub = {1, 1};
const vec is_eta = {0.2, 0.2};
//Disturbances
const vec ws_lb = {-0.5};
const vec ws_ub = {0.5};
const vec ws_eta = {0.1};

//standard deviation of each dimension
const vec sigma = {sqrt(1/1.3333), sqrt(1/1.3333)};

// logical expression for target region
auto target_condition = [](const vec& ss) { return (ss[0] >= 5.0 && ss[0] <= 8.0) && (ss[1] >= 5.0 && ss[1] <= 8.0); };

//dynamics - 3 parameters
auto dynamics = [](const vec& x, const vec& u, const vec& w) -> vec {
    vec xx(dim_x);
    xx[0] = x[0] + 2*u[0]*cos(u[1]) + w[0];
    xx[1] = x[1] + 2*u[0]*sin(u[1]) + w[0];
    return xx;
};

/*
 ################################# MAIN FUNCTION ##############################################
 */

int main() {
    
    /* ###### create IMDP object ###### */
    IMDP mdp(dim_x,dim_u,dim_w);
    
    /* ###### create finite sets for the different spaces ###### */
    mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
    mdp.setInputSpace(is_lb, is_ub, is_eta);
    mdp.setDisturbSpace(ws_lb, ws_ub, ws_eta);
    
    /* ###### relabel states based on specification ###### */
    mdp.setTargetSpace(target_condition, true);
    
    /*###### save the files ######*/
    mdp.saveStateSpace();
    mdp.saveInputSpace();
    mdp.saveDisturbSpace();
    mdp.saveTargetSpace();
    
    /*###### set dynamics and noise ######*/
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::NORMAL);
    mdp.setStdDev(sigma);
    
    /* ###### calculate abstraction for target vectors ######*/
    /// each bound can be done seperately using:
    //mdp.minTargetTransitionVector();
    //mdp.maxTargetTransitionVector();
    ///or combined using:
    mdp.targetTransitionVectorBounds();
    
    /* ###### save target vectors ######*/
    mdp.saveMinTargetTransitionVector();
    mdp.saveMaxTargetTransitionVector();
    
    /* ###### calculate abstraction for avoid vectors ######*/
    mdp.minAvoidTransitionVector();
    mdp.maxAvoidTransitionVector();
    
    /* ###### save avoid vectors ######*/
    mdp.saveMinAvoidTransitionVector();
    mdp.saveMaxAvoidTransitionVector();
    
    /* ###### calculate abstraction for transition matrices ######*/
    /// each bound can be done seperately using:
    //mdp.minTransitionMatrix();
    //mdp.maxTransitionMatrix();
    ///or combined using:
    mdp.transitionMatrixBounds();
    
    /* ###### save transition matrices ######*/
    mdp.saveMinTransitionMatrix();
    mdp.saveMaxTransitionMatrix();
    
    /* ###### synthesize infinite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.infiniteHorizonReachControllerSorted(true);
    
    /* ###### synthesize finite horizon controller (true = pessimistic, false = optimistic) ######*/
    //mdp.finiteHorizonReachControllerSorted(true,10);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}
