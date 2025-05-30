// Run: make
// Then: ./AS

// Code: Ben Wooding 8 Apr 2025

#include <iostream>
#include <vector>
#include <functional>
#include "../../../../src/IMDP.h"
#include <armadillo>
#include <chrono>

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
const vec ss_lb = {1,0, 0};
const vec ss_ub = {6,10, 10};
const vec ss_eta = {0.25,1,1};
// Inputs
const vec is_lb = {0,0};
const vec is_ub = {7,30};
const vec is_eta = {1,30};

//standard deviation of each dimension
const vec sigma = {sqrt(0.001), sqrt(0.001), sqrt(0.001)};

// logical expression for target region
auto target_condition = [](const vec& ss) { return (ss[0] >= 4.0 && ss[0] <= 6.0) && (ss[1] >= 8.0 && ss[1] <= 10.0) && (ss[2] >= 8.0 && ss[2] <= 10.0); };

//dynamics - 3 parameters
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    xx[0] = 0.8192*x[0] + 0.03412*x[1] +0.01265*x[2] + 0.01883*(u[0] + u[1]);
    xx[1] = 0.01646*x[0] + 0.9822*x[1] +0.0001*x[2] + 0.0002*(u[0] + u[1]);
    xx[2] = 0.0009*x[0] + 0.00002*x[1] +0.9989*x[2] + 0.00001*(u[0] + u[1]);
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
    //mdp.setDisturbSpace(ws_lb, ws_ub, ws_eta);
    
    /* ###### relabel states based on specification ###### */
    mdp.setTargetSpace(target_condition, true);
    
    /*###### save the files ######*/
    //mdp.saveStateSpace();
    //mdp.saveInputSpace();
    //mdp.saveDisturbSpace();
    //mdp.saveTargetSpace();
    
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
    //mdp.saveMinTargetTransitionVector();
    //mdp.saveMaxTargetTransitionVector();
    
    /* ###### calculate abstraction for avoid vectors ######*/
    mdp.minAvoidTransitionVector();
    mdp.maxAvoidTransitionVector();
    
    /* ###### save avoid vectors ######*/
    //mdp.saveMinAvoidTransitionVector();
    //mdp.saveMaxAvoidTransitionVector();
    
    /* ###### calculate abstraction for transition matrices ######*/
    /// each bound can be done seperately using:
    //mdp.minTransitionMatrix();
    //mdp.maxTransitionMatrix();
    ///or combined using:
    mdp.transitionMatrixBounds();
    
    /* ###### save transition matrices ######*/
    //mdp.saveMinTransitionMatrix();
    //mdp.saveMaxTransitionMatrix();
    
    /* ###### synthesize infinite horizon controller (true = pessimistic, false = optimistic) ######*/
    //mdp.infiniteHorizonReachControllerSorted(true);
    
    /* ###### synthesize finite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.finiteHorizonReachControllerSorted(true,10);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}
