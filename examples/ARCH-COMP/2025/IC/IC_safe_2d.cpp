// Run: make
//  ./BAS4D

// Created by Ben Wooding 6 Jan 2024

#include <iostream>
#include <vector>
#include <functional>
#include "../../../../src/IMDP.h"
#include <armadillo>
#include <chrono>

using namespace arma;
using namespace std;

/*
 ################################# PARAMETERS ###############################################
 */

// Set the dimensions
const int dim_x = 2;
const int dim_u = 1;
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {-10.0, -10.0};
const vec ss_ub = {10.0, 10.0};
const vec ss_eta = {0.5, 0.5};
// Inputs
const vec is_lb = {-1.0};
const vec is_ub = {1.0};
const vec is_eta = {0.5};

const vec sigma = {sqrt(0.01), sqrt(0.01)};
//dynamics - 2 parameters
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    float Ns = 0.1;
    xx[0] = x[0] + Ns*x[1] + (Ns*Ns)/2.0*u[0];
    xx[1] = x[1] + Ns*u[0];
    return xx;
};

//auto target_condition = [](const vec& ss) { return (ss(0) >= -8.0 && ss(0) <= 8.0) && (ss(1) >= -8.0 && ss(1) <= 8.0);};

/*
 ################################# MAIN FUNCTION ##############################################
 */

int main() {
    
    /* ###### create IMDP object ###### */
    IMDP mdp(dim_x,dim_u,dim_w);
    
    /* ###### create finite sets for the different spaces ###### */
    mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
    mdp.setInputSpace(is_lb, is_ub, is_eta);
    
    /*###### set dynamics and noise ######*/
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::NORMAL);
    mdp.setStdDev(sigma);
    
    /*#### set target space #######*/
    //mdp.setTargetSpace(target_condition, true);
    
    /* ###### calculate abstraction for target vectors ######*/
    //mdp.targetTransitionVectorBounds();
    
    /* ###### calculate abstraction for avoid vectors ######*/
    mdp.minAvoidTransitionVector();
    mdp.maxAvoidTransitionVector();

    /* ###### calculate abstraction for transition matrices ######*/
    mdp.transitionMatrixBounds();
    
    /* ###### synthesize finite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.finiteHorizonSafeControllerSorted(true,5);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}


