// Run: make
//  ./BAS7D

// Created by Ben Wooding 6 Jan 2024

#include <iostream>
#include <vector>
#include <functional>
#include "../../src/IMDP.h"
#include <armadillo>
#include <chrono>

using namespace arma;
using namespace std;

/*
 ################################# PARAMETERS ###############################################
 */

// Set the dimensions
const int dim_x = 7;
const int dim_u = 0;
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {-0.5, -0.5, -0.5, -0.5,-0.5,-0.5,-0.5};
const vec ss_ub = {0.5, 0.5, 0.5, 0.5,0.5,0.5,0.5};
const vec ss_eta = {0.05, 0.05, 0.5, 0.5,0.5,0.5,0.5};

//standard deviation of each dimension
const vec sigma = {sqrt(1/51.2821), sqrt(1/50.0),sqrt(1/21.7865), sqrt(1/23.5294),sqrt(1/25.1889),sqrt(1/26.5252),sqrt(1/91.7431)};

//dynamics - 1 parameter
auto dynamics = [](const vec& x) -> vec {
    vec xx(dim_x);
    const mat A = {{0.9678, 0, 0.0036, 0, 0.0036, 0, 0.0036}, {0, 0.9682, 0, 0.0034, 0, 0.0034, 0.0034}, {0.0106, 0, 0.9494, 0, 0, 0, 0}, {0, 0.0097, 0, 0.9523, 0, 0, 0}, {0.0106, 0, 0, 0, 0.9494, 0, 0}, {0, 0.0097, 0, 0, 0, 0.9523, 0}, {0.0106, 0.0097, 0, 0, 0, 0, 0.9794}};

    xx = A*x;
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
    
    /*###### save the files ######*/
    mdp.saveStateSpace();
    
    /*###### set dynamics and noise ######*/
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::NORMAL);
    mdp.setStdDev(sigma);
    
    /* ###### calculate abstraction for avoid vectors ######*/
    mdp.minAvoidTransitionVector();
    mdp.maxAvoidTransitionVector();
    
    /* ###### save avoid vectors ######*/
    mdp.saveMinAvoidTransitionVector();
    mdp.saveMaxAvoidTransitionVector();
    
    /* ###### calculate abstraction for transition matrices ######*/
    mdp.transitionMatrixBounds();
    
    /* ###### save transition matrices ######*/
    mdp.saveMinTransitionMatrix();
    mdp.saveMaxTransitionMatrix();
    
    /* ###### verification over infinite horizon (true = pessimistic, false = optimistic) ######*/
    mdp.infiniteHorizonSafeController(true);
    
    /* ###### verification over finite horizon (true = pessimistic, false = optimistic) ######*/
    //mdp.finiteHorizonSafeController(true,10);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}

