// Run: make
//  ./BAS4D

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
const int dim_x = 4;
const int dim_u = 1;
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {19.0, 19.0, 30.0, 30.0};
const vec ss_ub = {21.0, 21.0, 36.0, 36.0};
const vec ss_eta = {0.5, 0.5, 1.0, 1.0};
// Inputs
const vec is_lb = {17.0};
const vec is_ub = {20.0};
const vec is_eta = {1.0};

//standard deviation of each dimension
const vec sigma = {sqrt(0.0774), sqrt(0.0774),sqrt(0.3872), sqrt(0.3098)};

//dynamics - 2 parameters
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    const mat A = {{0.6682, 0.0, 0.02632, 0.0}, {0.0, 0.6830, 0.0, 0.02096}, {1.0005, 0.0,  -0.000499, 0.0}, {0.0, 0.8004, 0.0, 0.1996}};
    const vec B = {0.1320, 0.1402, 0.0, 0.0};
    const vec Q = {3.4378, 2.9272, 13.0207, 10.4166};
    
    xx = A*x + B*u + Q;
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
    
    /*###### save the files ######*/
    mdp.saveStateSpace();
    mdp.saveInputSpace();
    
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
    
    /* ###### synthesize infinite horizon controller (true = pessimistic, false = optimistic) ######*/
    //mdp.infiniteHorizonSafeController(true);
    
    /* ###### synthesize finite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.finiteHorizonSafeController(true,10);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}


