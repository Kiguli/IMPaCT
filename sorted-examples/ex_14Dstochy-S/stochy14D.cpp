//Run: make
// ./stochy14D

// Created by Ben Wooding 8 Jan 2024

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
const int dim_x = 14;
const int dim_u = 0;
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {-0.5, -0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5};
const vec ss_ub = {0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
const vec ss_eta = {1,1,1,1,1,1,1,1,1,1,1,1,1,1};

//standard deviation of each dimension
const vec sigma = {sqrt(0.2), sqrt(0.2), sqrt(0.2),sqrt(0.2),sqrt(0.2),sqrt(0.2),sqrt(0.2),sqrt(0.2),sqrt(0.2),sqrt(0.2),sqrt(0.2),sqrt(0.2),sqrt(0.2),sqrt(0.2)};

//dynamics - 1 parameter
auto dynamics = [](const vec& x) -> vec {
    vec xx(dim_x);
    xx[0] = 0.8*x[0];
    xx[1] = 0.8*x[1];
    xx[2] = 0.8*x[2];
    xx[3] = 0.8*x[3];
    xx[4] = 0.8*x[4];
    xx[5] = 0.8*x[5];
    xx[6] = 0.8*x[6];
    xx[7] = 0.8*x[7];
    xx[8] = 0.8*x[8];
    xx[9] = 0.8*x[9];
    xx[10] = 0.8*x[10];
    xx[11] = 0.8*x[11];
    xx[12] = 0.8*x[12];
    xx[13] = 0.8*x[13];
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
    
    /* ###### synthesize infinite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.infiniteHorizonSafeControllerSorted(true);
    
    /* ###### synthesize finite horizon controller (true = pessimistic, false = optimistic) ######*/
    //mdp.finiteHorizonSafeControllerSorted(true,10);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    
    return 0;
}


