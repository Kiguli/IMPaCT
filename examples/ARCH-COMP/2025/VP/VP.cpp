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
const int dim_u = 0;
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {-5.0, -5.0};
const vec ss_ub = {5.0, 5.0};
const vec ss_eta = {0.2, 0.2};

//dynamics - 2 parameters
auto dynamics = [](const vec& x) -> vec {
    vec xx(dim_x);
    xx[0] = x[0] + 0.1*x[1];
    xx[1] = x[1] + (-x[0] + (1 - x[0]) * (1 - x[0]) * x[1]) * 0.1; //(-x[0] + (1-pow(x[0],2))*x[1])*0.1;
    return xx;
};

auto target_condition = [](const vec& ss) { return (ss(0) >= -1.2 && ss(0) <= -0.9) && (ss(1) >= -2.9 && ss(1) <= -2.0);};

const vec sigma = {sqrt(0.04), sqrt(0.04)};

/// Custom PDF function for uniform distribution over [-0.02, 0.02] x [-0.02, 0.02]
double customPDF(double *x, size_t dim, void *params){

     // Bounds of the uniform distribution
     double lower = -0.02;
     double upper =  0.02;

     // Uniform density value
     double area = (upper - lower);  // dim=2 -> area = 0.04 * 0.04 = 0.0016
     return 1.0 / area;  // 625.0 for 2D
};

/*
 ################################# MAIN FUNCTION ##############################################
 */

int main() {
    
    /* ###### create IMDP object ###### */
    IMDP mdp(dim_x,dim_u,dim_w);
    
    /* ###### change algorithm ###### */
    //mdp.setAlgorithm(nlopt::LN_COBYLA);
    
    /* ###### create finite sets for the different spaces ###### */
    mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
    
    /*###### set dynamics and noise ######*/
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::NORMAL); //(seems to work okay, but takes a long time to run!)
    mdp.setStdDev(sigma);
    //mdp.setNoise(NoiseType::CUSTOM);
    //mdp.setCustomDistribution(customPDF,1000);
    
    /*#### set target space #######*/
    mdp.setTargetSpace(target_condition, true);
    
    /* ###### calculate abstraction for target vectors ######*/
    mdp.targetTransitionVectorBounds();
    
    /* ###### calculate abstraction for avoid vectors ######*/
    mdp.minAvoidTransitionVector();
    mdp.maxAvoidTransitionVector();

    /* ###### calculate abstraction for transition matrices ######*/
    mdp.transitionMatrixBounds();
    
    /* ###### synthesize infinite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.infiniteHorizonReachControllerSorted(true);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}


