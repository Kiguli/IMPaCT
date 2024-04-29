// Run: make
// ./roomTemp5D

// Created by Ben Wooding 6 Jan 2024

#include <iostream>
#include <vector>
#include <functional>
#include "../../src/IMDP.h"
#include <chrono>
#include <armadillo>

using namespace std;
using namespace arma;

/*
 ################################# PARAMETERS ###############################################
 */

// Set the dimensions
const int dim_x = 5;
const int dim_u = 2;
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {19, 19,19,19,19};
const vec ss_ub = {21,21,21,21,21};
const vec ss_eta = {0.4, 0.4, 0.4,0.4,0.4};
// Inputs
const vec is_lb = {0,0};
const vec is_ub = {1.0,1.0};
const vec is_eta = {0.2, 0.2};

//standard deviation of each dimension
const vec sigma = {sqrt(1/100), sqrt(1/100), sqrt(1/100),sqrt(1/100),sqrt(1/100)};

//dynamics - 2 parameters
const auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    float eta = 0.30f, beta = 0.022f, gamma = 0.05f, a = 1.0f - 2.0f*0.30f - 0.022f, T_h = 50.0f, T_e = -1.0f;
    xx(0) = (a - gamma*u(0))*x(0) + eta*(x(1) + x(4)) + gamma*T_h*u(0) + beta*T_e;
    xx(1) = a*x(1) + eta*(x(0) + x(2)) + beta*T_e;
    xx(2) = (a - gamma*u(1))*x(2) + eta*(x(3) + x(1)) + gamma*T_h*u(1) + beta*T_e;
    xx(3) = a*x(3) + eta*(x(2)+x(4))+beta*T_e;
    xx(4) = a*x(4) + eta*(x(3)+x(0))+beta*T_e;
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
    mdp.infiniteHorizonSafeControllerSorted(true);
    
    /* ###### synthesize finite horizon controller (true = pessimistic, false = optimistic) ######*/
    //mdp.finiteHorizonSafeControllerSorted(true,10);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}


