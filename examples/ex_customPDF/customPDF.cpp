// Adjust customPDF function in custom.cpp from src directory
// Run: make
//      ./customPDF


// Code: Ben Wooding 6 Jan 2024

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
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {-10, -10};
const vec ss_ub = {10, 10};
const vec ss_eta = {1, 1};
// Inputs
const vec is_lb = {-1,-1};
const vec is_ub = {1, 1};
const vec is_eta = {0.2, 0.2};

// logical expression for target region
auto target_condition = [](const vec& ss) { return (ss[0] >= 5.0 && ss[0] <= 8.0) && (ss[1] >= 5.0 && ss[1] <= 8.0); };

//dynamics - 2 parameters
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    xx[0] = x[0] + 2*u[0]*cos(u[1]);
    xx[1] = x[1] + 2*u[0]*sin(u[1]);
    return xx;
};

/// Custom PDF function, change this to the PDF function desired that will be integrated over with Monte Carlo integration
double customPDF(double *x, size_t dim, void *params){
     //custom PDF parameters that are passed in (not all need to be used)
     customParams *p = reinterpret_cast<customParams*>(params);
     vec mean = p-> mean;
     vec state_start = p->state_start;
     function<vec(const vec&,const vec&)> dynamics2 = p-> dynamics2;
     vec input = p-> input;
     vec lb = p-> lb;
     vec ub = p-> ub;
     vec eta = p-> eta;
     /* Other parameters of struct unused in the ex_custom_distribution example:*/
     //function<vec(const vec&)> dynamics1 = p->dynamics1;
     //function<vec(const vec&,const vec&)> dynamics3 = p-> dynamics3;
     //vec disturb = p-> disturb;

     //multivariate normal PDF example:
     double cov_det = 0.5625;
     mat inv_cov = {{1.3333, 0},{0, 1.3333}};
     double norm = 1.0 / (pow(2 * M_PI, dim / 2.0) * sqrt(cov_det));

     double exponent = 0.0;
     for (size_t i = 0; i < dim; ++i) {
         for (size_t j = 0; j < dim; ++j) {
             exponent -= 0.5 * (x[i] - mean[i]) * (x[j] - mean[j]) * inv_cov(i,j);
         }
     }
     return norm * exp(exponent);
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
    
    /* ###### relabel states based on specification ###### */
    mdp.setTargetSpace(target_condition, true);
    
    /*###### save the files ######*/
    mdp.saveStateSpace();
    mdp.saveInputSpace();
    mdp.saveTargetSpace();
    
    /*###### set dynamics and noise ######*/
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::CUSTOM);
    mdp.setCustomDistribution(customPDF,1000);
    
    /* ###### calculate abstraction for target vectors ######*/
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
    mdp.transitionMatrixBounds();
    
    /* ###### save transition matrices ######*/
    mdp.saveMinTransitionMatrix();
    mdp.saveMaxTransitionMatrix();
    
    /* ###### synthesize infinite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.infiniteHorizonReachController(true);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}


