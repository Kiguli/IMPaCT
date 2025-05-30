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
const int dim_x = 6;
const int dim_u = 2;
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {-0.5, -0.5,-10,-5,-3.14,-3.14};
const vec ss_ub = {0.5, 0.5,4,4,3.14, 3.14};
const vec ss_eta = {0.5, 0.5,1,1,3.14,3.14};
// Inputs
const vec is_lb = {0.5,0.5};
const vec is_ub = {3.5,3.5};
const vec is_eta = {1.5, 1.5};

//standard deviation of each dimension
const vec sigma = {0,0,0,0,0.001,0.001};

// logical expression for target region and avoid region
auto target_condition = [](const vec& ss) { return (ss(0) >= -0.5 && ss(0) <= 0.5) && (ss(1) >= -0.5 && ss(1) <= -0.5) && (ss(2) >= -10 && ss(2) <= -10) && (ss(3) >= -5 && ss(3) <= -5) && (ss(4) >= -3.14 && ss(4) <= 3.14) && (ss(5) >= -3.14 && ss(5) <= 3.14);};
auto avoid_condition = [](const vec& ss) { return (ss(0) >= -0.5 && ss(0) <= 0.5) && (ss(1) >= -0.5 && ss(1) <= -0.5) && (ss(2) >= -4 && ss(2) <= 4) && (ss(3) >= 2.5 && ss(3) <= 4) && (ss(4) >= -3.14 && ss(4) <= 3.14) && (ss(5) >= -3.14 && ss(5) <= 3.14);};

//dynamics - 2 parameters
const auto dynamics = [](const vec& x, const vec& u) -> vec {
    float Ts = 0.1;
    float M = 700;
    float D = 1.2;
    float a = 0.35;
    float b = 0.15;
    float d = 0.1;
    float Afy = 199;
    float Izz = 394;
    float pi = 3.14;
    float Cgamma = 22000;
    vec xx = x;
    
    xx[0] = x[0] + (Ts/Izz)*(Afy*sin(x[5]+(pi/2))*a + 2*Cgamma*tan((2*(x[1]-b*x[0])/(u[0]+u[1])))*b);
    xx[1] = x[1] + (Ts/M)*(Afy*sin(x[5]+(pi/2))*a + 2*Cgamma*tan((2*(x[1]-b*x[0])/(u[0]+u[1]))))-(Ts*(u[0]+u[1]))/2*x[0];
    xx[2] = x[2] + (Ts*(u[0]+u[1])/2)*cos(x[4]);
    xx[3] = x[3] + (Ts*(u[0]+u[1])/2)*sin(x[4]);
    xx[4] = x[4] + Ts*((u[0]+u[1])/(2*D)+x[0]);
    xx[5] = x[5] - Ts*(u[0]+u[1])/(2*d)*cos(x[5])-(Ts*(u[0]+u[1])*(d+sqrt(D*D + (a+b)*(a+b))*sin(x[5])))/(2*D*d);
    
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
    //mdp.infiniteHorizonReachControllerSorted(true);
    
    /* ###### synthesize finite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.finiteHorizonReachControllerSorted(true,200);
    
    /* ###### save controller ######*/
    //mdp.saveController();
    
    return 0;
}
