// Run: make
//      ./safe

// Created by Ben Wooding 8 Jan 2024

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
const int dim_x = 2;
const int dim_u = 0;
const int dim_w = 0;

/*
 ################################# MAIN FUNCTION ##############################################
 */

int main() {
    
    /* ###### create IMDP object ###### */
    IMDP mdp(dim_x,dim_u,dim_w);
    
    /*###### load the files ######*/
    mdp.loadStateSpace("ss.h5");
    mdp.loadMinAvoidTransitionVector("minatm.h5");
    mdp.loadMaxAvoidTransitionVector("maxatm.h5");
    mdp.loadMinTransitionMatrix("mintm.h5");
    mdp.loadMaxTransitionMatrix("maxtm.h5");
    
    /*###### Run over infinite horizon, absorbing state so the two bounds will not converge ######*/
    mdp.infiniteHorizonSafeController(true);
    
    /*###### Take steps given from previous function to get convergent solution ######*/
    mdp.finiteHorizonSafeController(true,67);
    
    /*###### Save the controller ######*/
    mdp.saveController();
    
    return 0;
}


