/// To use this example, run and implement the 2Drobot-RU case study. 
/// Then copy to this folder the files is.h5, ss.h5, minttm.h5, maxttm.h5,
/// mintm.h5, maxtm.h5, minatm.h5 and maxatm.h5
 
// Run: make
// ./load


// Code: Ben Wooding 4 Jan 2024

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

/*
 ################################# MAIN FUNCTION ##############################################
 */

int main() {
    
    /* ###### create IMDP object ###### */
    IMDP mdp(dim_x,dim_u,dim_w);
    
    /* ###### load the different spaces ###### */
    mdp.loadStateSpace();
    mdp.loadInputSpace();

    /* ###### load matrices and vectors ######*/
    mdp.loadMinTargetTransitionVector();
    mdp.loadMaxTargetTransitionVector();
    mdp.loadMinAvoidTransitionVector();
    mdp.loadMaxAvoidTransitionVector();
    mdp.loadMinTransitionMatrix();
    mdp.loadMaxTransitionMatrix();

    /* ###### synthesize infinite horizon controller ######*/
    mdp.infiniteHorizonReachController(true);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}

