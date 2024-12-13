/// To use this example, run and implement the 2Drobot-RU case study. 
/// Then copy to this folder the files is.h5, ss.h5, minttm.h5, maxttm.h5,
/// mintm.h5, maxtm.h5, minatm.h5 and maxatm.h5
 
// Run: make
// ./GPU


// Code: Ben Wooding 4 Jan 2024

#include <iostream>
#include <vector>
#include <functional>
#include "../../src/IMDP.h"
#include "../../src/MDP.h"
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
    mdp.loadStateSpace("ss.h5");
    mdp.loadInputSpace("is.h5");

    /* ###### load matrices and vectors ######*/
    mdp.loadMinTargetTransitionVector("minttm.h5");
    mdp.loadMaxTargetTransitionVector("minttm.h5");
    mdp.loadMinAvoidTransitionVector("minatm.h5");
    mdp.loadMaxAvoidTransitionVector("maxatm.h5");
    mdp.loadMinTransitionMatrix("mintm.h5");
    mdp.loadMaxTransitionMatrix("maxtm.h5");


    /* ###### storeMDP, choose optimization as pessimistic = true, or optimistic = false, also choose how many steps to save MDP from ######*/
    mdp.finiteHorizonReachControllerSortedStoreMDP(true,1);

    mdp.saveTransitionMatrix();
    mdp.saveAvoidTransitionVector();
    mdp.saveTargetTransitionVector();

    /* ###### save controller ######*/
    mdp.saveController();
    
    return 0;
}

