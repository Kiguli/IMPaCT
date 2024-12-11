#ifndef IMDP_H
#define IMDP_H

#include "MDP.h"
#include <armadillo>
#include <nlopt.hpp>

using namespace arma;
using namespace std;

/* IMDP class with is a child of MDP class*/

class IMDP: public MDP {
    
    /* IMDP Protected Variables*/
protected:
    /// Transition Matrices and Vectors
    vec minTargetM;
    vec maxTargetM;
    vec minAvoidM;
    vec maxAvoidM;
    mat minTransitionM;
    mat maxTransitionM;

    ///check for if you want to store Q values in synthesis iterations
    bool storeMDP;

    /// Controller
    mat controller;
    
    ///Algorithm used for nonlinear optimization
    nlopt::algorithm algo = nlopt::LN_SBPLX;
    
    /* IMDP Public Functions*/
public:
    /// Inherit functions from parent
    using MDP::MDP;
    /// Destructor
    ~IMDP();
    
    /// Set the Nonlinear Optimization Algorithm (choice of others found at: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/, e.g. LN_COBYLA)
    void setAlgorithm(nlopt::algorithm alg);

    void trackMDP(bool store);
    
    /// Matrix and Vector Abstraction Functions
    void minTransitionMatrix();
    void maxTransitionMatrix();
    void minTargetTransitionVector();
    void maxTargetTransitionVector();
    void minAvoidTransitionVector();
    void maxAvoidTransitionVector();
    
    /// Low-Cost Abstraction Functions
    void transitionMatrixBounds();
    void targetTransitionVectorBounds();
    
    /// Synthesis Functions for Infinite and Finite Time Horizons
    void infiniteHorizonReachController(bool IMDP_lower);
    void infiniteHorizonSafeController(bool IMDP_lower);
    void finiteHorizonReachController(bool IMDP_lower, size_t timeHorizon);
    void finiteHorizonSafeController(bool IMDP_lower, size_t timeHorizon);
    /// Sorted Versions
    void infiniteHorizonReachControllerSorted(bool IMDP_lower);
    void finiteHorizonReachControllerSorted(bool IMDP_lower, size_t timeHorizon);
    void infiniteHorizonSafeControllerSorted(bool IMDP_lower);
    void finiteHorizonSafeControllerSorted(bool IMDP_lower, size_t timeHorizon);


    /// Functions to Save the Vectors, Matrices and Controller
    void saveMinTargetTransitionVector();
    void saveMinAvoidTransitionVector();
    void saveMinTransitionMatrix();
    void saveMaxTargetTransitionVector();
    void saveMaxAvoidTransitionVector();
    void saveMaxTransitionMatrix();
    void saveController();
    
    /// Functions to Load the Vectors, Matrices and Controller
    void loadMinTargetTransitionVector(string filename);
    void loadMinAvoidTransitionVector(string filename);
    void loadMinTransitionMatrix(string filename);
    void loadMaxTargetTransitionVector(string filename);
    void loadMaxAvoidTransitionVector(string filename);
    void loadMaxTransitionMatrix(string filename);
    void loadController(string filename);
};

#endif
