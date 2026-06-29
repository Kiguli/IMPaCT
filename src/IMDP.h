#ifndef IMDP_H
#define IMDP_H

#include "MDP.h"
#include <armadillo>
#include <nlopt.hpp>

using namespace arma;
using namespace std;

/// Convergence method for the INFINITE-horizon robust value iteration.
///   IntervalIteration (default) — sound interval iteration: iterate from 0 (lower)
///     and from 1 (upper) and stop when the gap |upper-lower| < epsilon. Rigorous
///     two-sided bracket, but on weakly-contracting / end-component models it can be
///     slow or (with end components) never close the gap (see ISSUE-0003).
///   ValueIteration — the method peer tools (PRISM / Storm / IntervalMDP.jl) use:
///     plain value iteration with RESIDUAL stopping (stop when the per-sweep change
///     of the iterates < epsilon). Far fewer sweeps and it converges even with end
///     components (the from-0 iterate -> least fixed point = robust value), but the
///     residual stopping is not a sound two-sided certificate.
enum class IterationMethod { IntervalIteration, ValueIteration };

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



    /// Controller
    mat controller;
    
    ///Algorithm used for nonlinear optimization
    nlopt::algorithm algo = nlopt::LN_SBPLX;

    ///Convergence method for infinite-horizon synthesis (see IterationMethod above)
    IterationMethod iterMethod = IterationMethod::IntervalIteration;

    // Internal implementation helpers for controller synthesis
    void infiniteHorizonControllerImpl(bool IMDP_lower, bool is_reach);
    void finiteHorizonControllerImpl(bool IMDP_lower, size_t timeHorizon, bool is_reach);

    // Internal implementation helpers for transition abstractions
    void transitionMatrixImpl(mat& output, bool is_min);
    void targetTransitionVectorImpl(vec& output, bool is_min);
    void avoidTransitionVectorImpl(vec& output, bool is_min);

    /* IMDP Public Functions*/
public:
    /// Inherit functions from parent
    using MDP::MDP;
    /// Destructor
    ~IMDP();
    
    /// Set the Nonlinear Optimization Algorithm (choice of others found at: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/, e.g. LN_COBYLA)
    void setAlgorithm(nlopt::algorithm alg);

    /// Select the infinite-horizon convergence method: IterationMethod::IntervalIteration
    /// (default, sound bracket) or IterationMethod::ValueIteration (peer-style residual VI).
    void setIterationMethod(IterationMethod m);

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

    void finiteHorizonReachControllerSortedStoreMDP(bool IMDP_lower, size_t timeHorizon);


    /// Functions to Save the Vectors, Matrices and Controller
    void saveMinTargetTransitionVector();
    void saveMinAvoidTransitionVector();
    void saveMinTransitionMatrix();
    void saveMaxTargetTransitionVector();
    void saveMaxAvoidTransitionVector();
    void saveMaxTransitionMatrix();
    void saveController();

    /// Export the abstracted Interval MDP to the neutral .imdp exchange format
    /// (src/imdp_io.h) for cross-tool comparison against IntervalMDP.jl / Storm /
    /// PRISM. Cells become states 0..state_space_size-1; an accepting target sink
    /// and a dead avoid sink (capturing the explicit avoid region plus mass leaving
    /// the bounded domain) are appended so each (state,action) row is a sound
    /// interval distribution. Call AFTER the transition/target/avoid abstractions.
    void exportIMDP(const string& filename);

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
