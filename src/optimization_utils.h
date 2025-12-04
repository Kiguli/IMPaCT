/**
 * optimization_utils.h
 *
 * NLopt optimization helper utilities for IMPaCT
 *
 * Phase 3 Refactoring: Consolidates repeated NLopt setup code that appears
 * 120+ times in IMDP.cpp into reusable helper functions.
 *
 * This eliminates the 15-20 line optimization setup pattern that is duplicated
 * across all abstraction functions.
 */

#ifndef OPTIMIZATION_UTILS_H
#define OPTIMIZATION_UTILS_H

#include <armadillo>
#include <nlopt.hpp>
#include <vector>
#include <functional>
#include <iostream>

using namespace arma;
using namespace std;

namespace IMPaCT_Optimization {

/**
 * Helper function to run NLopt optimization with max objective
 *
 * @param state_start Starting state vector
 * @param eta Grid granularity vector
 * @param algo NLopt algorithm to use
 * @param cost_fn Cost function pointer
 * @param cost_data Pointer to cost function data struct
 * @return Optimized probability value (1.0 - minf for complementary problems)
 */
template<typename CostData>
inline double optimizeMaxObjective(
    const vec& state_start,
    const vec& eta,
    nlopt::algorithm algo,
    double (*cost_fn)(unsigned, const double*, double*, void*),
    CostData* cost_data
) {
    // Setup optimizer
    nlopt::opt opt(algo, state_start.size());

    // Set bounds: [state - eta/2, state + eta/2]
    vector<double> lb(state_start.size());
    vector<double> ub(state_start.size());
    for (size_t m = 0; m < state_start.size(); ++m) {
        lb[m] = state_start[m] - eta[m] / 2.0;
        ub[m] = state_start[m] + eta[m] / 2.0;
    }
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    // Set convergence tolerance
    opt.set_xtol_rel(1e-3);

    // Set objective function
    opt.set_max_objective(cost_fn, cost_data);

    // Optimize starting from current state
    vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
    double maxf;

    try {
        nlopt::result result = opt.optimize(initial_guess, maxf);
    } catch (exception& e) {
        cout << "nlopt failed: " << e.what() << endl;
        maxf = 0.0;  // Return 0 on failure
    }

    return maxf;
}

/**
 * Helper function to run NLopt optimization with min objective
 *
 * @param state_start Starting state vector
 * @param eta Grid granularity vector
 * @param algo NLopt algorithm to use
 * @param cost_fn Cost function pointer
 * @param cost_data Pointer to cost function data struct
 * @return Optimized probability value
 */
template<typename CostData>
inline double optimizeMinObjective(
    const vec& state_start,
    const vec& eta,
    nlopt::algorithm algo,
    double (*cost_fn)(unsigned, const double*, double*, void*),
    CostData* cost_data
) {
    // Setup optimizer
    nlopt::opt opt(algo, state_start.size());

    // Set bounds: [state - eta/2, state + eta/2]
    vector<double> lb(state_start.size());
    vector<double> ub(state_start.size());
    for (size_t m = 0; m < state_start.size(); ++m) {
        lb[m] = state_start[m] - eta[m] / 2.0;
        ub[m] = state_start[m] + eta[m] / 2.0;
    }
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    // Set convergence tolerance
    opt.set_xtol_rel(1e-3);

    // Set objective function
    opt.set_min_objective(cost_fn, cost_data);

    // Optimize starting from current state
    vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
    double minf;

    try {
        nlopt::result result = opt.optimize(initial_guess, minf);
    } catch (exception& e) {
        cout << "nlopt failed: " << e.what() << endl;
        minf = 0.0;  // Return 0 on failure
    }

    return minf;
}

/**
 * Helper function for complementary probability problems
 * (calculates 1.0 - max_objective)
 *
 * Common pattern for avoid transition calculations where we need
 * the complement of the maximum probability.
 */
template<typename CostData>
inline double optimizeComplementary(
    const vec& state_start,
    const vec& eta,
    nlopt::algorithm algo,
    double (*cost_fn)(unsigned, const double*, double*, void*),
    CostData* cost_data
) {
    double maxf = optimizeMaxObjective(state_start, eta, algo, cost_fn, cost_data);
    return 1.0 - maxf;
}

} // namespace IMPaCT_Optimization

#endif // OPTIMIZATION_UTILS_H
