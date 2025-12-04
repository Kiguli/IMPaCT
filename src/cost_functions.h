/**
 * cost_functions.h
 *
 * Template-based cost function framework for IMPaCT
 *
 * Phase 2 Refactoring: Consolidates 18 cost function structs and functions
 * into a unified template-based system that eliminates code duplication.
 *
 * This replaces lines 136-839 in IMDP.cpp with a clean, maintainable design.
 */

#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <armadillo>
#include <functional>
#include <vector>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_rng.h>
#include <cmath>

using namespace arma;
using namespace std;

namespace IMPaCT_CostFunctions {

/* ============================================================================
 * NOISE MODEL POLICIES
 * ============================================================================ */

/// Multivariate normal noise distribution parameters
struct multivariateNormalParams {
    vec mean;
    mat inv_cov;
    double det;
};

/// Multivariate normal noise distribution PDF
inline double multivariateNormalPDF(double *x, size_t dim, void *params) {
    multivariateNormalParams *p = reinterpret_cast<multivariateNormalParams*>(params);
    double norm = 1.0 / (pow(2 * M_PI, dim / 2.0) * sqrt(p->det));
    double exponent = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            exponent -= 0.5 * (x[i] - p->mean[i]) * (x[j] - p->mean[j]) * p->inv_cov(i,j);
        }
    }
    return norm * exp(exponent);
}

/// Closed form integral for 1D normal distribution CDF
inline double normal1DCDF(const double& x0, const double& x1, const double& mu, const double& sigma) {
    double cdf_x0 = 0.5 * (1 + erf((x0 - mu) / (sigma * sqrt(2))));
    double cdf_x1 = 0.5 * (1 + erf((x1 - mu) / (sigma * sqrt(2))));
    return cdf_x1 - cdf_x0;
}

/// Noise model policy for diagonal normal distribution
struct DiagonalNormal {
    vec sigma;

    double computeProbability(const vec& mu, const vec& eta, const vec& bounds_lower, const vec& bounds_upper) const {
        double probability_product = 1.0;
        for (size_t m = 0; m < mu.n_rows; ++m) {
            double x0 = bounds_lower[m];
            double x1 = bounds_upper[m];
            double probability = normal1DCDF(x0, x1, mu[m], sigma[m]);
            probability_product *= probability;
        }
        return probability_product;
    }
};

/// Noise model policy for off-diagonal (multivariate) normal distribution
struct OffDiagonalNormal {
    double dim;
    mat inv_cov;
    double det;
    size_t samples;

    double computeProbability(const vec& mu, const vec& eta, const vec& bounds_lower, const vec& bounds_upper) const {
        multivariateNormalParams params;
        params.mean = mu;
        params.inv_cov = inv_cov;
        params.det = det;

        gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
        gsl_monte_function F;
        F.f = &multivariateNormalPDF;
        F.dim = mu.n_rows;
        F.params = const_cast<multivariateNormalParams*>(&params);

        vector<double> lb_vec, ub_vec;
        for (size_t m = 0; m < mu.n_rows; ++m) {
            lb_vec.push_back(bounds_lower[m]);
            ub_vec.push_back(bounds_upper[m]);
        }
        double* lb = lb_vec.data();
        double* ub = ub_vec.data();
        double result, error;

        gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
        gsl_monte_vegas_integrate(&F, lb, ub, dim, samples, rng, s, &result, &error);
        gsl_monte_vegas_free(s);
        gsl_rng_free(rng);

        return result;
    }
};

/* ============================================================================
 * SPACE VARIANT POLICIES
 * ============================================================================ */

/// Regular space variant (single state_end point)
struct RegularSpace {
    static void computeBounds(const vec& state_ref, const vec& eta,
                             vec& bounds_lower, vec& bounds_upper) {
        for (size_t m = 0; m < state_ref.n_rows; ++m) {
            bounds_lower[m] = state_ref[m] - eta[m] / 2.0;
            bounds_upper[m] = state_ref[m] + eta[m] / 2.0;
        }
    }
};

/// Full space variant (range with lb/ub)
struct FullSpace {
    static void computeBounds(const vec& lb, const vec& ub, const vec& eta,
                             vec& bounds_lower, vec& bounds_upper) {
        for (size_t m = 0; m < lb.n_rows; ++m) {
            bounds_lower[m] = lb[m] - eta[m] / 2.0;
            bounds_upper[m] = ub[m] + eta[m] / 2.0;
        }
    }
};

/* ============================================================================
 * TEMPLATED COST FUNCTION DATA STRUCTURES
 * ============================================================================ */

/// Template cost function data for 1 parameter (state only)
template<typename NoiseModel, typename SpaceVariant>
struct CostFunctionData1 {
    vec state_ref;  // state_end for Regular, state_start for Full
    vec lb, ub;     // Only used for Full variant
    vec eta;
    NoiseModel noise_model;
    function<vec(const vec&)> dynamics;

    vec computeBounds(bool is_lower) const {
        vec bounds(state_ref.n_rows);
        if constexpr (is_same_v<SpaceVariant, RegularSpace>) {
            for (size_t m = 0; m < state_ref.n_rows; ++m) {
                bounds[m] = state_ref[m] + (is_lower ? -eta[m] / 2.0 : eta[m] / 2.0);
            }
        } else {  // FullSpace
            for (size_t m = 0; m < state_ref.n_rows; ++m) {
                bounds[m] = (is_lower ? lb[m] : ub[m]) + (is_lower ? -eta[m] / 2.0 : eta[m] / 2.0);
            }
        }
        return bounds;
    }
};

/// Template cost function data for 2 parameters (state + input/disturb)
template<typename NoiseModel, typename SpaceVariant>
struct CostFunctionData2 {
    vec state_ref;
    vec lb, ub;
    vec second;  // input or disturb
    vec eta;
    NoiseModel noise_model;
    function<vec(const vec&, const vec&)> dynamics;

    vec computeBounds(bool is_lower) const {
        vec bounds(state_ref.n_rows);
        if constexpr (is_same_v<SpaceVariant, RegularSpace>) {
            for (size_t m = 0; m < state_ref.n_rows; ++m) {
                bounds[m] = state_ref[m] + (is_lower ? -eta[m] / 2.0 : eta[m] / 2.0);
            }
        } else {
            for (size_t m = 0; m < state_ref.n_rows; ++m) {
                bounds[m] = (is_lower ? lb[m] : ub[m]) + (is_lower ? -eta[m] / 2.0 : eta[m] / 2.0);
            }
        }
        return bounds;
    }
};

/// Template cost function data for 3 parameters (state + input + disturb)
template<typename NoiseModel, typename SpaceVariant>
struct CostFunctionData3 {
    vec state_ref;
    vec lb, ub;
    vec input;
    vec disturb;
    vec eta;
    NoiseModel noise_model;
    function<vec(const vec&, const vec&, const vec&)> dynamics;

    vec computeBounds(bool is_lower) const {
        vec bounds(state_ref.n_rows);
        if constexpr (is_same_v<SpaceVariant, RegularSpace>) {
            for (size_t m = 0; m < state_ref.n_rows; ++m) {
                bounds[m] = state_ref[m] + (is_lower ? -eta[m] / 2.0 : eta[m] / 2.0);
            }
        } else {
            for (size_t m = 0; m < state_ref.n_rows; ++m) {
                bounds[m] = (is_lower ? lb[m] : ub[m]) + (is_lower ? -eta[m] / 2.0 : eta[m] / 2.0);
            }
        }
        return bounds;
    }
};

/* ============================================================================
 * TEMPLATED COST FUNCTIONS
 * ============================================================================ */

/// Cost function for 1 parameter
template<typename NoiseModel, typename SpaceVariant>
double costFunction1(unsigned n, const double* x, double* grad, void* my_func_data) {
    auto* data = static_cast<CostFunctionData1<NoiseModel, SpaceVariant>*>(my_func_data);

    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)));

    vec bounds_lower = data->computeBounds(true);
    vec bounds_upper = data->computeBounds(false);

    return data->noise_model.computeProbability(mu, data->eta, bounds_lower, bounds_upper);
}

/// Cost function for 2 parameters
template<typename NoiseModel, typename SpaceVariant>
double costFunction2(unsigned n, const double* x, double* grad, void* my_func_data) {
    auto* data = static_cast<CostFunctionData2<NoiseModel, SpaceVariant>*>(my_func_data);

    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->second);

    vec bounds_lower = data->computeBounds(true);
    vec bounds_upper = data->computeBounds(false);

    return data->noise_model.computeProbability(mu, data->eta, bounds_lower, bounds_upper);
}

/// Cost function for 3 parameters
template<typename NoiseModel, typename SpaceVariant>
double costFunction3(unsigned n, const double* x, double* grad, void* my_func_data) {
    auto* data = static_cast<CostFunctionData3<NoiseModel, SpaceVariant>*>(my_func_data);

    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->input, data->disturb);

    vec bounds_lower = data->computeBounds(true);
    vec bounds_upper = data->computeBounds(false);

    return data->noise_model.computeProbability(mu, data->eta, bounds_lower, bounds_upper);
}

/* ============================================================================
 * TYPE ALIASES FOR BACKWARD COMPATIBILITY
 * ============================================================================ */

// 1 parameter variants
using CostFunctionDataNormaldiagonal1 = CostFunctionData1<DiagonalNormal, RegularSpace>;
using CostFunctionDataNormaloffdiagonal1 = CostFunctionData1<OffDiagonalNormal, RegularSpace>;
using CostFunctionDataNormaldiagonal1Full = CostFunctionData1<DiagonalNormal, FullSpace>;
using CostFunctionDataNormaloffdiagonal1Full = CostFunctionData1<OffDiagonalNormal, FullSpace>;

// 2 parameter variants
using CostFunctionDataNormaldiagonal2 = CostFunctionData2<DiagonalNormal, RegularSpace>;
using CostFunctionDataNormaloffdiagonal2 = CostFunctionData2<OffDiagonalNormal, RegularSpace>;
using CostFunctionDataNormaldiagonal2Full = CostFunctionData2<DiagonalNormal, FullSpace>;
using CostFunctionDataNormaloffdiagonal2Full = CostFunctionData2<OffDiagonalNormal, FullSpace>;

// 3 parameter variants
using CostFunctionDataNormaldiagonal3 = CostFunctionData3<DiagonalNormal, RegularSpace>;
using CostFunctionDataNormaloffdiagonal3 = CostFunctionData3<OffDiagonalNormal, RegularSpace>;
using CostFunctionDataNormaldiagonal3Full = CostFunctionData3<DiagonalNormal, FullSpace>;
using CostFunctionDataNormaloffdiagonal3Full = CostFunctionData3<OffDiagonalNormal, FullSpace>;

// Function pointer aliases
constexpr auto costFunctionNormaldiagonal1 = costFunction1<DiagonalNormal, RegularSpace>;
constexpr auto costFunctionNormaloffdiagonal1 = costFunction1<OffDiagonalNormal, RegularSpace>;
constexpr auto costFunctionNormaldiagonal1Full = costFunction1<DiagonalNormal, FullSpace>;
constexpr auto costFunctionNormaloffdiagonal1Full = costFunction1<OffDiagonalNormal, FullSpace>;

constexpr auto costFunctionNormaldiagonal2 = costFunction2<DiagonalNormal, RegularSpace>;
constexpr auto costFunctionNormaloffdiagonal2 = costFunction2<OffDiagonalNormal, RegularSpace>;
constexpr auto costFunctionNormaldiagonal2Full = costFunction2<DiagonalNormal, FullSpace>;
constexpr auto costFunctionNormaloffdiagonal2Full = costFunction2<OffDiagonalNormal, FullSpace>;

constexpr auto costFunctionNormaldiagonal3 = costFunction3<DiagonalNormal, RegularSpace>;
constexpr auto costFunctionNormaloffdiagonal3 = costFunction3<OffDiagonalNormal, RegularSpace>;
constexpr auto costFunctionNormaldiagonal3Full = costFunction3<DiagonalNormal, FullSpace>;
constexpr auto costFunctionNormaloffdiagonal3Full = costFunction3<OffDiagonalNormal, FullSpace>;

} // namespace IMPaCT_CostFunctions

#endif // COST_FUNCTIONS_H
