#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <nlopt.hpp>
#include <iomanip>
#include <sycl/sycl.hpp>
#include <chrono>
#include "IMDP.h"
#include <glpk.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <armadillo>
#include <hdf5.h>
#include "custom.cpp"

#include "GPU_synthesis.cpp"

using namespace std;
using namespace arma;

/* IMDP Functions*/


/// Set Nonlinear Optimization Algorithm
void IMDP::setAlgorithm(nlopt::algorithm alg){
    algo = alg;
}

/* Supporter Functions for the Abstractions for Different Distributions */

/// Closed form integral for 1d normal distribution CDF
double normal1DCDF(const double& x0, const double& x1, const double& mu, const double& sigma) {
    double cdf_x0 = 0.5 * (1 + erf((x0 - mu) / (sigma * sqrt(2))));
    double cdf_x1 = 0.5 * (1 + erf((x1 - mu) / (sigma * sqrt(2))));
    return cdf_x1 - cdf_x0;
}

/// Struct to for multivariate normal noise distributions
struct multivariateNormalParams {
    vec mean;
    mat inv_cov;
    double det;
};

/// Multivariate normal noise distribution PDF
double multivariateNormalPDF(double *x, size_t dim, void *params)
{
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

/// Helper function to calculate probability product for diagonal distributions
double calculateProbabilityProduct(const vec& state_end, const vec& eta, const vec& mu, const vec& sigma) {
    double probability_product = 1.0;
    for (size_t m = 0; m < state_end.n_rows; ++m) {
        double x0 = state_end[m] - eta[m] / 2.0;
        double x1 = state_end[m] + eta[m] / 2.0;
        double probability = normal1DCDF(x0, x1, mu[m], sigma[m]);
        probability_product *= probability;
    }
    return probability_product;
}

/// Helper function to perform Monte Carlo integration for offdiagonal distributions
double performMonteCarloIntegration(const vec& mu, const mat& inv_cov, double det, const vec& state_end, const vec& eta, double dim, size_t samples) {
    multivariateNormalParams params;
    params.mean = mu;
    params.inv_cov = inv_cov;
    params.det = det;

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_monte_function F;
    F.f = &multivariateNormalPDF;
    F.dim = mu.n_rows;
    F.params = &params;

    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < state_end.n_rows; ++m) {
        lower_bounds.push_back(state_end[m] - eta[m] / 2.0);
        upper_bounds.push_back(state_end[m] + eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;

    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    gsl_monte_vegas_integrate(&F, lb, ub, dim, samples, rng, s, &result, &error);
    gsl_monte_vegas_free(s);
    gsl_rng_free(rng);

    return result;
}

/// Struct for normal distribution with 3 parameters
struct costFunctionDataNormal3 {
    double dim; //Used for offdiagonal
    vec state_end;
    vec input;
    vec disturb;
    vec eta;
    mat inv_cov; // Used for offdiagonal
    double det;  // Used for offdiagonal
    vec sigma;   // Used for diagonal
    function<vec(const vec&, const vec&, const vec&)> dynamics;
    size_t samples; // Used for offdiagonal
    bool is_diagonal; // Flag to indicate if the distribution is diagonal
};

/// Cost function for normal distribution with 3 parameters
double costFunctionNormal3(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormal3* data = static_cast<costFunctionDataNormal3*>(my_func_data);
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->input, data->disturb);

    if (data->is_diagonal) {
        return calculateProbabilityProduct(data->state_end, data->eta, mu, data->sigma);
    } else {
        return performMonteCarloIntegration(mu, data->inv_cov, data->det, data->state_end, data->eta, data->dim, data->samples);
    }
}

/// Struct for normal distribution with 2 parameters
struct costFunctionDataNormal2 {
    double dim; // Used for offdiagonal
    vec state_end;
    vec second;
    vec eta;
    mat inv_cov; // Used for offdiagonal
    double det;  // Used for offdiagonal
    vec sigma;   // Used for diagonal
    function<vec(const vec&, const vec&)> dynamics;
    size_t samples; // Used for offdiagonal
    bool is_diagonal; // Flag to indicate if the distribution is diagonal
};

/// Cost function for normal distribution with 2 parameters
double costFunctionNormal2(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormal2* data = static_cast<costFunctionDataNormal2*>(my_func_data);
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->second);

    if (data->is_diagonal) {
        return calculateProbabilityProduct(data->state_end, data->eta, mu, data->sigma);
    } else {
        return performMonteCarloIntegration(mu, data->inv_cov, data->det, data->state_end, data->eta, data->dim, data->samples);
    }
}

/// Struct for normal distribution with 1 parameter
struct costFunctionDataNormal1 {
    double dim; // Used for offdiagonal
    vec state_end;
    vec eta;
    mat inv_cov; // Used for offdiagonal
    double det;  // Used for offdiagonal
    vec sigma;   // Used for diagonal
    function<vec(const vec&)> dynamics;
    size_t samples; // Used for offdiagonal
    bool is_diagonal; // Flag to indicate if the distribution is diagonal
};

/// Cost function for normal distribution with 1 parameter
double costFunctionNormal1(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormal1* data = static_cast<costFunctionDataNormal1*>(my_func_data);
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)));

    if (data->is_diagonal) {
        return calculateProbabilityProduct(data->state_end, data->eta, mu, data->sigma);
    } else {
        return performMonteCarloIntegration(mu, data->inv_cov, data->det, data->state_end, data->eta, data->dim, data->samples);
    }
}

/* cost functions for transition to full state space */

/// normal offdiagonal cost function with 3 parameters
struct costFunctionDataNormaloffdiagonal3Full{
    double dim;
    vec state_start;
    vec lb;
    vec ub;
    vec input;
    vec disturb;
    vec eta;
    mat inv_cov;
    double det;
    function<vec(const vec&, const vec&, const vec&)> dynamics;
    size_t samples;
};

/// normal offdiagonal cost function with 3 parameters
double costFunctionNormaloffdiagonal3Full(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormaloffdiagonal3Full* data = static_cast<costFunctionDataNormaloffdiagonal3Full*>(my_func_data);
    
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->input, data->disturb);
    
    multivariateNormalParams params;
    params.mean = mu;
    params.inv_cov = data->inv_cov;
    params.det = data->det;
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    
    gsl_monte_function F;
    F.f = &multivariateNormalPDF;
    F.dim = mu.n_rows;
    F.params = &params;
    
    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        lower_bounds.push_back(data->lb[m] - data->eta[m] / 2.0);
        upper_bounds.push_back(data->ub[m] + data->eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;
    
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    
    gsl_monte_vegas_integrate(&F, lb, ub, data->dim, data->samples, rng, s, &result, &error);
    
    gsl_monte_vegas_free(s);
    
    gsl_rng_free(rng);
    
    return result;
}

/// normal diagonal cost function with 3 parameters
struct costFunctionDataNormaldiagonal3Full{
    vec state_start;
    vec ub;
    vec lb;
    vec input;
    vec disturb;
    vec eta;
    vec sigma;
    function<vec(const vec&, const vec&, const vec&)> dynamics;
};

/// normal diagonal cost function with 3 parameters
double costFunctionNormaldiagonal3Full(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormaldiagonal3Full* data = static_cast<costFunctionDataNormaldiagonal3Full*>(my_func_data);
    
    vec mu = data->dynamics(conv_to<vec>::from( vector<double>(x, x + n)), data->input, data->disturb);
    
    double probability_product = 1.0;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        double x0 = data->lb[m] - data->eta[m] / 2.0;
        double x1 = data->ub[m] + data->eta[m] / 2.0;
        
        double probability = normal1DCDF(x0, x1, mu[m], data->sigma[m]);
        probability_product *= probability;
    }
    return probability_product;
}

/// normal offdiagonal cost function with 2 parameters
struct costFunctionDataNormaloffdiagonal2Full{
    double dim;
    vec state_start;
    vec lb;
    vec ub;
    vec second;
    vec eta;
    mat inv_cov;
    double det;
    function<vec(const vec&, const vec&)> dynamics;
    size_t samples;
};

/// normal offdiagonal cost function with 2 parameters
double costFunctionNormaloffdiagonal2Full(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormaloffdiagonal2Full* data = static_cast<costFunctionDataNormaloffdiagonal2Full*>(my_func_data);
    
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->second);
    
    multivariateNormalParams params;
    params.mean = mu;
    params.inv_cov = data->inv_cov;
    params.det = data->det;
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_monte_function F;
    F.f = &multivariateNormalPDF;
    F.dim = mu.n_rows;
    F.params = &params;
    
    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        lower_bounds.push_back(data->lb[m] - data->eta[m] / 2.0);
        upper_bounds.push_back(data->ub[m] + data->eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;
    
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    gsl_monte_vegas_integrate(&F, lb, ub, data->dim, data->samples, rng, s, &result, &error);
    gsl_monte_vegas_free(s);
    gsl_rng_free(rng);
    
    return result;
}


/// normal diagonal cost function with 2 parameters
struct costFunctionDataNormaldiagonal2Full{
    vec state_start;
    vec lb;
    vec ub;
    vec second;
    vec eta;
    vec sigma;
    function<vec(const vec&, const vec&)> dynamics;
};

/// normal diagonal cost function with 2 parameters
double costFunctionNormaldiagonal2Full(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormaldiagonal2Full* data = static_cast<costFunctionDataNormaldiagonal2Full*>(my_func_data);
    vec mu = data->dynamics(conv_to<vec>::from( vector<double>(x, x + n)), data->second);
    
    double probability_product = 1.0;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        double x0 = data->lb[m] - data->eta[m] / 2.0;
        double x1 = data->ub[m] + data->eta[m] / 2.0;
        
        double probability = normal1DCDF(x0, x1, mu[m], data->sigma[m]);
        probability_product *= probability;
    }
    return probability_product;
}

/// normal offdiagonal cost function with 1 parameters
struct costFunctionDataNormaloffdiagonal1Full{
    double dim;
    vec state_start;
    vec lb;
    vec ub;
    vec eta;
    mat inv_cov;
    double det;
    function<vec(const vec&)> dynamics;
    size_t samples;
};

/// normal offdiagonal cost function with 1 parameters
double costFunctionNormaloffdiagonal1Full(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormaloffdiagonal1Full* data = static_cast<costFunctionDataNormaloffdiagonal1Full*>(my_func_data);
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)));
    
    multivariateNormalParams params;
    params.mean = mu;
    params.inv_cov = data->inv_cov;
    params.det = data->det;
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_monte_function F;
    F.f = &multivariateNormalPDF;
    F.dim = mu.n_rows;
    F.params = &params;
    
    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        lower_bounds.push_back(data->lb[m] - data->eta[m] / 2.0);
        upper_bounds.push_back(data->ub[m] + data->eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;
    
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    gsl_monte_vegas_integrate(&F, lb, ub, data->dim, data->samples, rng, s, &result, &error);
    gsl_monte_vegas_free(s);
    gsl_rng_free(rng);
    
    return result;
}

/// normal diagonal cost function with 1 parameters
struct costFunctionDataNormaldiagonal1Full{
    vec state_start;
    vec lb;
    vec ub;
    vec eta;
    vec sigma;
    function<vec(const vec&)> dynamics;
};

/// normal diagonal cost function with 1 parameters
double costFunctionNormaldiagonal1Full(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormaldiagonal1Full* data = static_cast<costFunctionDataNormaldiagonal1Full*>(my_func_data);
    
    vec mu = data->dynamics(conv_to<vec>::from( vector<double>(x, x + n)));
    
    double probability_product = 1.0;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        double x0 = data->lb[m] - data->eta[m] / 2.0;
        double x1 = data->ub[m] + data->eta[m] / 2.0;
        
        double probability = normal1DCDF(x0, x1, mu[m], data->sigma[m]);
        probability_product *= probability;
    }
    return probability_product;
}

/* CUSTOM DISTRIBUTIONS */

/// custom cost function with 1 dimension for full state space
struct costcustom1Full{
    double dim;
    vec state_start;
    vec lb;
    vec ub;
    vec eta;
    function<vec(const vec&)> dynamics;
    size_t samples;
};

/// custom cost function with 1 dimension for full state space
double custom1Full(unsigned n, const double* x, double* grad, void* my_func_data) {
    costcustom1Full* data = static_cast<costcustom1Full*>(my_func_data);
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)));
    
    customParams params;
    params.mean = mu;
    params.dynamics1 = data->dynamics;
    params.state_start = data->state_start;
    params.lb = data->lb;
    params.ub = data->ub;
    params.eta = data-> eta;
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_monte_function F;
    F.f = &customPDF;
    F.dim = mu.n_rows;
    F.params = &params;
    
    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        lower_bounds.push_back(data->lb[m] - data->eta[m] / 2.0);
        upper_bounds.push_back(data->ub[m] + data->eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;
    
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    gsl_monte_vegas_integrate(&F, lb, ub, data->dim, data->samples, rng, s, &result, &error);
    gsl_monte_vegas_free(s);
    gsl_rng_free(rng);
    
    return result;
}

/// custom cost function with 2 dimension for full state space
struct costcustom2Full{
    double dim;
    vec state_start;
    vec lb;
    vec ub;
    vec second;
    vec eta;
    function<vec(const vec&, const vec&)> dynamics;
    size_t samples;
    size_t input_space_size;
};

/// custom cost function with 2 dimension for full state space
double custom2Full(unsigned n, const double* x, double* grad, void* my_func_data) {
    costcustom2Full* data = static_cast<costcustom2Full*>(my_func_data);
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->second);
    
    customParams params;
    params.mean = mu;
    params.dynamics2 = data->dynamics;
    params.state_start = data->state_start;
    params.lb = data->lb;
    params.ub = data->ub;
    params.eta = data-> eta;
    if (data->input_space_size == 0){
        params.disturb = data->second;
    }else{
        params.input = data-> second;
    }
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_monte_function F;
    F.f = &customPDF;
    F.dim = mu.n_rows;
    F.params = &params;
    
    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        lower_bounds.push_back(data->lb[m] - data->eta[m] / 2.0);
        upper_bounds.push_back(data->ub[m] + data->eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;
    
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    gsl_monte_vegas_integrate(&F, lb, ub, data->dim, data->samples, rng, s, &result, &error);
    gsl_monte_vegas_free(s);
    gsl_rng_free(rng);
    
    return result;
}

/// custom cost function with 3 dimension for full state space
struct costcustom3Full{
    double dim;
    vec state_start;
    vec lb;
    vec ub;
    vec input;
    vec disturb;
    vec eta;
    function<vec(const vec&, const vec&, const vec&)> dynamics;
    function<double(double *x, size_t dim, void *params)> customPDF;
    size_t samples;
};

/// custom cost function with 3 dimension for full state space
double custom3Full(unsigned n, const double* x, double* grad, void* my_func_data) {
    costcustom3Full* data = static_cast<costcustom3Full*>(my_func_data);
    
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->input, data->disturb);
    
    customParams params;
    params.mean = mu;
    params.dynamics3 = data->dynamics;
    params.state_start = data->state_start;
    params.lb = data->lb;
    params.ub = data->ub;
    params.eta = data-> eta;
    params.input = data-> input;
    params.disturb = data->disturb;
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_monte_function F;
    F.f = &customPDF;
    F.dim = mu.n_rows;
    F.params = &params;
    
    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        lower_bounds.push_back(data->lb[m] - data->eta[m] / 2.0);
        upper_bounds.push_back(data->ub[m] + data->eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;
    
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    gsl_monte_vegas_integrate(&F, lb, ub, data->dim, data->samples, rng, s, &result, &error);
    gsl_monte_vegas_free(s);
    gsl_rng_free(rng);
    
    return result;
}

/// custom cost function with 1 dimension
struct costcustom1{
    double dim;
    vec state_start;
    vec state_end;
    vec lb;
    vec ub;
    vec eta;
    function<vec(const vec&)> dynamics;
    size_t samples;
};

/// custom cost function with 1 dimension
double custom1(unsigned n, const double* x, double* grad, void* my_func_data) {
    costcustom1* data = static_cast<costcustom1*>(my_func_data);
    
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)));
    
    customParams params;
    params.mean = mu;
    params.dynamics1 = data->dynamics;
    params.state_start = data->state_start;
    params.lb = data->lb;
    params.ub = data->ub;
    params.eta = data-> eta;
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_monte_function F;
    F.f = &customPDF;
    F.dim = mu.n_rows;
    F.params = &params;
    
    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        lower_bounds.push_back(data->state_end[m] - data->eta[m] / 2.0);
        upper_bounds.push_back(data->state_end[m] + data->eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;
    
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    gsl_monte_vegas_integrate(&F, lb, ub, data->dim, data->samples, rng, s, &result, &error);
    gsl_monte_vegas_free(s);
    gsl_rng_free(rng);
    
    return result;
}

/// custom cost function with 2 dimension
struct costcustom2{
    double dim;
    vec state_start;
    vec state_end;
    vec lb;
    vec ub;
    vec second;
    vec eta;
    function<vec(const vec&, const vec&)> dynamics;
    size_t samples;
    size_t input_space_size;
};

/// custom cost function with 2 dimension
double custom2(unsigned n, const double* x, double* grad, void* my_func_data) {
    costcustom2* data = static_cast<costcustom2*>(my_func_data);
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->second);
    
    customParams params;
    params.mean = mu;
    params.dynamics2 = data->dynamics;
    params.state_start = data->state_start;
    params.lb = data->lb;
    params.ub = data->ub;
    params.eta = data-> eta;
    if (data->input_space_size == 0){
        params.disturb = data->second;
    }else{
        params.input = data-> second;
    }
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    
    gsl_monte_function F;
    F.f = &customPDF;
    F.dim = mu.n_rows;
    F.params = &params;
    
    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        lower_bounds.push_back(data->state_end[m] - data->eta[m] / 2.0);
        upper_bounds.push_back(data->state_end[m] + data->eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;
    
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    
    gsl_monte_vegas_integrate(&F, lb, ub, data->dim, data->samples, rng, s, &result, &error);
    
    gsl_monte_vegas_free(s);
    
    gsl_rng_free(rng);
    
    return result;
}

/// custom cost function with 3 dimension
struct costcustom3{
    double dim;
    vec state_start;
    vec state_end;
    vec lb;
    vec ub;
    vec input;
    vec disturb;
    vec eta;
    function<vec(const vec&, const vec&, const vec&)> dynamics;
    size_t samples;
};

/// custom cost function with 3 dimension
double custom3(unsigned n, const double* x, double* grad, void* my_func_data) {
    costcustom3* data = static_cast<costcustom3*>(my_func_data);
    
    vec mu = data->dynamics(conv_to<vec>::from(vector<double>(x, x + n)), data->input, data->disturb);
    
    customParams params;
    params.mean = mu;
    params.dynamics3 = data->dynamics;
    params.state_start = data->state_start;
    params.lb = data->lb;
    params.ub = data->ub;
    params.eta = data-> eta;
    params.input = data-> input;
    params.disturb = data->disturb;
    
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_monte_function F;
    F.f = &customPDF;
    F.dim = mu.n_rows;
    F.params = &params;
    
    vector<double> lower_bounds, upper_bounds;
    for (size_t m = 0; m < data->state_start.n_rows; ++m) {
        lower_bounds.push_back(data->state_end[m] - data->eta[m] / 2.0);
        upper_bounds.push_back(data->state_end[m] + data->eta[m] / 2.0);
    }
    double* lb = lower_bounds.data();
    double* ub = upper_bounds.data();
    double result, error;
    
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    gsl_monte_vegas_integrate(&F, lb, ub, data->dim, data->samples, rng, s, &result, &error);
    gsl_monte_vegas_free(s);
    gsl_rng_free(rng);
    
    return result;
}



/* Avoid Vector Abstractions */

/// Calculate Abstraction for Minimum Avoid Transition Vector (part 1 - transitions outside state space, part 2 - sum transitions to labelled avoid states)
void IMDP::minAvoidTransitionVector(){
    auto start = chrono::steady_clock::now();
    cout << "Calculating minimal avoid transition probability Vector." << endl;
    if (disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space.n_rows;
        // transitions outside the state space
        cout << "Calculate transition to outside state space: " << total_states << " x " << 1 << endl;
        minAvoidM.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(),minAvoidM.n_rows);
                
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t i = index % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaldiagonal1Full data;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics1;
                        opt.set_max_objective(costFunctionNormaldiagonal1Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        double ans = 1.0 - minf;
                        cdfAccessor[index] = ans;
                    });
                });
            }
            queue.wait_and_throw();
            // add any states labelled avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t i = row % state_space_size;
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaldiagonal1 data;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics1;
                            opt.set_min_objective(costFunctionNormaldiagonal1, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(),minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<1> idx) {
                        size_t index = idx[0];
                        double cdf_product = 1.0;
                        size_t i = index % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaloffdiagonal1Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal1Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        double ans = 1.0 - minf;
                        cdfAccessor[index] = ans;
                    });
                });
            }
            queue.wait_and_throw();
            //sum any states labelled avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t i = row % state_space_size;
                            const vec state_start = avoid_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = target_space.row(col).t();
                            costFunctionDataNormaloffdiagonal1 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics1;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormaloffdiagonal1, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(),minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<1> idx) {
                        size_t index = idx[0];
                        double cdf_product = 1.0;
                        size_t i = index % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costcustom1Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_max_objective(custom1Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        double ans = 1.0 - minf;
                        cdfAccessor[index] = ans;
                    });
                });
            }
            queue.wait_and_throw();
            //sum any states labelled avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t i = row % state_space_size;
                            const vec state_start = avoid_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = target_space.row(col).t();
                            costcustom1 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.dynamics = dynamics1;
                            data.samples = calls;
                            opt.set_min_objective(custom1, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (disturb_space_size == 0){
        const size_t total_states = state_space_size * input_space_size;
        cout << "Avoid Vector dimensions before summation: " << total_states << " x " << 1 << endl;
        minAvoidM.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(), minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaldiagonal2Full data;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_max_objective(costFunctionNormaldiagonal2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            //sum any states labelled avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaldiagonal2 data;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics2;
                            opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(),minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaloffdiagonal2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            //sum any states labelled avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaloffdiagonal2 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if(noise == NoiseType::CUSTOM) {
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(),minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costcustom2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_max_objective(custom2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                        
                    });
                });
            }
            queue.wait_and_throw();
            //sum any states labelled avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        //sycl::range<2> local{1, state_space_size};
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costcustom2 data;
                            data.state_start = state_start;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            data.input_space_size = input_space_size;
                            opt.set_min_objective(custom2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (input_space_size == 0){
        const size_t total_states = state_space_size * disturb_space_size;
        cout << "Avoid Vector dimensions before summation: " << total_states << " x " << 1 << endl;
        minAvoidM.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(), minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % disturb_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaldiagonal2Full data;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_max_objective(costFunctionNormaldiagonal2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            //sum any states labelled avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % disturb_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaldiagonal2 data;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics2;
                            opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(),minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % disturb_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaloffdiagonal2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            //sum any states labelled avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaloffdiagonal2 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(),minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % disturb_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costcustom2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_max_objective(custom2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            // sum any states labelled avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costcustom2 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            data.input_space_size = input_space_size;
                            opt.set_min_objective(custom2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else{
        const size_t total_states = state_space_size * disturb_space_size*input_space_size;
        cout << "Avoid Vector dimensions before summation: " << total_states << " x " << 1 << endl;
        minAvoidM.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(), minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t l = index /(input_space_size*state_space_size);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaldiagonal3Full data;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.disturb = disturb;
                        data.input = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics3;
                        opt.set_max_objective(costFunctionNormaldiagonal3Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            //sum any avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaldiagonal3 data;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics3;
                            opt.set_min_objective(costFunctionNormaldiagonal3, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(),minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t l = index / (input_space_size * state_space_size);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaloffdiagonal3Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.disturb = disturb;
                        data.input = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal3Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            //sum any avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaloffdiagonal3 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics3;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormaloffdiagonal3, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minAvoidM.memptr(),minAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t l = index / (input_space_size * state_space_size);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costcustom3Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.disturb = disturb;
                        data.input = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_max_objective(custom3Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            // sum any avoid states
            if (avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costcustom3 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.dynamics = dynamics3;
                            data.samples = calls;
                            opt.set_min_objective(custom3, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                minAvoidM = minAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/// Calculate Abstraction for Maximum Avoid Transition Vector (part 1 - transitions outside state space, part 2 - sum transitions to labelled avoid states)
void IMDP::maxAvoidTransitionVector(){
    // Start timer
    auto start = chrono::steady_clock::now();
    cout << "Calculating maximal avoid transition probability Vector." << endl;
    
    if (disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space.n_rows;
        // transitions outside the state space
        cout << "Calculate transition to outside state space: " << total_states << " x " << 1 << endl;
        maxAvoidM.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(),maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t i = item.get_id(0);
                        double cdf_product = 1.0;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaldiagonal1Full data;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics1;
                        opt.set_min_objective(costFunctionNormaldiagonal1Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        double ans = 1.0 - minf;
                        cdfAccessor[i] = ans;
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0 ){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t i = row % state_space_size;
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaldiagonal1 data;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics1;
                            opt.set_max_objective(costFunctionNormaldiagonal1, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(),maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<1> idx) {
                        size_t i = idx[0];
                        double cdf_product = 1.0;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaloffdiagonal1Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal1Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        double ans = 1.0 - minf;
                        cdfAccessor[i] = ans;
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t i = row % state_space_size;
                            const vec state_start = avoid_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaloffdiagonal1 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics1;
                            data.samples = calls;
                            opt.set_max_objective(costFunctionNormaloffdiagonal1, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(),maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<1> idx) {
                        size_t i = idx[0];
                        double cdf_product = 1.0;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costcustom1Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_min_objective(custom1Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        double ans = 1.0 - minf;
                        cdfAccessor[i] = ans;
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t i = row % state_space_size;
                            const vec state_start = avoid_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costcustom1 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.dynamics = dynamics1;
                            data.samples = calls;
                            opt.set_max_objective(custom1, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (disturb_space_size == 0){
        const size_t total_states = state_space_size * input_space_size;
        cout << "Avoid Vector dimensions before summation: " << total_states << " x " << 1 << endl;
        maxAvoidM.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(), maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaldiagonal2Full data;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_min_objective(costFunctionNormaldiagonal2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaldiagonal2 data;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics2;
                            opt.set_max_objective(costFunctionNormaldiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(),maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaloffdiagonal2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaloffdiagonal2 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            opt.set_max_objective(costFunctionNormaloffdiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(),maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costcustom2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_min_objective(custom2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costcustom2 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            data.input_space_size = input_space_size;
                            opt.set_max_objective(custom2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (input_space_size == 0){
        const size_t total_states = state_space_size * disturb_space_size;
        cout << "Avoid Vector dimensions before summation: " << total_states << " x " << 1 << endl;
        maxAvoidM.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(), maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % disturb_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaldiagonal2Full data;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_min_objective(costFunctionNormaldiagonal2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % disturb_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaldiagonal2 data;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics2;
                            opt.set_max_objective(costFunctionNormaldiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(),maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % disturb_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaloffdiagonal2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows >0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaloffdiagonal2 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            opt.set_max_objective(costFunctionNormaloffdiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(),maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t k = (index / state_space_size) % disturb_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costcustom2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_min_objective(custom2Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                        
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows >0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costcustom2 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            data.input_space_size = input_space_size;
                            opt.set_max_objective(custom2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else{
        const size_t total_states = state_space_size * disturb_space_size*input_space_size;
        cout << "Avoid Vector dimensions before summation: " << total_states << " x " << 1 << endl;
        maxAvoidM.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(), maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t l = index /(input_space_size*state_space_size);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaldiagonal3Full data;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.disturb = disturb;
                        data.input = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics3;
                        opt.set_min_objective(costFunctionNormaldiagonal3Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaldiagonal3 data;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics3;
                            opt.set_max_objective(costFunctionNormaldiagonal3, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = maxAvoidM+sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(),maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t l = index / (input_space_size * state_space_size);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costFunctionDataNormaloffdiagonal3Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.disturb = disturb;
                        data.input = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal3Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                        
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormaloffdiagonal3 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics3;
                            data.samples = calls;
                            opt.set_max_objective(costFunctionNormaloffdiagonal3, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if(noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxAvoidM.memptr(),maxAvoidM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        double cdf_product = 1.0;
                        size_t l = index / (input_space_size * state_space_size);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        costcustom3Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.disturb = disturb;
                        data.input = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_min_objective(custom3Full, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = 1.0-minf;
                        
                    });
                });
            }
            queue.wait_and_throw();
            // sum other avoid states
            if(avoid_space.n_rows > 0){
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = avoid_space.row(col).t();
                            costcustom3 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.dynamics = dynamics3;
                            data.samples = calls;
                            opt.set_max_objective(custom3, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        });
                    });
                }
                queue.wait_and_throw();
                maxAvoidM = sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/// Abstraction for minimal transition matrix
void IMDP::minTransitionMatrix(){
    //Start timer
    auto start = chrono::steady_clock::now();
    cout << "Calculating minimal transition probability matrix." << endl;
    
    if (disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space_size;
        cout << "minimum transition matrix dimensions: " << total_states << " x " << state_space_size << endl;
        minTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal Minimal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaldiagonal1 data;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics1;
                        opt.set_min_objective(costFunctionNormaldiagonal1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaloffdiagonal1 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom1 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_min_objective(custom1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    
    else if (disturb_space_size == 0){
        const size_t total_states = state_space_size * input_space_size;
        cout << "minimum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        minTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaldiagonal2 data;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaloffdiagonal2 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if(noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_min_objective(custom2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                        
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (input_space_size == 0){
        const size_t total_states = state_space_size * disturb_space_size;
        cout << "minimum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        minTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % disturb_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaldiagonal2 data;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaloffdiagonal2 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_min_objective(custom2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }else{
        const size_t total_states = state_space_size * input_space_size * disturb_space_size;
        cout << "minimum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        minTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaldiagonal3 data;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics3;
                        opt.set_min_objective(costFunctionNormaldiagonal3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaloffdiagonal3 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom3 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_min_objective(custom3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/// Abstraction for maximal transition matrix
void IMDP::maxTransitionMatrix(){
    //Start timer
    auto start = chrono::steady_clock::now();
    cout << "Calculating maximal transition probability matrix." << endl;
    
    if (disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space_size;
        cout << "maximum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        maxTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaldiagonal1 data;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics1;
                        opt.set_max_objective(costFunctionNormaldiagonal1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaloffdiagonal1 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    
                    sycl::range<2> global(total_states, state_space_size);
                    
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom1 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_max_objective(custom1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    
    else if (disturb_space_size == 0){
        const size_t total_states = state_space_size * input_space_size;
        cout << "maximum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        maxTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaldiagonal2 data;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_max_objective(costFunctionNormaldiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaloffdiagonal2 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }else if (noise ==NoiseType::CUSTOM){
            cout << "Parallel run for Custom TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_max_objective(custom2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (input_space_size == 0){
        const size_t total_states = state_space_size * disturb_space_size;
        cout << "maximum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        maxTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % disturb_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaldiagonal2 data;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_max_objective(costFunctionNormaldiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaloffdiagonal2 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_max_objective(custom2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }else{
        
        const size_t total_states = state_space_size * input_space_size * disturb_space_size;
        cout << "maximum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        maxTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaldiagonal3 data;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics3;
                        opt.set_max_objective(costFunctionNormaldiagonal3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormaloffdiagonal3 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(maxTransitionM.memptr(),maxTransitionM.n_rows*maxTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom3 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_max_objective(custom3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }// Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

///Abstraction of minimal target transition vector
void IMDP::minTargetTransitionVector(){
    auto start = chrono::steady_clock::now();
    cout << "Calculating maximal transition probability Vector." << endl;
    if (disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        mat temp;
        temp.set_size(total_states, target_space.n_rows);
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb, " << total_states*target_space.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaldiagonal1 data;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics1;
                        opt.set_min_objective(costFunctionNormaldiagonal1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                        
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = target_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaloffdiagonal1 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = target_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costcustom1 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_min_objective(custom1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    
    else if (disturb_space_size == 0){
        const size_t total_states = state_space_size * input_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        mat temp;
        temp.set_size(total_states, target_space.n_rows);
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb, " << total_states*target_space.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetVector>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaldiagonal2 data;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaloffdiagonal2 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        } else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_min_objective(custom2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (input_space_size == 0){
        const size_t total_states = state_space_size * disturb_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        mat temp;
        temp.set_size(total_states, target_space.n_rows);
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb, " << total_states*target_space.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % disturb_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaldiagonal2 data;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaloffdiagonal2 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_min_objective(custom2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }else{
        const size_t total_states = state_space_size * input_space_size * disturb_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        mat temp;
        temp.set_size(total_states, target_space.n_rows);
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb, " << total_states*target_space.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaldiagonal3 data;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics3;
                        opt.set_min_objective(costFunctionNormaldiagonal3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaloffdiagonal3 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_min_objective(costFunctionNormaloffdiagonal3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        } else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costcustom3 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_min_objective(custom3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            minTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}


/// Abstraction of maximal target transition vector
void IMDP::maxTargetTransitionVector(){
    auto start = chrono::steady_clock::now();
    cout << "Calculating maximal transition probability Vector." << endl;
    if (disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        mat temp;
        temp.set_size(total_states, target_space.n_rows);
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb, " << total_states*target_space.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaldiagonal1 data;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics1;
                        opt.set_max_objective(costFunctionNormaldiagonal1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = target_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaloffdiagonal1 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        } else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t i = row % state_space_size;
                        const vec state_start = target_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costcustom1 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.samples = calls;
                        opt.set_max_objective(custom1, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    
    else if (disturb_space_size == 0){
        const size_t total_states = state_space_size * input_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        mat temp;
        temp.set_size(total_states, target_space.n_rows);
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb, " << total_states*target_space.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaldiagonal2 data;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_max_objective(costFunctionNormaldiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaloffdiagonal2 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        } else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_max_objective(custom2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (input_space_size == 0){
        const size_t total_states = state_space_size * disturb_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        mat temp;
        temp.set_size(total_states, target_space.n_rows);
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb, " << total_states*target_space.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % disturb_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaldiagonal2 data;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics2;
                        opt.set_max_objective(costFunctionNormaldiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaloffdiagonal2 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        } else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        opt.set_max_objective(custom2, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }else{
        
        const size_t total_states = state_space_size * input_space_size * disturb_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        mat temp;
        temp.set_size(total_states, target_space.n_rows);
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb, " << total_states*target_space.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaldiagonal3 data;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics = dynamics3;
                        opt.set_max_objective(costFunctionNormaldiagonal3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormaloffdiagonal3 data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_max_objective(costFunctionNormaloffdiagonal3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        } else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        double cdf_product = 1.0;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        vector<double> lb(state_start.size());
                        vector<double> ub(state_start.size());
                        for (size_t m = 0; m < state_start.size(); ++m) {
                            lb[m] = state_start[m] - ss_eta[m] / 2.0;
                            ub[m] = state_start[m] + ss_eta[m] / 2.0;
                        }
                        opt.set_lower_bounds(lb);
                        opt.set_upper_bounds(ub);
                        opt.set_xtol_rel(1e-3);
                        
                        // Prepare data for costfunction
                        const vec state_end = target_space.row(col).t();
                        costcustom3 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics3;
                        data.samples = calls;
                        opt.set_max_objective(custom3, &data);
                        vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                        double minf;
                        try {
                            nlopt::result result = opt.optimize(initial_guess, minf);
                        } catch (exception& e) {
                            cout << "nlopt failed: " << e.what() << endl;
                        }
                        cdfAccessor[index] = minf;
                    });
                });
            }
            queue.wait_and_throw();
            maxTargetM = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/* Low-cost Abstractions */

/// Low-cost abstraction of transition matrices
void IMDP::transitionMatrixBounds(){
    // Find upper bound to compare against
    maxTransitionMatrix();
    
    // Start timer for lower bound
    auto start = chrono::steady_clock::now();
    cout << "Calculating minimal transition probability matrix." << endl;
    
    if (disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space_size;
        cout << "minimum transition matrix dimensions: " << total_states << " x " << state_space_size << endl;
        minTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal Minimal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t i = row % state_space_size;
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormaldiagonal1 data;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics1;
                            opt.set_min_objective(costFunctionNormaldiagonal1, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormaloffdiagonal1 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics1;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormaloffdiagonal1, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if(noise == NoiseType::CUSTOM){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costcustom1 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.dynamics = dynamics1;
                            data.samples = calls;
                            opt.set_min_objective(custom1, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise to custom." << endl;
        }
    }
    
    else if (disturb_space_size == 0){
        const size_t total_states = state_space_size * input_space_size;
        cout << "minimum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        minTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormaldiagonal2 data;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics2;
                            opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormaloffdiagonal2 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Normal-offdiagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costcustom2 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            data.input_space_size = input_space_size;
                            opt.set_min_objective(custom2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (input_space_size == 0){
        const size_t total_states = state_space_size * disturb_space_size;
        cout << "minimum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        minTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % disturb_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormaldiagonal2 data;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics2;
                            opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormaloffdiagonal2 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Normal-offdiagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costcustom2 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.samples = calls;
                            data.input_space_size = input_space_size;
                            opt.set_min_objective(custom2, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }else{
        const size_t total_states = state_space_size * input_space_size * disturb_space_size;
        cout << "minimum transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        minTransitionM.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormaldiagonal3 data;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics = dynamics3;
                            opt.set_min_objective(costFunctionNormaldiagonal3, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormaloffdiagonal3 data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics = dynamics3;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormaloffdiagonal3, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Normal-offdiagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTransitionM.memptr(),minTransitionM.n_rows*minTransitionM.n_cols);
                
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, state_space_size);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * state_space_size + x1;
                        if(maxTransitionM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            double cdf_product = 1.0;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            vector<double> lb(state_start.size());
                            vector<double> ub(state_start.size());
                            for (size_t m = 0; m < state_start.size(); ++m) {
                                lb[m] = state_start[m] - ss_eta[m] / 2.0;
                                ub[m] = state_start[m] + ss_eta[m] / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costcustom3 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.dynamics = dynamics3;
                            data.samples = calls;
                            opt.set_min_objective(custom3, &data);
                            vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                            double minf;
                            try {
                                nlopt::result result = opt.optimize(initial_guess, minf);
                            } catch (exception& e) {
                                cout << "nlopt failed: " << e.what() << endl;
                            }
                            cdfAccessor[index] = minf;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }// Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

///Low-cost abstraction of target transition vector bounds
void IMDP::targetTransitionVectorBounds(){
    maxTargetTransitionVector();
    auto start = chrono::steady_clock::now();
    cout << "Calculating minimal target transition Vector." << endl;
    if(disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000.0 << "Kb, " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb" << endl;
        minTargetM.set_size(total_states);
        cout << "Parallel run for minimum target transition Vector... " << endl;
        
        if(noise == NoiseType::NORMAL && diagonal == true){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t i = index % state_space_size;
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormaldiagonal1 data;
                                data.state_end = state_end;
                                data.eta = ss_eta;
                                data.sigma = sigma;
                                data.dynamics = dynamics1;
                                opt.set_min_objective(costFunctionNormaldiagonal1, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }else if(noise == NoiseType::NORMAL && diagonal == false){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t i = index % state_space_size;
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormaloffdiagonal1 data;
                                data.dim = dim_x;
                                data.state_end = state_end;
                                data.eta = ss_eta;
                                data.inv_cov = inv_covariance_matrix;
                                data.det = covariance_matrix_determinant;
                                data.dynamics = dynamics1;
                                data.samples = calls;
                                opt.set_min_objective(costFunctionNormaloffdiagonal1, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }
        else if (noise == NoiseType::CUSTOM){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t i = index % state_space_size;
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costcustom1 data;
                                data.dim = dim_x;
                                data.state_start = state_start;
                                data.state_end = state_end;
                                data.eta = ss_eta;
                                data.dynamics = dynamics1;
                                data.samples = calls;
                                opt.set_min_objective(custom1, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (disturb_space_size == 0){
        const size_t total_states = state_space_size * input_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000.0 << "Kb, " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb" << endl;
        minTargetM.set_size(total_states);
        cout << "Parallel run for minimum target transition Vector... " << endl;
        if(noise == NoiseType::NORMAL && diagonal == true){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t k = (index / state_space_size) % input_space_size;
                            size_t i = index % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormaldiagonal2 data;
                                data.state_end = state_end;
                                data.second = input;
                                data.eta = ss_eta;
                                data.sigma = sigma;
                                data.dynamics = dynamics2;
                                opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }else if(noise == NoiseType::NORMAL && diagonal == false){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t k = (index / state_space_size) % input_space_size;
                            size_t i = index % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormaloffdiagonal2 data;
                                data.dim = dim_x;
                                data.state_end = state_end;
                                data.second = input;
                                data.eta = ss_eta;
                                data.inv_cov = inv_covariance_matrix;
                                data.det = covariance_matrix_determinant;
                                data.dynamics = dynamics2;
                                data.samples = calls;
                                opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }else if (noise == NoiseType::CUSTOM){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t k = (index / state_space_size) % input_space_size;
                            size_t i = index % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costcustom2 data;
                                data.dim = dim_x;
                                data.state_start = state_start;
                                data.state_end = state_end;
                                data.second = input;
                                data.eta = ss_eta;
                                data.dynamics = dynamics2;
                                data.samples = calls;
                                data.input_space_size = input_space_size;
                                opt.set_min_objective(custom2, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else if (input_space_size == 0){
        const size_t total_states = state_space_size * disturb_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000.0 << "Kb, " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb" << endl;
        minTargetM.set_size(total_states);
        cout << "Parallel run for minimum target transition Vector... " << endl;
        
        if(noise == NoiseType::NORMAL && diagonal == true){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t k = (index / state_space_size) % disturb_space_size;
                            size_t i = index % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormaldiagonal2 data;
                                data.state_end = state_end;
                                data.second = disturb;
                                data.eta = ss_eta;
                                data.sigma = sigma;
                                data.dynamics = dynamics2;
                                opt.set_min_objective(costFunctionNormaldiagonal2, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }else if(noise == NoiseType::NORMAL && diagonal == false){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t k = (index / state_space_size) % disturb_space_size;
                            size_t i = index % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormaloffdiagonal2 data;
                                data.dim = dim_x;
                                data.state_end = state_end;
                                data.second = disturb;
                                data.eta = ss_eta;
                                data.inv_cov = inv_covariance_matrix;
                                data.det = covariance_matrix_determinant;
                                data.dynamics = dynamics2;
                                data.samples = calls;
                                opt.set_min_objective(costFunctionNormaloffdiagonal2, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }
        else if (noise == NoiseType::CUSTOM){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t k = (index / state_space_size) % disturb_space_size;
                            size_t i = index % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costcustom2 data;
                                data.dim = dim_x;
                                data.state_start = state_start;
                                data.state_end = state_end;
                                data.second = disturb;
                                data.eta = ss_eta;
                                data.dynamics = dynamics2;
                                data.samples = calls;
                                data.input_space_size = input_space_size;
                                opt.set_min_objective(custom2, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }else{
        const size_t total_states = state_space_size * input_space_size * disturb_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000.0 << "Kb, " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb" << endl;
        minTargetM.set_size(total_states);
        cout << "Parallel run for minimum target transition Vector... " << endl;
        if(noise == NoiseType::NORMAL && diagonal == true){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t l = index / (input_space_size * state_space_size);
                            size_t k = (index / state_space_size) % input_space_size;
                            size_t i = index % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormaldiagonal3 data;
                                data.state_end = state_end;
                                data.input = input;
                                data.disturb = disturb;
                                data.eta = ss_eta;
                                data.sigma = sigma;
                                data.dynamics = dynamics3;
                                opt.set_min_objective(costFunctionNormaldiagonal3, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                                double minf;
                                if(minf <= 1e-28){
                                    minf = 0;
                                }
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }else if(noise == NoiseType::NORMAL && diagonal == false){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t l = index / (input_space_size * state_space_size);
                            size_t k = (index / state_space_size) % input_space_size;
                            size_t i = index % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormaloffdiagonal3 data;
                                data.dim = dim_x;
                                data.state_end = state_end;
                                data.input = input;
                                data.disturb = disturb;
                                data.eta = ss_eta;
                                data.inv_cov = inv_covariance_matrix;
                                data.det = covariance_matrix_determinant;
                                data.dynamics = dynamics3;
                                data.samples = calls;
                                opt.set_min_objective(costFunctionNormaloffdiagonal3, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }
        else if (noise==NoiseType::CUSTOM){
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(minTargetM.memptr(),minTargetM.n_rows);
                
                // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        if(maxTargetM(index) == 0){
                            cdfAccessor[index] = 0;
                        }else{
                            size_t l = index / (input_space_size * state_space_size);
                            size_t k = (index / state_space_size) % input_space_size;
                            size_t i = index % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.n_rows);
                            vector<double> lb(dim_x);
                            vector<double> ub(dim_x);
                            for (size_t m = 0; m < dim_x; ++m) {
                                lb[m] = state_start(m) - ss_eta(m) / 2.0;
                                ub[m] = state_start(m) + ss_eta(m) / 2.0;
                            }
                            opt.set_lower_bounds(lb);
                            opt.set_upper_bounds(ub);
                            opt.set_xtol_rel(1e-3);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costcustom3 data;
                                data.dim = dim_x;
                                data.state_end = state_end;
                                data.input = input;
                                data.disturb = disturb;
                                data.eta = ss_eta;
                                data.dynamics = dynamics3;
                                data.samples = calls;
                                opt.set_min_objective(custom3, &data);
                                vector<double> initial_guess = conv_to<vector<double>>::from( state_start);
                                double minf;
                                try {
                                    nlopt::result result = opt.optimize(initial_guess, minf);
                                    if(minf <= 1e-28){
                                        minf = 0;
                                    }
                                    cdf_sum += minf;
                                } catch (exception& e) {
                                    cout << "nlopt failed: " << e.what() << endl;
                                }
                            }
                            cdfAccessor[index] = cdf_sum;
                        }
                    });
                });
            }
            queue.wait_and_throw();
            cout << " Complete. ";
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/* Synthesis Functions */

/// infinite horizon reachability synthesis
void IMDP::infiniteHorizonReachController(bool IMDP_lower) {
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for infinite horizon reach controller... " << endl;
    
    if(input_space_size == 0 && disturb_space_size == 0){
        vec first0(state_space_size, 1, fill::zeros);
        vec firstnew0(state_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        vec firstnew1(state_space_size, 1, fill::zeros);
        
        double max_diff = 1.0;
        double min_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP lower bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(first1, firstnew1, "absdiff", 1e-8)) and ((approx_equal(first0, firstnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the state space, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = firstnew0;
                first1 = firstnew1;
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constrasize_t for the sum of elements in P <= 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(first1, firstnew1, "absdiff", 1e-8)) and ((approx_equal(first0, firstnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = firstnew0;
                first1 = firstnew1;
            }
            max_diff = max(abs(first1-first0));
            min_diff = min(abs(first1-first0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "verification lower bound found, finding upper bound." << endl;
        }else{
            cout << "verification upper bound found, finding lower bound." << endl;
        }
        
        vec second0(state_space_size, 1, fill::zeros);
        mat secondnew0(state_space_size, 1, fill::zeros);
        vec second1(state_space_size, 1, fill::ones);
        mat secondnew1(state_space_size, 1, fill::zeros);
        
        max_diff = 1.0;
        min_diff = 1.0;
        converge = 0;
        cout << "second loop iterations: " << endl;
        
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            //vector<double> optimal_P(n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
            }
            max_diff = max(abs(second1-second0));
            min_diff = min(abs(second1-second0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = first0;
        controller.col(dim_x + 1) = second1;
        
    }else if(input_space_size == 0){
        vec first0(state_space_size, 1, fill::zeros);
        mat firstnew0(state_space_size*disturb_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        mat firstnew1(state_space_size*disturb_space_size, 1, fill::zeros);
        //reduce matrix by choosing minimal probability from disturbances at each state
        double min_diff = 1.0;
        double max_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP lower bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size, disturb_space_size);
                firstnew1.reshape(state_space_size, disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size, disturb_space_size);
                firstnew1.reshape(state_space_size, disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
            }
            max_diff = max(abs(first1-first0));
            min_diff = min(abs(first1-first0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "verification lower bound found, finding upper bound." << endl;
        }else{
            cout << "verification upper bound found, finding lower bound." << endl;
        }
        
        vec second0(state_space_size, 1, fill::zeros);
        mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
        vec second1(state_space_size, 1, fill::ones);
        mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
        
        min_diff = 1.0;
        max_diff = 1.0;
        converge = 0;
        cout << "second loop iterations: " << endl;
        
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size, disturb_space_size);
                secondnew1.reshape(state_space_size, disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                            
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size, disturb_space_size);
                secondnew1.reshape(state_space_size, disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
            }
            max_diff = max(abs(second1-second0));
            min_diff = min(abs(second1-second0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = first0;
        controller.col(dim_x + 1) = second1;
        
    }else if (disturb_space_size == 0){
        vec first0(state_space_size, 1, fill::zeros);
        mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
        uvec U_pos(state_space_size, 1, fill::zeros);
        
        double max_diff = 1.0;
        double min_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP lower bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer0(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            firstnew0.row(index).max(cdfAccessor0[index]);
                        });
                    });
                }
                Q.wait_and_throw();
                
            }else{ // for IMDP upper bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                /*Choose input to store for controller*/
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer0(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            firstnew0.row(index).max(cdfAccessor0[index]);
                        });
                    });
                }
                Q.wait_and_throw();
                
            }
            max_diff = max(abs(first1-first0));
            min_diff = min(abs(first1-first0));
        }
        cout << endl;
        if (IMDP_lower){
            cout << "control policy for lower bound found, finding upper bound." << endl;
        }else{
            cout << "control policy for upper bound found, finding lower bound." << endl;
        }
        
        vec second0(state_space_size, 1, fill::zeros);
        mat secondnew0(state_space_size, 1, fill::zeros);
        vec second1(state_space_size, 1, fill::ones);
        mat secondnew1(state_space_size, 1, fill::zeros);
        max_diff = 1.0;
        min_diff = 1.0;
        converge = 0;
        cout << "second loop iterations: " << endl;
        mat tempTmin(state_space_size, state_space_size, fill::zeros);
        mat tempTmax(state_space_size, state_space_size, fill::zeros);
        vec tempTTmin(state_space_size, 1, fill::zeros);
        vec tempTTmax(state_space_size, 1, fill::zeros);
        vec tempATmax(state_space_size, 1, fill::zeros);
        vec tempATmin(state_space_size, 1, fill::zeros);
        
        cout << "Create reduced matrix where input is fixed." << endl;
        for (size_t i = 0; i < state_space_size; i++){
            tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
            tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
            tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
            tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
            tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
            tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
        }
        
        cout << "Matrix Fixed" << endl;
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            glp_set_col_name(lp, n+2, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
                
            }else{ // for IMDP lower bound (opposite to first)
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            glp_set_col_name(lp, n+2, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
            }
            max_diff = max(abs(second1-second0));
            min_diff = min(abs(second1-second0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + dim_u + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x+dim_u) = first0;
        controller.col(dim_x+dim_u + 1) = second1;
        for (size_t i = 0; i < state_space_size; ++i) {
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
        }
    }else{
        vec first0(state_space_size, 1, fill::zeros);
        mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
        uvec U_pos(state_space_size, 1, fill::zeros);
        mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
        mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
        
        double max_diff = 1.0;
        double min_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP lower bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+1] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer0(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            input_and_state0.row(index).max(cdfAccessor0[index]);
                        });
                    });
                }
                Q.wait_and_throw();
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                firstnew0.reshape(state_space_size*input_space_size,disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state0 = min(firstnew0,1);
                input_and_state1 = min(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(max(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(max(input_and_state1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer0(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            input_and_state0.row(index).max(cdfAccessor0[index]);
                        });
                    });
                }
                Q.wait_and_throw();
            }
            max_diff = max(abs(first1-first0));
            min_diff = min(abs(first1-first0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "control policy for lower bound found, finding upper bound." << endl;
        }else{
            cout << "control policy for upper bound found, finding lower bound." << endl;
        }
        vec second0(state_space_size, 1, fill::zeros);
        mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
        vec second1(state_space_size, 1, fill::ones);
        mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
        
        mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
        mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
        vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
        vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
        vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
        vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
        
        cout << "Create reduced matrix where input is fixed." << endl;
        for (size_t j = 0; j < disturb_space_size; j++){
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(j*state_space_size+i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
            }
        }
        
        max_diff = 1.0;
        min_diff = 1.0;
        converge = 0;
        cout << "second loop iterations: " << endl;
        
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(min(secondnew0,1));
                vec check1 = conv_to< colvec >::from(min(secondnew1,1));
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
            }
            max_diff = max(abs(second1-second0));
            min_diff = min(abs(second1-second0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + dim_u + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x+dim_u) = first0;
        controller.col(dim_x+dim_u + 1) = second1;
        for (size_t i = 0; i < state_space_size; ++i) {
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}


/// infinite horizon safeity synthesis
void IMDP::infiniteHorizonSafeController(bool IMDP_lower) {
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for infinite horizon safe controller... " << endl;
    
    // Control Synthesis
    if(input_space_size == 0 && disturb_space_size == 0){
        vec first0(state_space_size, 1, fill::zeros);
        vec firstnew0(state_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        vec firstnew1(state_space_size, 1, fill::zeros);
        vec temp0(state_space_size, 1, fill::zeros);
        vec temp1(state_space_size, 1, fill::zeros);
        
        double max_diff = 1.0;
        double min_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP lower bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as V
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients as 1.0
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            //vector<double> optimal_P(n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(firstnew1, first1, "absdiff", 1e-8)) and ((approx_equal(firstnew0, first0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = firstnew0;
                first1 = firstnew1;
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as V
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            //vector<double> optimal_P(n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(firstnew1, first1, "absdiff", 1e-8)) and ((approx_equal(firstnew0, first0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = firstnew0;
                first1 = firstnew1;
            }
            max_diff = max(abs(first1-first0));
            min_diff = min(abs(first1-first0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "verification lower bound found, finding upper bound." << endl;
        }else{
            cout << "verification upper bound found, finding lower bound." << endl;
        }
        
        vec second0(state_space_size, 1, fill::zeros);
        mat secondnew0(state_space_size, 1, fill::zeros);
        vec second1(state_space_size, 1, fill::ones);
        mat secondnew1(state_space_size, 1, fill::zeros);
        
        min_diff = 1.0;
        max_diff = 1.0;
        converge = 0;
        cout << "second loop iterations: " << endl;
        
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as V
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(secondnew1, second1, "absdiff", 1e-8)) and ((approx_equal(secondnew0, second0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); //
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(secondnew1, second1, "absdiff", 1e-8)) and ((approx_equal(secondnew0, second0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
            }
            max_diff = max(abs(second1-second0));
            min_diff = min(abs(second1-second0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = ones(state_space_size)-first0;
        controller.col(dim_x + 1) = ones(state_space_size)-second0;
        
    }else if(input_space_size == 0){
        vec first0(state_space_size, 1, fill::zeros);
        mat firstnew0(state_space_size*disturb_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        mat firstnew1(state_space_size*disturb_space_size, 1, fill::zeros);
        vec temp0(state_space_size*disturb_space_size, 1, fill::zeros);
        vec temp1(state_space_size*disturb_space_size, 1, fill::zeros);
        //reduce matrix by choosing minimal probability from disturbances at each state
        
        double min_diff = 1.0;
        double max_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;
        while (max_diff > epsilon) {
            cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP lower bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size, disturb_space_size);
                firstnew1.reshape(state_space_size, disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size, disturb_space_size);
                firstnew1.reshape(state_space_size, disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(firstnew0,1));
                vec check1 = conv_to< colvec >::from(max(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
            }
            max_diff = max(abs(first1-first0));
            min_diff = min(abs(first1-first0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "verification lower bound found, finding upper bound." << endl;
        }else{
            cout << "verification upper bound found, finding lower bound." << endl;
        }
        
        vec second0(state_space_size, 1, fill::zeros);
        mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
        vec second1(state_space_size, 1, fill::ones);
        mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
        
        min_diff = 1.0;
        max_diff = 1.0;
        converge = 0;
        cout << "second loop iterations: " << endl;
        
        while (max_diff > epsilon) {
            converge++;
            //cout << "." << flush;
            cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size, disturb_space_size);
                secondnew1.reshape(state_space_size, disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size, disturb_space_size);
                secondnew1.reshape(state_space_size, disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
            }
            max_diff = max(abs(second1-second0));
            min_diff = min(abs(second1-second0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = ones(state_space_size) - first0;
        controller.col(dim_x + 1) = ones(state_space_size) - second0;
        
    }else if (disturb_space_size == 0){
        vec first0(state_space_size, 1, fill::zeros);
        mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
        uvec U_pos(state_space_size, 1, fill::zeros);
        vec temp0(state_space_size*input_space_size, 1, fill::zeros);
        vec temp1(state_space_size*input_space_size, 1, fill::zeros);
        
        double min_diff = 1.0;
        double max_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP lower bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            //vector<double> optimal_P(n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                //Choose input to store for controller
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer0(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            firstnew0.row(index).min(cdfAccessor0[index]);
                        });
                    });
                }
                Q.wait_and_throw();
                
            }else{ // for IMDP upper bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (int i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(firstnew0,1));
                vec check1 = conv_to< colvec >::from(min(firstnew1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                
                
                /*Choose input to store for controller*/
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer0(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            firstnew0.row(index).min(cdfAccessor0[index]);
                        });
                    });
                }
                Q.wait_and_throw();
                
            }
            max_diff = max(abs(first1-first0));
            min_diff = min(abs(first1-first0));
        }
        //cout << endl;
        if (IMDP_lower){
            cout << "control policy for lower bound found, finding upper bound." << endl;
        }else{
            cout << "control policy for upper bound found, finding lower bound." << endl;
        }
        
        vec second0(state_space_size, 1, fill::zeros);
        mat secondnew0(state_space_size, 1, fill::zeros);
        vec second1(state_space_size, 1, fill::ones);
        mat secondnew1(state_space_size, 1, fill::zeros);
        min_diff = 1.0;
        max_diff = 1.0;
        converge = 0;
        cout << "second loop iterations: " << endl;
        mat tempTmin(state_space_size, state_space_size, fill::zeros);
        mat tempTmax(state_space_size, state_space_size, fill::zeros);
        vec tempTTmin(state_space_size, 1, fill::zeros);
        vec tempTTmax(state_space_size, 1, fill::zeros);
        
        cout << "Create reduced matrix where input is fixed." << endl;
        for (size_t i = 0; i < state_space_size; i++){
            tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
            tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
            tempTTmin(i)= minAvoidM(U_pos(i)*state_space_size+i);
            tempTTmax(i)= maxAvoidM(U_pos(i)*state_space_size+i);
        }
        
        cout << "Matrix Fixed" << endl;
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (int i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
                
            }else{ // for IMDP lower bound (opposite to first)
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (int i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;
            }
            max_diff = max(abs(second1-second0));
            min_diff = min(abs(second1-second0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + dim_u + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x+dim_u) = ones(state_space_size) - first1;
        controller.col(dim_x+dim_u + 1) = ones(state_space_size) - second1;
        for (size_t i = 0; i < state_space_size; ++i) {
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
        }
    }else{
        vec first0(state_space_size, 1, fill::zeros);
        mat firstnew0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        mat firstnew1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
        uvec U_pos(state_space_size, 1, fill::zeros);
        vec temp0(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
        vec temp1(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
        //reduce matrix by choosing minimal probability from disturbances at each state
        mat input_and_state0(input_space_size*state_space_size, 1, fill::zeros);
        mat input_and_state1(input_space_size*state_space_size, 1, fill::zeros);
        
        double min_diff = 1.0;
        double max_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP lower bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew0.reshape(state_space_size*input_space_size, disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size, disturb_space_size);
                input_and_state0 = max(firstnew0,1);
                input_and_state1 = max(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(min(input_and_state1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                /*Choose input to store for controller*/
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer0(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            input_and_state0.row(index).min(cdfAccessor0[index]);
                        });
                    });
                }
                Q.wait_and_throw();
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (int i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                firstnew0.reshape(state_space_size*input_space_size, disturb_space_size);
                firstnew1.reshape(state_space_size*input_space_size, disturb_space_size);
                input_and_state0 = max(firstnew0,1);
                input_and_state1 = max(firstnew1,1);
                
                /*Resize to maximise over input*/
                input_and_state0.reshape(state_space_size, input_space_size);
                input_and_state1.reshape(state_space_size, input_space_size);
                vec check0 = conv_to< colvec >::from(min(input_and_state0,1));
                vec check1 = conv_to< colvec >::from(min(input_and_state1,1));
                if((approx_equal(first1, check1, "absdiff", 1e-8)) and ((approx_equal(first0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                first0 = check0;
                first1 = check1;
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer0(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            input_and_state0.row(index).min(cdfAccessor0[index]);
                        });
                    });
                }
                Q.wait_and_throw();
            }
            max_diff = max(abs(first1-first0));
            min_diff = min(abs(first1-first0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "control policy for lower bound found, finding upper bound." << endl;
        }else{
            cout << "control policy for upper bound found, finding lower bound." << endl;
        }
        vec second0(state_space_size, 1, fill::zeros);
        mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
        vec second1(state_space_size, 1, fill::ones);
        mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);
        
        mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
        mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
        vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
        vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
        
        cout << "Create reduced matrix where input is fixed." << endl;
        for (size_t j = 0; j < disturb_space_size; j++){
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(j*state_space_size+ i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
            }
        }
        
        min_diff = 1.0;
        max_diff = 1.0;
        converge = 0;
        cout << "second loop iterations: " << endl;
        
        while (max_diff > epsilon) {
            converge++;
            cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (int i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
                
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        auto cdfAccessor1 = cdfBuffer1.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            cdfAccessor0[index] += glp_get_col_prim(lp,n+1);
                            cdfAccessor1[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew0.reshape(state_space_size,disturb_space_size);
                secondnew1.reshape(state_space_size,disturb_space_size);
                vec check0 = conv_to< colvec >::from(max(secondnew0,1));
                vec check1 = conv_to< colvec >::from(max(secondnew1,1));
                if((approx_equal(second1, check1, "absdiff", 1e-8)) and ((approx_equal(second0, check0, "absdiff", 1e-8)))){
                    cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    break;
                }
                second0 = check0;
                second1 = check1;
                
            }
            max_diff = max(abs(second1-second0));
            min_diff = min(abs(second1-second0));
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + dim_u + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x+dim_u) = ones(state_space_size) - first1;
        controller.col(dim_x+dim_u + 1) = ones(state_space_size) - second1;
        for (size_t i = 0; i < state_space_size; ++i) {
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/// finite horizon reachability synthesis
void IMDP::finiteHorizonReachController(bool IMDP_lower, size_t timeHorizon) {
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for infinite horizon reach controller... " << endl;
    
    if(input_space_size == 0 && disturb_space_size == 0){
        vec first(state_space_size, 1, fill::zeros);
        vec firstnew(state_space_size, 1, fill::zeros);
        
        double max_diff = 1.0;
        double min_diff = 1.0;
        size_t k = 0;
        cout << "first loop iterations: " << endl;
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP lower bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                first = firstnew;
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                first = firstnew;
            }
            k++;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "verification lower bound found, finding upper bound." << endl;
        }else{
            cout << "verification upper bound found, finding lower bound." << endl;
        }
        
        vec second(state_space_size, 1, fill::zeros);
        mat secondnew(state_space_size, 1, fill::zeros);
        
        max_diff = 1.0;
        min_diff = 1.0;
        k=0;
        cout << "second loop iterations: " << endl;
        
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                second = secondnew;
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                second = secondnew;
            }
            k++;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = first;
        controller.col(dim_x + 1) = second;
        
    }else if(input_space_size == 0){
        vec first(state_space_size, 1, fill::zeros);
        mat firstnew(state_space_size*disturb_space_size, 1, fill::zeros);
        //reduce matrix by choosing minimal probability from disturbances at each state
        double min_diff = 1.0;
        double max_diff = 1.0;
        size_t k = 0;
        cout << "first loop iterations: " << endl;
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP lower bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew.reshape(state_space_size, disturb_space_size);
                first = conv_to< colvec >::from(min(firstnew,1));
                
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew.reshape(state_space_size, disturb_space_size);
                first = conv_to< colvec >::from(min(firstnew,1));
            }
            k++;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "verification lower bound found, finding upper bound." << endl;
        }else{
            cout << "verification upper bound found, finding lower bound." << endl;
        }
        
        vec second(state_space_size, 1, fill::zeros);
        mat secondnew(state_space_size*disturb_space_size, 1, fill::zeros);
        
        k=0;
        cout << "second loop iterations: " << endl;
        
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew.reshape(state_space_size, disturb_space_size);
                second = conv_to< colvec >::from(min(secondnew,1));
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew.reshape(state_space_size, disturb_space_size);
                second = conv_to< colvec >::from(min(secondnew,1));
            }
            k++;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = first;
        controller.col(dim_x + 1) = second;
        
    }else if (disturb_space_size == 0){
        vec first(state_space_size, 1, fill::zeros);
        mat firstnew(state_space_size*input_space_size, 1, fill::zeros);
        uvec U_pos(state_space_size, 1, fill::zeros);
        
        size_t k = 0;
        cout << "first loop iterations: " << endl;
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP lower bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew.reshape(state_space_size, input_space_size);
                first = conv_to< colvec >::from(max(firstnew,1));
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            firstnew.row(index).max(cdfAccessor[index]);
                        });
                    });
                }
                Q.wait_and_throw();
                
            }else{ // for IMDP upper bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew.reshape(state_space_size, input_space_size);
                first = conv_to< colvec >::from(max(firstnew,1));
                
                /*Choose input to store for controller*/
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            firstnew.row(index).max(cdfAccessor[index]);
                        });
                    });
                }
                Q.wait_and_throw();
                
            }
            k++;
        }
        cout << endl;
        if (IMDP_lower){
            cout << "control policy for lower bound found, finding upper bound." << endl;
        }else{
            cout << "control policy for upper bound found, finding lower bound." << endl;
        }
        
        vec second(state_space_size, 1, fill::zeros);
        mat secondnew(state_space_size, 1, fill::zeros);
        k = 0;
        cout << "second loop iterations: " << endl;
        mat tempTmin(state_space_size, state_space_size, fill::zeros);
        mat tempTmax(state_space_size, state_space_size, fill::zeros);
        vec tempTTmin(state_space_size, 1, fill::zeros);
        vec tempTTmax(state_space_size, 1, fill::zeros);
        vec tempATmax(state_space_size, 1, fill::zeros);
        vec tempATmin(state_space_size, 1, fill::zeros);
        
        cout << "Create reduced matrix where input is fixed." << endl;
        for (size_t i = 0; i < state_space_size; i++){
            tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
            tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
            tempTTmin(i)= minTargetM(U_pos(i)*state_space_size+i);
            tempTTmax(i)= maxTargetM(U_pos(i)*state_space_size+i);
            tempATmin(i) = minAvoidM(U_pos(i)*state_space_size+i);
            tempATmax(i) = maxAvoidM(U_pos(i)*state_space_size+i);
        }
        
        cout << "Matrix Fixed" << endl;
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            glp_set_col_name(lp, n+2, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                second = secondnew;
                
            }else{ // for IMDP lower bound (opposite to first)
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0); // Set objective coefficients from V vector
                            glp_set_col_name(lp, n+2, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                second = secondnew;
            }
            k++;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + dim_u + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x+dim_u) = first;
        controller.col(dim_x+dim_u + 1) = second;
        for (size_t i = 0; i < state_space_size; ++i) {
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
        }
    }else{
        vec first(state_space_size, 1, fill::zeros);
        mat firstnew(state_space_size*input_space_size*disturb_space_size, 1, fill::zeros);
        uvec U_pos(state_space_size, 1, fill::zeros);
        mat input_and_state(input_space_size*state_space_size, 1, fill::zeros);
        
        size_t k = 0;
        cout << "first loop iterations: " << endl;
        //for (int t = 0; t < 4; t++){
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP lower bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state = min(firstnew,1);
                
                /*Resize to maximise over input*/
                input_and_state.reshape(state_space_size, input_space_size);
                first = conv_to< colvec >::from(max(input_and_state,1));
                /*Choose input to store for controller*/
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            input_and_state.row(index).max(cdfAccessor[index]);
                        });
                    });
                }
                Q.wait_and_throw();
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(minTargetM(index) == maxTargetM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minTargetM(index), maxTargetM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minTargetM(index), maxTargetM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+2, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+2, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                firstnew.reshape(state_space_size*input_space_size,disturb_space_size);
                input_and_state = min(firstnew,1);
                
                /*Resize to maximise over input*/
                input_and_state.reshape(state_space_size, input_space_size);
                first = conv_to< colvec >::from(max(input_and_state,1));
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            input_and_state.row(index).max(cdfAccessor[index]);
                        });
                    });
                }
                Q.wait_and_throw();
            }
            k++;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "control policy for lower bound found, finding upper bound." << endl;
        }else{
            cout << "control policy for upper bound found, finding lower bound." << endl;
        }
        vec second(state_space_size, 1, fill::zeros);
        mat secondnew(state_space_size*disturb_space_size, 1, fill::zeros);
        
        mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
        mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
        vec tempTTmin(state_space_size*disturb_space_size, 1, fill::zeros);
        vec tempTTmax(state_space_size*disturb_space_size, 1, fill::zeros);
        vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
        vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
        
        cout << "Create reduced matrix where input is fixed." << endl;
        for (size_t j = 0; j < disturb_space_size; j++){
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(j*state_space_size+ i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempTTmin(j*state_space_size+i)= minTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempTTmax(j*state_space_size+i)= maxTargetM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
            }
        }
        
        k=0;
        cout << "second loop iterations: " << endl;
        
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew.reshape(state_space_size,disturb_space_size);
                second = conv_to< colvec >::from(min(secondnew,1));
                
                
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+2);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "T");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 1.0);
                            
                            glp_set_col_name(lp, n+2, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+2, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 3); // Indices of the P vector elements
                            vector<double> ar(n + 3); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            ja[n+2] = n+2;
                            ar[n+2] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+2, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            cdfAccessor[index] += glp_get_col_prim(lp,n+1);
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew.reshape(state_space_size,disturb_space_size);
                second = conv_to< colvec >::from(min(secondnew,1));
                
            }
            k++;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + dim_u + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x+dim_u) = first;
        controller.col(dim_x+dim_u + 1) = second;
        for (size_t i = 0; i < state_space_size; ++i) {
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/// finite horizon safety synthesis
void IMDP::finiteHorizonSafeController(bool IMDP_lower, size_t timeHorizon) {
    auto start = chrono::steady_clock::now();
    cout << "Finding control policy for infinite horizon safe controller... " << endl;
    
    cout << "Approximate memory required if stored (each): " << minTargetM.n_rows*sizeof(double)/1000000.0 << "Mb, " << minTargetM.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
    
    if(input_space_size == 0 && disturb_space_size == 0){
        vec first(state_space_size, 1, fill::ones);
        vec firstnew(state_space_size, 1, fill::ones);
        
        size_t k = 0;
        cout << "first loop iterations: " << endl;
        //for (int t = 0; t < 4; t++){
        while (k != timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP lower bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                first = firstnew;
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                first = firstnew;
            }
            k = k+1;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "verification lower bound found, finding upper bound." << endl;
        }else{
            cout << "verification upper bound found, finding lower bound." << endl;
        }
        
        vec second(state_space_size, 1, fill::ones);
        mat secondnew(state_space_size, 1, fill::ones);
        
        k = 0;
        cout << "second loop iterations: " << endl;
        
        while (k != timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                second = secondnew;
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                second = secondnew;
            }
            k = k+1;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = first;
        controller.col(dim_x + 1) = second;
        
        //DISTURB ONLY
    }else if(input_space_size == 0){
        vec first(state_space_size, 1, fill::ones);
        mat firstnew(state_space_size*disturb_space_size, 1, fill::ones);
        //reduce matrix by choosing minimal probability from disturbances at each state
        
        size_t k = 0;
        cout << "first loop iterations: " << endl;
        while (k != timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP lower bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            //glp_set_matrix(lp, n, ia, ja, ar);
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew.reshape(state_space_size, disturb_space_size);
                first = conv_to< colvec >::from(min(firstnew,1));
                
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew.reshape(state_space_size, disturb_space_size);
                first = conv_to< colvec >::from(min(firstnew,1));
            }
            k = k+1;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "verification lower bound found, finding upper bound." << endl;
        }else{
            cout << "verification upper bound found, finding lower bound." << endl;
        }
        
        vec second(state_space_size, 1, fill::ones);
        mat secondnew(state_space_size*disturb_space_size, 1, fill::ones);
        
        k = 0;
        cout << "second loop iterations: " << endl;
        
        while (k != timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew.reshape(state_space_size, disturb_space_size);
                second = conv_to< colvec >::from(min(secondnew,1));
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew.reshape(state_space_size, disturb_space_size);
                second = conv_to< colvec >::from(min(secondnew,1));
            }
            k = k+1;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = first;
        controller.col(dim_x + 1) = second;
        
        //INPUT ONLY
    }else if (disturb_space_size == 0){
        vec first(state_space_size, 1, fill::ones);
        mat firstnew(state_space_size*input_space_size, 1, fill::ones);
        uvec U_pos(state_space_size, 1, fill::zeros);
        
        size_t k = 0;
        cout << "first loop iterations: " << endl;
        while (k != timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP lower bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew.reshape(state_space_size, input_space_size);
                first = conv_to< colvec >::from(max(firstnew,1));
                
                /*Choose input to store for controller*/
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            firstnew.row(index).max(cdfAccessor[index]);
                        });
                    });
                }
                Q.wait_and_throw();
                
            }else{ // for IMDP upper bound
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                            
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over input*/
                firstnew.reshape(state_space_size, input_space_size);
                first = conv_to< colvec >::from(max(firstnew,1));
                
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            firstnew.row(index).max(cdfAccessor[index]);
                        });
                    });
                }
                Q.wait_and_throw();
                
            }
            k = k+1;
        }
        //cout << endl;
        if (IMDP_lower){
            cout << "control policy for lower bound found, finding upper bound." << endl;
        }else{
            cout << "control policy for upper bound found, finding lower bound." << endl;
        }
        
        vec second(state_space_size, 1, fill::ones);
        mat secondnew(state_space_size, 1, fill::ones);
        k = 0;
        cout << "second loop iterations: " << endl;
        mat tempTmin(state_space_size, state_space_size, fill::zeros);
        mat tempTmax(state_space_size, state_space_size, fill::zeros);
        vec tempTTmin(state_space_size, 1, fill::zeros);
        vec tempTTmax(state_space_size, 1, fill::zeros);
        
        cout << "Create reduced matrix where input is fixed." << endl;
        for (size_t i = 0; i < state_space_size; i++){
            tempTmin.row(i) = minTransitionM.row(U_pos(i)*state_space_size+i);
            tempTmax.row(i) = maxTransitionM.row(U_pos(i)*state_space_size+i);
            tempTTmin(i)= minAvoidM(U_pos(i)*state_space_size+i);
            tempTTmax(i)= maxAvoidM(U_pos(i)*state_space_size+i);
        }
        
        cout << "Matrix Fixed" << endl;
        while (k != timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                second = secondnew;
                
            }else{ // for IMDP lower bound (opposite to first)
                
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(tempTTmin(index) == tempTTmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempTTmin(index), tempTTmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0); // Set objective coefficients from V vector
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                second = secondnew;
            }
            k = k+1;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + dim_u + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x+dim_u) = first;
        controller.col(dim_x+dim_u + 1) = second;
        for (size_t i = 0; i < state_space_size; ++i) {
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
        }
        //THE ELSE
    }else{
        vec first(state_space_size, 1, fill::ones);
        mat firstnew(state_space_size*input_space_size*disturb_space_size, 1, fill::ones);
        uvec U_pos(state_space_size, 1, fill::zeros);
        //reduce matrix by choosing minimal probability from disturbances at each state
        mat input_and_state(input_space_size*state_space_size, 1, fill::zeros);
        
        size_t k = 0;
        cout << "first loop iterations: " << endl;
        while (k != 0) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP lower bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to minimise over disturbance - worst case scenario*/
                firstnew.reshape(state_space_size*input_space_size, disturb_space_size);
                input_and_state = min(firstnew,1);
                
                /*Resize to maximise over input*/
                input_and_state.reshape(state_space_size, input_space_size);
                first = conv_to< colvec >::from(max(input_and_state,1));
                /*Choose input to store for controller*/
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer0(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor0 = cdfBuffer0.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            input_and_state.row(index).max(cdfAccessor0[index]);
                        });
                    });
                }
                Q.wait_and_throw();
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = minTransitionM.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1)); // Set objective coefficients as 1.0, multiply by V later
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(minAvoidM(index) == maxAvoidM(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0); // Set objective coefficients from V vector
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                /*Resize to maximise over disturbance - best case scenario*/
                firstnew.reshape(state_space_size*input_space_size, disturb_space_size);
                input_and_state = min(firstnew,1);
                
                /*Resize to maximise over input*/
                input_and_state.reshape(state_space_size, input_space_size);
                first = conv_to< colvec >::from(max(input_and_state,1));
                /*Choose input to store for controller*/
                sycl::queue Q;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<uword> cdfBuffer(U_pos.memptr(),U_pos.n_rows);
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    Q.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            input_and_state.row(index).max(cdfAccessor[index]);
                        });
                    });
                }
                Q.wait_and_throw();
            }
            k = k+1;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "control policy for lower bound found, finding upper bound." << endl;
        }else{
            cout << "control policy for upper bound found, finding lower bound." << endl;
        }
        vec second(state_space_size, 1, fill::ones);
        mat secondnew(state_space_size*disturb_space_size, 1, fill::ones);
        
        mat tempTmin(state_space_size*disturb_space_size, state_space_size, fill::zeros);
        mat tempTmax(state_space_size*disturb_space_size, state_space_size, fill::zeros);
        vec tempATmin(state_space_size*disturb_space_size, 1, fill::zeros);
        vec tempATmax(state_space_size*disturb_space_size, 1, fill::zeros);
        
        cout << "Create reduced matrix where input is fixed." << endl;
        for (size_t j = 0; j < disturb_space_size; j++){
            for (size_t i = 0; i < state_space_size; i++){
                tempTmin.row(j*state_space_size+ i) = minTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempTmax.row(j*state_space_size+i) = maxTransitionM.row(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempATmin(j*state_space_size+i)= minAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
                tempATmax(j*state_space_size+i)= maxAvoidM(j*input_space_size*state_space_size+U_pos(i)*state_space_size+i);
            }
        }
        
        k = 0;
        cout << "second loop iterations: " << endl;
        
        while (k != timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP upper bound (opposite to before)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to maximise over disturbance - best case scenario*/
                secondnew.reshape(state_space_size,disturb_space_size);
                second = conv_to< colvec >::from(min(secondnew,1));
                
                
            }else{ // for IMDP lower bound (opposite to first)
                sycl::queue queue;
                {
                    // Create a SYCL buffer to store the space
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);
                    
                    // Submit a SYCL kernel to calculate the coordinates and store them in the space buffer
                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        
                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);
                            // Add columns (variables) to the problem for the P vector
                            size_t n = tempTmin.row(index).n_cols; // Number of elements in P vector
                            glp_add_cols(lp, n+1);
                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1)); // Set objective coefficients from V vector
                            }
                            glp_set_col_name(lp, n+1, "A");
                            if(tempATmin(index) == tempATmax(index)){
                                glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                            }else{
                                glp_set_col_bnds(lp, n+1, GLP_DB,tempATmin(index), tempATmax(index));
                            }
                            glp_set_obj_coef(lp, n+1, 0.0);
                            
                            // Add a constraint for the sum of elements in P = 1
                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0); // Sum of P elements = 1
                            vector<int> ia = {0}; // GLPK uses 1-based indexing, so 0 is added as a placeholder
                            vector<int> ja(n + 2); // Indices of the P vector elements
                            vector<double> ar(n + 2); // Coefficients for the P vector elements in the constraint
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0; // Coefficient 1 for the corresponding P element
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            glp_set_mat_row(lp, 1, n+1, &ja[0], &ar[0]);
                            
                            // Use the simplex method to solve the LP problem
                            glp_simplex(lp, nullptr);
                            
                            // Retrieve the optimal objective value and P vector values
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            // Clean up GLPK data structures
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                
                /*Resize to minimise over disturbance - worst case scenario*/
                secondnew.reshape(state_space_size,disturb_space_size);
                second = conv_to< colvec >::from(min(secondnew,1));
                
            }
            k = k+1;
        }
        cout << endl;
        
        if (IMDP_lower){
            cout << "Upper bound found." << endl;
        }else{
            cout << "Lower bound found." << endl;
        }
        
        controller.set_size(state_space_size, dim_x + dim_u + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x+dim_u) = first;
        controller.col(dim_x+dim_u + 1) = second;
        for (size_t i = 0; i < state_space_size; ++i) {
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
        }
    }
    // Stop the timer
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}
