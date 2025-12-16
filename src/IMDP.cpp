#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <nlopt.hpp>
#include <iomanip>
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <chrono>
#include "IMDP.h"
#include <glpk.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <armadillo>
#include <hdf5/serial/hdf5.h>
#include "custom.cpp"

#include "GPU_synthesis.cpp"

using namespace std;
using namespace arma;

/* IMDP Functions*/


/// Set Nonlinear Optimization Algorithm
void IMDP::setAlgorithm(nlopt::algorithm alg){
    algo = alg;
}

/// Initialize NLopt optimizer with state space bounds (free function for SYCL kernel use)
inline void initializeOptimizer(nlopt::opt& opt, const vec& state_start, const vec& eta) {
    vector<double> lb(state_start.size());
    vector<double> ub(state_start.size());
    for (size_t m = 0; m < state_start.size(); ++m) {
        lb[m] = state_start[m] - eta[m] / 2.0;
        ub[m] = state_start[m] + eta[m] / 2.0;
    }
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_xtol_rel(1e-3);
}

/// Execute NLopt optimization with exception handling (free function for SYCL kernel use)
inline bool executeOptimization(nlopt::opt& opt, const vec& state_start, double& result) {
    vector<double> initial_guess = conv_to<vector<double>>::from(state_start);
    try {
        nlopt::result res = opt.optimize(initial_guess, result);
        return true;
    } catch (exception& e) {
        cout << "nlopt failed: " << e.what() << endl;
        return false;
    }
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

/// Struct for normal distribution with variable parameters
struct costFunctionDataNormal {
    double dim; // Used for offdiagonal
    vec state_end;
    vec input;
    vec disturb;
    vec second;
    vec eta;
    mat inv_cov; // Used for offdiagonal
    double det;  // Used for offdiagonal
    vec sigma;   // Used for diagonal
    function<vec(const vec&)> dynamics1;
    function<vec(const vec&, const vec&)> dynamics2;
    function<vec(const vec&, const vec&, const vec&)> dynamics3;
    size_t samples; // Used for offdiagonal
    bool is_diagonal; // Flag to indicate if the distribution is diagonal
};

/// Cost function for normal distribution with variable parameters
double costFunctionNormal(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormal* data = static_cast<costFunctionDataNormal*>(my_func_data);
    vec mu;

    if (data->dynamics3) {
        mu = data->dynamics3(conv_to<vec>::from(vector<double>(x, x + n)), data->input, data->disturb);
    } else if (data->dynamics2) {
        mu = data->dynamics2(conv_to<vec>::from(vector<double>(x, x + n)), data->second);
    } else {
        mu = data->dynamics1(conv_to<vec>::from(vector<double>(x, x + n)));
    }

    if (data->is_diagonal) {
        return calculateProbabilityProduct(data->state_end, data->eta, mu, data->sigma);
    } else {
        return performMonteCarloIntegration(mu, data->inv_cov, data->det, data->state_end, data->eta, data->dim, data->samples);
    }
}

/* cost functions for transition to full state space */

/// Helper function to calculate probability product for diagonal distributions
double calculateProbabilityProductFull(const vec& state_start, const vec& lb, const vec& ub, const vec& eta, const vec& mu, const vec& sigma) {
    double probability_product = 1.0;
    for (size_t m = 0; m < state_start.n_rows; ++m) {
        double x0 = lb[m] - eta[m] / 2.0;
        double x1 = ub[m] + eta[m] / 2.0;
        double probability = normal1DCDF(x0, x1, mu[m], sigma[m]);
        probability_product *= probability;
    }
    return probability_product;
}

/// Helper function to perform Monte Carlo integration for offdiagonal distributions
double performMonteCarloIntegrationFull(const vec& mu, const mat& inv_cov, double det, const vec& state_start, const vec& lb, const vec& ub, const vec& eta, double dim, size_t samples) {
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
    for (size_t m = 0; m < state_start.n_rows; ++m) {
        lower_bounds.push_back(lb[m] - eta[m] / 2.0);
        upper_bounds.push_back(ub[m] + eta[m] / 2.0);
    }
    double* lb_ptr = lower_bounds.data();
    double* ub_ptr = upper_bounds.data();
    double result, error;

    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(mu.n_rows);
    gsl_monte_vegas_integrate(&F, lb_ptr, ub_ptr, dim, samples, rng, s, &result, &error);
    gsl_monte_vegas_free(s);
    gsl_rng_free(rng);

    return result;
}

/// Struct for normal distribution with full state space
struct costFunctionDataNormalFull {
    double dim; // Used for offdiagonal
    vec state_start;
    vec lb;
    vec ub;
    vec input;
    vec disturb;
    vec second;
    vec eta;
    mat inv_cov; // Used for offdiagonal
    double det;  // Used for offdiagonal
    vec sigma;   // Used for diagonal
    function<vec(const vec&, const vec&, const vec&)> dynamics3;
    function<vec(const vec&, const vec&)> dynamics2;
    function<vec(const vec&)> dynamics1;
    size_t samples; // Used for offdiagonal
    bool is_diagonal; // Flag to indicate if the distribution is diagonal
};

/// Cost function for normal distribution with full state space
double costFunctionNormalFull(unsigned n, const double* x, double* grad, void* my_func_data) {
    costFunctionDataNormalFull* data = static_cast<costFunctionDataNormalFull*>(my_func_data);
    vec mu;

    if (data->dynamics3) {
        mu = data->dynamics3(conv_to<vec>::from(vector<double>(x, x + n)), data->input, data->disturb);
    } else if (data->dynamics2) {
        mu = data->dynamics2(conv_to<vec>::from(vector<double>(x, x + n)), data->second);
    } else {
        mu = data->dynamics1(conv_to<vec>::from(vector<double>(x, x + n)));
    }

    if (data->is_diagonal) {
        return calculateProbabilityProductFull(data->state_start, data->lb, data->ub, data->eta, mu, data->sigma);
    } else {
        return performMonteCarloIntegrationFull(mu, data->inv_cov, data->det, data->state_start, data->lb, data->ub, data->eta, data->dim, data->samples);
    }
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
    //function<double(double *x, size_t dim, void *params)> customPDF;
    //function<double(double *x, size_t dim, void *params)> customPDF;
    double (*customPDF)(double *x, size_t dim, void *params);
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
    F.f = data->customPDF;
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
    //function<double(double *x, size_t dim, void *params)> customPDF;
    double (*customPDF)(double *x, size_t dim, void *params);
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
    F.f = data->customPDF;
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
    //function<double(double *x, size_t dim, void *params)> customPDF;
    double (*customPDF)(double *x, size_t dim, void *params);
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
    F.f = data->customPDF;
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
    //function<double(double *x, size_t dim, void *params)> customPDF;
    double (*customPDF)(double *x, size_t dim, void *params);
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
    F.f = data->customPDF;
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
    //function<double(double *x, size_t dim, void *params)> customPDF;
    double (*customPDF)(double *x, size_t dim, void *params);
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
    F.f = data->customPDF;
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
    //function<double(double *x, size_t dim, void *params)> customPDF;
    double (*customPDF)(double *x, size_t dim, void *params);
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
    F.f = data->customPDF;
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

/// Internal implementation for avoid transition vector abstraction (handles both min and max)
/// The avoid vector has TWO stages:
/// 1. "Outside state space" probability using costFunctionNormalFull
///    - For is_min=true: use MAX objective (to minimize outside prob = maximize inside prob)
///    - For is_min=false: use MIN objective (to maximize outside prob = minimize inside prob)
///    - Result: ans = 1.0 - minf (invert to get outside probability)
/// 2. "Avoid states" probability using costFunctionNormal (if avoid_space exists)
///    - For is_min=true: use MIN objective
///    - For is_min=false: use MAX objective
///    - Result: output = outside_prob + sum(temp, 1)
void IMDP::avoidTransitionVectorImpl(vec& output, bool is_min){
    auto start = chrono::steady_clock::now();
    const char* bound_type = is_min ? "minimal" : "maximal";
    cout << "Calculating " << bound_type << " avoid transition probability vector." << endl;

    if (disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space.n_rows;
        cout << "Calculate transition to outside state space: " << total_states << " x " << 1 << endl;
        output.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t i = index % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costFunctionDataNormalFull data;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics1 = dynamics1;
                        data.is_diagonal = diagonal;
                        // INVERTED: min avoid -> max inside, max avoid -> min inside
                        if (is_min) {
                            opt.set_max_objective(costFunctionNormalFull, &data);
                        } else {
                            opt.set_min_objective(costFunctionNormalFull, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t i = row % state_space_size;
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormal data;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics1 = dynamics1;
                            data.is_diagonal = diagonal;
                            // NORMAL: min avoid -> min, max avoid -> max
                            if (is_min) {
                                opt.set_min_objective(costFunctionNormal, &data);
                            } else {
                                opt.set_max_objective(costFunctionNormal, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<1> idx) {
                        size_t i = idx[0];
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costFunctionDataNormalFull data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics1 = dynamics1;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_max_objective(costFunctionNormalFull, &data);
                        } else {
                            opt.set_min_objective(costFunctionNormalFull, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t i = row % state_space_size;
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormal data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics1 = dynamics1;
                            data.is_diagonal = diagonal;
                            data.samples = calls;
                            if (is_min) {
                                opt.set_min_objective(costFunctionNormal, &data);
                            } else {
                                opt.set_max_objective(costFunctionNormal, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<1> global(total_states);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<1> idx) {
                        size_t i = idx[0];
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costcustom1Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_max_objective(custom1Full, &data);
                        } else {
                            opt.set_min_objective(custom1Full, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t i = row % state_space_size;
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costcustom1 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.dynamics = dynamics1;
                            data.customPDF = customPDF;
                            data.samples = calls;
                            if (is_min) {
                                opt.set_min_objective(custom1, &data);
                            } else {
                                opt.set_max_objective(custom1, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination." << endl;
        }
    }

    else if (disturb_space_size == 0){
        const size_t total_states = state_space.n_rows * input_space_size;
        cout << "Calculate transition to outside state space: " << total_states << " x " << 1 << endl;
        output.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costFunctionDataNormalFull data;
                        data.state_start = state_start;
                        data.second = input;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_max_objective(costFunctionNormalFull, &data);
                        } else {
                            opt.set_min_objective(costFunctionNormalFull, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormal data;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics2 = dynamics2;
                            data.is_diagonal = diagonal;
                            if (is_min) {
                                opt.set_min_objective(costFunctionNormal, &data);
                            } else {
                                opt.set_max_objective(costFunctionNormal, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costFunctionDataNormalFull data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.second = input;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_max_objective(costFunctionNormalFull, &data);
                        } else {
                            opt.set_min_objective(costFunctionNormalFull, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormal data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics2 = dynamics2;
                            data.is_diagonal = diagonal;
                            data.samples = calls;
                            if (is_min) {
                                opt.set_min_objective(costFunctionNormal, &data);
                            } else {
                                opt.set_max_objective(costFunctionNormal, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costcustom2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.second = input;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_max_objective(custom2Full, &data);
                        } else {
                            opt.set_min_objective(custom2Full, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costcustom2 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.customPDF = customPDF;
                            data.samples = calls;
                            data.input_space_size = input_space_size;
                            if (is_min) {
                                opt.set_min_objective(custom2, &data);
                            } else {
                                opt.set_max_objective(custom2, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination." << endl;
        }
    }
    else if (input_space_size == 0){
        // Case 3: Verification with disturbance (no inputs)
        const size_t total_states = state_space_size * disturb_space_size;
        cout << "Calculate transition to outside state space: " << total_states << " x " << 1 << endl;
        output.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t k = (index / state_space_size) % disturb_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costFunctionDataNormalFull data;
                        data.state_start = state_start;
                        data.second = disturb;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_max_objective(costFunctionNormalFull, &data);
                        } else {
                            opt.set_min_objective(costFunctionNormalFull, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t k = (row / state_space_size) % disturb_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormal data;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics2 = dynamics2;
                            data.is_diagonal = diagonal;
                            if (is_min) {
                                opt.set_min_objective(costFunctionNormal, &data);
                            } else {
                                opt.set_max_objective(costFunctionNormal, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t k = (index / state_space_size) % disturb_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costFunctionDataNormalFull data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.second = disturb;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_max_objective(costFunctionNormalFull, &data);
                        } else {
                            opt.set_min_objective(costFunctionNormalFull, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t k = (row / state_space_size) % disturb_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormal data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics2 = dynamics2;
                            data.is_diagonal = diagonal;
                            data.samples = calls;
                            if (is_min) {
                                opt.set_min_objective(costFunctionNormal, &data);
                            } else {
                                opt.set_max_objective(costFunctionNormal, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t k = (index / state_space_size) % disturb_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costcustom2Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.second = disturb;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_max_objective(custom2Full, &data);
                        } else {
                            opt.set_min_objective(custom2Full, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t k = (row / state_space_size) % disturb_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costcustom2 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.customPDF = customPDF;
                            data.samples = calls;
                            data.input_space_size = input_space_size;
                            if (is_min) {
                                opt.set_min_objective(custom2, &data);
                            } else {
                                opt.set_max_objective(custom2, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination." << endl;
        }
    }
    else{
        // Case 4: Full IMDP with both inputs and disturbances
        const size_t total_states = state_space_size * input_space_size * disturb_space_size;
        cout << "Calculate transition to outside state space: " << total_states << " x " << 1 << endl;
        output.set_size(total_states);
        mat temp;
        if (avoid_space.n_rows > 0){
            temp.set_size(total_states, avoid_space.n_rows);
        }
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t l = index / (input_space_size * state_space_size);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costFunctionDataNormalFull data;
                        data.state_start = state_start;
                        data.input = input;
                        data.disturb = disturb;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics3 = dynamics3;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_max_objective(costFunctionNormalFull, &data);
                        } else {
                            opt.set_min_objective(costFunctionNormalFull, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormal data;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics3 = dynamics3;
                            data.is_diagonal = diagonal;
                            if (is_min) {
                                opt.set_min_objective(costFunctionNormal, &data);
                            } else {
                                opt.set_max_objective(costFunctionNormal, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t l = index / (input_space_size * state_space_size);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costFunctionDataNormalFull data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.input = input;
                        data.disturb = disturb;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics3 = dynamics3;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_max_objective(costFunctionNormalFull, &data);
                        } else {
                            opt.set_min_objective(costFunctionNormalFull, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costFunctionDataNormal data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics3 = dynamics3;
                            data.is_diagonal = diagonal;
                            data.samples = calls;
                            if (is_min) {
                                opt.set_min_objective(costFunctionNormal, &data);
                            } else {
                                opt.set_max_objective(costFunctionNormal, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom AvoidTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    cgh.parallel_for<class SetMatrix>(sycl::range<1>(total_states), [=](sycl::item<1> item) {
                        size_t index = item.get_id(0);
                        size_t l = index / (input_space_size * state_space_size);
                        size_t k = (index / state_space_size) % input_space_size;
                        size_t i = index % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        costcustom3Full data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.input = input;
                        data.disturb = disturb;
                        data.lb = ss_lb;
                        data.ub = ss_ub;
                        data.eta = ss_eta;
                        data.dynamics = dynamics3;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_max_objective(custom3Full, &data);
                        } else {
                            opt.set_min_objective(custom3Full, &data);
                        }
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
            if(avoid_space.n_rows > 0){
                sycl::queue queue2;
                {
                    sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                    queue2.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                        sycl::range<2> global(total_states, avoid_space.n_rows);
                        cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                            const size_t x0 = idx[0];
                            const size_t x1 = idx[1];
                            size_t index = x0 * avoid_space.n_rows + x1;
                            size_t row = index%total_states;
                            size_t col = index/total_states;
                            size_t l = row / (input_space_size * state_space_size);
                            size_t k = (row / state_space_size) % input_space_size;
                            size_t i = row % state_space_size;
                            const vec disturb = disturb_space.row(l).t();
                            const vec input = input_space.row(k).t();
                            const vec state_start = state_space.row(i).t();
                            nlopt::opt opt(algo, state_start.size());
                            initializeOptimizer(opt, state_start, ss_eta);

                            const vec state_end = avoid_space.row(col).t();
                            costcustom3 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.dynamics = dynamics3;
                            data.customPDF = customPDF;
                            data.samples = calls;
                            if (is_min) {
                                opt.set_min_objective(custom3, &data);
                            } else {
                                opt.set_max_objective(custom3, &data);
                            }
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
                queue2.wait_and_throw();
                output = output + sum(temp,1);
            }
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination." << endl;
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/// Calculate Abstraction for Minimum Avoid Transition Vector (part 1 - transitions outside state space, part 2 - sum transitions to labelled avoid states)
void IMDP::minAvoidTransitionVector(){
    avoidTransitionVectorImpl(minAvoidM, true);
}


/// Calculate Abstraction for Maximum Avoid Transition Vector (part 1 - transitions outside state space, part 2 - sum transitions to labelled avoid states)
void IMDP::maxAvoidTransitionVector(){
    avoidTransitionVectorImpl(maxAvoidM, false);
}


/// Internal implementation for transition matrix abstraction (handles both min and max)
void IMDP::transitionMatrixImpl(mat& output, bool is_min){
    //Start timer
    auto start = chrono::steady_clock::now();
    const char* bound_type = is_min ? "minimal" : "maximal";
    const char* bound_type_short = is_min ? "minimum" : "maximum";
    cout << "Calculating " << bound_type << " transition probability matrix." << endl;

    if (disturb_space_size == 0 && input_space_size == 0){
        const size_t total_states = state_space_size;
        cout << bound_type_short << " transition matrix dimensions: " << total_states << " x " << state_space_size << endl;
        output.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormal data;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics1 = dynamics1;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormal data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics1 = dynamics1;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom1 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(custom1, &data);
                        } else {
                            opt.set_max_objective(custom1, &data);
                        }
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
        cout << bound_type_short << " transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        output.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormal data;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormal data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        if (is_min) {
                            opt.set_min_objective(custom2, &data);
                        } else {
                            opt.set_max_objective(custom2, &data);
                        }
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
        cout << bound_type_short << " transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        output.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal Transition Matrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormal data;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormal data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        if (is_min) {
                            opt.set_min_objective(custom2, &data);
                        } else {
                            opt.set_max_objective(custom2, &data);
                        }
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
        cout << bound_type_short << " transition Matrix dimensions: " << total_states << " x " << state_space_size << endl;
        output.set_size(total_states, state_space_size);
        cout << "Approximate memory required if stored: " << total_states*state_space_size*sizeof(double)/1000000.0 << "Mb, " << total_states*state_space_size*sizeof(double)/1000000000.0 << "Gb" << endl;
        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TransitionMatrix... " << endl;
            sycl::queue queue;
            {
                // Create a SYCL buffer to store the space
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormal data;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics3 = dynamics3;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

                        // Prepare data for costfunction
                        const vec state_end = state_space.row(col).t();
                        costFunctionDataNormal data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics3 = dynamics3;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
                sycl::buffer<double> cdfBuffer(output.memptr(),output.n_rows*output.n_cols);
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
                        initializeOptimizer(opt, state_start, ss_eta);

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
                        data.customPDF = customPDF;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(custom3, &data);
                        } else {
                            opt.set_max_objective(custom3, &data);
                        }
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

/// Abstraction for minimal transition matrix
void IMDP::minTransitionMatrix(){
    transitionMatrixImpl(minTransitionM, true);
}

/// Abstraction for maximal transition matrix
void IMDP::maxTransitionMatrix(){
    transitionMatrixImpl(maxTransitionM, false);
}

/// Internal implementation for target transition vector abstraction (handles both min and max)
void IMDP::targetTransitionVectorImpl(vec& output, bool is_min){
    auto start = chrono::steady_clock::now();
    const char* bound_type = is_min ? "minimal" : "maximal";
    cout << "Calculating " << bound_type << " target transition probability vector." << endl;

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
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t i = row % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormal data;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics1 = dynamics1;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t i = row % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormal data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics1 = dynamics1;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t i = row % state_space_size;
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costcustom1 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.eta = ss_eta;
                        data.dynamics = dynamics1;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(custom1, &data);
                        } else {
                            opt.set_max_objective(custom1, &data);
                        }
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
            output = sum(temp,1);
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
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormal data;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormal data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if(noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = input;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        if (is_min) {
                            opt.set_min_objective(custom2, &data);
                        } else {
                            opt.set_max_objective(custom2, &data);
                        }
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
            output = sum(temp,1);
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
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t k = (row / state_space_size) % disturb_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormal data;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t k = (row / state_space_size) % disturb_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormal data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics2 = dynamics2;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t k = (row / state_space_size) % disturb_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costcustom2 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.second = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics2;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        data.input_space_size = input_space_size;
                        if (is_min) {
                            opt.set_min_objective(custom2, &data);
                        } else {
                            opt.set_max_objective(custom2, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    else{
        const size_t total_states = state_space_size * input_space_size * disturb_space_size;
        cout << "Target Vector dimensions before summation: " << total_states << " x " << target_space.n_rows << endl;
        mat temp;
        temp.set_size(total_states, target_space.n_rows);
        cout << "Approximate memory required if stored: " << total_states*target_space.n_rows*sizeof(double)/1000000.0 << "Mb, " << total_states*target_space.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;

        if (noise == NoiseType::NORMAL && diagonal == true){
            cout << "Parallel run for Normal-diagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormal data;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.sigma = sigma;
                        data.dynamics3 = dynamics3;
                        data.is_diagonal = diagonal;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::NORMAL && diagonal == false){
            cout << "Parallel run for Normal-offdiagonal TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costFunctionDataNormal data;
                        data.dim = dim_x;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.inv_cov = inv_covariance_matrix;
                        data.det = covariance_matrix_determinant;
                        data.dynamics3 = dynamics3;
                        data.is_diagonal = diagonal;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(costFunctionNormal, &data);
                        } else {
                            opt.set_max_objective(costFunctionNormal, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else if (noise == NoiseType::CUSTOM){
            cout << "Parallel run for Custom TargetTransitionVector... " << endl;
            sycl::queue queue;
            {
                sycl::buffer<double> cdfBuffer(temp.memptr(),temp.n_rows*temp.n_cols);
                queue.submit([&](sycl::handler& cgh) {
                    auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);
                    sycl::range<2> global(total_states, target_space.n_rows);
                    cgh.parallel_for<class SetMatrix>(global, [=](sycl::id<2> idx) {
                        const size_t x0 = idx[0];
                        const size_t x1 = idx[1];
                        size_t index = x0 * target_space.n_rows + x1;
                        size_t row = index%total_states;
                        size_t col = index/total_states;
                        size_t l = row / (input_space_size * state_space_size);
                        size_t k = (row / state_space_size) % input_space_size;
                        size_t i = row % state_space_size;
                        const vec disturb = disturb_space.row(l).t();
                        const vec input = input_space.row(k).t();
                        const vec state_start = state_space.row(i).t();
                        nlopt::opt opt(algo, state_start.size());
                        initializeOptimizer(opt, state_start, ss_eta);

                        const vec state_end = target_space.row(col).t();
                        costcustom3 data;
                        data.dim = dim_x;
                        data.state_start = state_start;
                        data.state_end = state_end;
                        data.input = input;
                        data.disturb = disturb;
                        data.eta = ss_eta;
                        data.dynamics = dynamics3;
                        data.customPDF = customPDF;
                        data.samples = calls;
                        if (is_min) {
                            opt.set_min_objective(custom3, &data);
                        } else {
                            opt.set_max_objective(custom3, &data);
                        }
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
            output = sum(temp,1);
            cout << " Complete." << endl;
        }
        else{
            cout << "Unsupported noise combination, either swap offdiagonal/diagonal or change type of noise." << endl;
        }
    }
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

///Abstraction of minimal target transition vector
void IMDP::minTargetTransitionVector(){
    targetTransitionVectorImpl(minTargetM, true);
}

/// Abstraction of maximal target transition vector
void IMDP::maxTargetTransitionVector(){
    targetTransitionVectorImpl(maxTargetM, false);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormal data;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics1 = dynamics1;
                            data.is_diagonal = diagonal;
                            opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormal data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics1 = dynamics1;
                            data.is_diagonal = diagonal;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costcustom1 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.eta = ss_eta;
                            data.dynamics = dynamics1;
                            data.customPDF = customPDF;
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormal data;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics2 = dynamics2;
                            data.is_diagonal = diagonal;
                            opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormal data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics2 = dynamics2;
                            data.is_diagonal = diagonal;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costcustom2 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.second = input;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.customPDF = customPDF;
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormal data;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics2 = dynamics2;
                            data.is_diagonal = diagonal;
                            opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormal data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics2 = dynamics2;
                            data.is_diagonal = diagonal;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costcustom2 data;
                            data.dim = dim_x;
                            data.state_start = state_start;
                            data.state_end = state_end;
                            data.second = disturb;
                            data.eta = ss_eta;
                            data.dynamics = dynamics2;
                            data.customPDF = customPDF;
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormal data;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.sigma = sigma;
                            data.dynamics3 = dynamics3;
                            data.is_diagonal = diagonal;
                            opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            // Prepare data for costfunction
                            const vec state_end = state_space.row(col).t();
                            costFunctionDataNormal data;
                            data.dim = dim_x;
                            data.state_end = state_end;
                            data.input = input;
                            data.disturb = disturb;
                            data.eta = ss_eta;
                            data.inv_cov = inv_covariance_matrix;
                            data.det = covariance_matrix_determinant;
                            data.dynamics3 = dynamics3;
                            data.is_diagonal = diagonal;
                            data.samples = calls;
                            opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
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
                            data.customPDF = customPDF;
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormal data;
                                data.state_end = state_end;
                                data.eta = ss_eta;
                                data.sigma = sigma;
                                data.dynamics1 = dynamics1;
                                data.is_diagonal = diagonal;
                                opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormal data;
                                data.dim = dim_x;
                                data.state_end = state_end;
                                data.eta = ss_eta;
                                data.inv_cov = inv_covariance_matrix;
                                data.det = covariance_matrix_determinant;
                                data.dynamics1 = dynamics1;
                                data.is_diagonal = diagonal;
                                data.samples = calls;
                                opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
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
                                data.customPDF = customPDF;
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormal data;
                                data.state_end = state_end;
                                data.second = input;
                                data.eta = ss_eta;
                                data.sigma = sigma;
                                data.dynamics2 = dynamics2;
                                data.is_diagonal = diagonal;
                                opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormal data;
                                data.dim = dim_x;
                                data.state_end = state_end;
                                data.second = input;
                                data.eta = ss_eta;
                                data.inv_cov = inv_covariance_matrix;
                                data.det = covariance_matrix_determinant;
                                data.dynamics2 = dynamics2;
                                data.is_diagonal = diagonal;
                                data.samples = calls;
                                opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
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
                                data.customPDF = customPDF;
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormal data;
                                data.state_end = state_end;
                                data.second = disturb;
                                data.eta = ss_eta;
                                data.sigma = sigma;
                                data.dynamics2 = dynamics2;
                                data.is_diagonal = diagonal;
                                opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormal data;
                                data.dim = dim_x;
                                data.state_end = state_end;
                                data.second = disturb;
                                data.eta = ss_eta;
                                data.inv_cov = inv_covariance_matrix;
                                data.det = covariance_matrix_determinant;
                                data.dynamics2 = dynamics2;
                                data.is_diagonal = diagonal;
                                data.samples = calls;
                                opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            
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
                                data.customPDF = customPDF;
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormal data;
                                data.state_end = state_end;
                                data.input = input;
                                data.disturb = disturb;
                                data.eta = ss_eta;
                                data.sigma = sigma;
                                data.dynamics3 = dynamics3;
                                data.is_diagonal = diagonal;
                                opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
                            double cdf_sum = 0.0;
                            for (size_t j = 0; j < target_space.n_rows; ++j) {
                                // Prepare data for costfunction
                                const vec state_end = target_space.row(j).t();
                                costFunctionDataNormal data;
                                data.dim = dim_x;
                                data.state_end = state_end;
                                data.input = input;
                                data.disturb = disturb;
                                data.eta = ss_eta;
                                data.inv_cov = inv_covariance_matrix;
                                data.det = covariance_matrix_determinant;
                                data.dynamics3 = dynamics3;
                                data.is_diagonal = diagonal;
                                data.samples = calls;
                                opt.set_min_objective(costFunctionNormal, &data);
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
                            initializeOptimizer(opt, state_start, ss_eta);
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
                                data.customPDF = customPDF;
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

/// Internal implementation for infinite horizon controller synthesis (reach and safe)
void IMDP::infiniteHorizonControllerImpl(bool IMDP_lower, bool is_reach) {
    auto start = chrono::steady_clock::now();
    if (is_reach) {
        cout << "Finding control policy for infinite horizon reach controller... " << endl;
    } else {
        cout << "Finding control policy for infinite horizon safe controller... " << endl;
    }

    // LP configuration: Reach uses n+2 columns (P + Target + Avoid), Safe uses n+1 (P + Avoid)
    // Direction is inverted between reach and safe for the same IMDP_lower setting

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
            if (is_reach) {
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            } else {
                cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            }

            // Determine LP direction based on is_reach and IMDP_lower
            // Reach: IMDP_lower=true -> GLP_MIN, IMDP_lower=false -> GLP_MAX
            // Safe:  IMDP_lower=true -> GLP_MAX, IMDP_lower=false -> GLP_MIN (inverted)
            bool use_min_direction = is_reach ? IMDP_lower : !IMDP_lower;

            if (use_min_direction){ // GLP_MIN case
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1));
                            }

                            if (is_reach) {
                                // Reach: add Target column (n+1) then Avoid column (n+2)
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
                            } else {
                                // Safe: only Avoid column (n+1)
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(first1, firstnew1, "absdiff", 1e-8)) and ((approx_equal(first0, firstnew0, "absdiff", 1e-8)))){
                    if (is_reach) {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the state space, try running the finite Horizon solution using this number of steps." << endl;
                    } else {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    }
                    break;
                }
                first0 = firstnew0;
                first1 = firstnew1;
            }else{ // GLP_MAX case
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if (is_reach) {
                    if((approx_equal(first1, firstnew1, "absdiff", 1e-8)) and ((approx_equal(first0, firstnew0, "absdiff", 1e-8)))){
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                        break;
                    }
                } else {
                    if((approx_equal(firstnew1, first1, "absdiff", 1e-8)) and ((approx_equal(firstnew0, first0, "absdiff", 1e-8)))){
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                        break;
                    }
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

        // Second phase: direction is opposite to first phase
        bool use_min_direction_second = is_reach ? !IMDP_lower : IMDP_lower;

        while (max_diff > epsilon) {
            converge++;
            if (is_reach) {
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            } else {
                cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            }

            if (!use_min_direction_second){ // GLP_MAX case (opposite for second phase)
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if (is_reach) {
                    if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                        break;
                    }
                } else {
                    if((approx_equal(secondnew1, second1, "absdiff", 1e-8)) and ((approx_equal(secondnew0, second0, "absdiff", 1e-8)))){
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                        break;
                    }
                }
                second0 = secondnew0;
                second1 = secondnew1;
            }else{ // GLP_MIN case
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if (is_reach) {
                    if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                        break;
                    }
                } else {
                    if((approx_equal(secondnew1, second1, "absdiff", 1e-8)) and ((approx_equal(secondnew0, second0, "absdiff", 1e-8)))){
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                        break;
                    }
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
        if (is_reach) {
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second1;
        } else {
            controller.col(dim_x) = ones(state_space_size)-first0;
            controller.col(dim_x + 1) = ones(state_space_size)-second0;
        }

    }else if(input_space_size == 0){
        // Disturbance only case - verification with disturbance
        vec first0(state_space_size, 1, fill::zeros);
        mat firstnew0(state_space_size*disturb_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        mat firstnew1(state_space_size*disturb_space_size, 1, fill::zeros);

        double min_diff = 1.0;
        double max_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;
        while (max_diff > epsilon) {
            converge++;
            if (is_reach) {
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            } else {
                cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            }

            bool use_min_direction = is_reach ? IMDP_lower : !IMDP_lower;

            if (use_min_direction){
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                // Resize and reduce over disturbances
                firstnew0.reshape(state_space_size, disturb_space_size);
                firstnew1.reshape(state_space_size, disturb_space_size);
                first0 = conv_to<colvec>::from(min(firstnew0, 1));
                first1 = conv_to<colvec>::from(min(firstnew1, 1));
                firstnew0.reshape(state_space_size*disturb_space_size, 1);
                firstnew1.reshape(state_space_size*disturb_space_size, 1);
            }else{
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                // Resize and reduce over disturbances
                firstnew0.reshape(state_space_size, disturb_space_size);
                firstnew1.reshape(state_space_size, disturb_space_size);
                first0 = conv_to<colvec>::from(min(firstnew0, 1));
                first1 = conv_to<colvec>::from(min(firstnew1, 1));
                firstnew0.reshape(state_space_size*disturb_space_size, 1);
                firstnew1.reshape(state_space_size*disturb_space_size, 1);
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

        // Second phase
        vec second0(state_space_size, 1, fill::zeros);
        mat secondnew0(state_space_size*disturb_space_size, 1, fill::zeros);
        vec second1(state_space_size, 1, fill::ones);
        mat secondnew1(state_space_size*disturb_space_size, 1, fill::zeros);

        max_diff = 1.0;
        min_diff = 1.0;
        converge = 0;
        cout << "second loop iterations: " << endl;

        while (max_diff > epsilon) {
            converge++;
            if (is_reach) {
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            } else {
                cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            }

            bool use_min_direction_second = is_reach ? !IMDP_lower : IMDP_lower;

            if (!use_min_direction_second){
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                secondnew0.reshape(state_space_size, disturb_space_size);
                secondnew1.reshape(state_space_size, disturb_space_size);
                second0 = conv_to<colvec>::from(min(secondnew0, 1));
                second1 = conv_to<colvec>::from(min(secondnew1, 1));
                secondnew0.reshape(state_space_size*disturb_space_size, 1);
                secondnew1.reshape(state_space_size*disturb_space_size, 1);
            }else{
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                secondnew0.reshape(state_space_size, disturb_space_size);
                secondnew1.reshape(state_space_size, disturb_space_size);
                second0 = conv_to<colvec>::from(min(secondnew0, 1));
                second1 = conv_to<colvec>::from(min(secondnew1, 1));
                secondnew0.reshape(state_space_size*disturb_space_size, 1);
                secondnew1.reshape(state_space_size*disturb_space_size, 1);
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
        controller.cols(0, dim_x - 1) = state_space;
        if (is_reach) {
            controller.col(dim_x) = first0;
            controller.col(dim_x + 1) = second1;
        } else {
            controller.col(dim_x) = ones(state_space_size) - first0;
            controller.col(dim_x + 1) = ones(state_space_size) - second1;
        }

    }else{
        // Input-based synthesis (controller synthesis with inputs)
        // This is the main controller synthesis case

        vec first0(state_space_size, 1, fill::zeros);
        mat firstnew0(state_space_size*input_space_size, 1, fill::zeros);
        vec first1(state_space_size, 1, fill::ones);
        mat firstnew1(state_space_size*input_space_size, 1, fill::zeros);
        uvec U_pos(state_space_size);

        double min_diff = 1.0;
        double max_diff = 1.0;
        size_t converge = 0;
        cout << "first loop iterations: " << endl;

        while (max_diff > epsilon) {
            converge++;
            if (is_reach) {
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            } else {
                cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            }

            bool use_min_direction = is_reach ? IMDP_lower : !IMDP_lower;

            if (use_min_direction){
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                // Resize and reduce over inputs - maximize probability for reach, minimize for safe
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                if (is_reach) {
                    // Reach: maximize
                    first0 = conv_to<colvec>::from(max(firstnew0, 1));
                    first1 = conv_to<colvec>::from(max(firstnew1, 1));
                    U_pos = index_max(firstnew1, 1);
                } else {
                    // Safe: minimize (worst case)
                    first0 = conv_to<colvec>::from(min(firstnew0, 1));
                    first1 = conv_to<colvec>::from(min(firstnew1, 1));
                    U_pos = index_min(firstnew1, 1);
                }
                firstnew0.reshape(state_space_size*input_space_size, 1);
                firstnew1.reshape(state_space_size*input_space_size, 1);

                if((approx_equal(first1, firstnew1.col(0), "absdiff", 1e-8)) and ((approx_equal(first0, firstnew0.col(0), "absdiff", 1e-8)))){
                    if (is_reach) {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    } else {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    }
                    break;
                }
            }else{
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(firstnew0.memptr(),firstnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(firstnew1.memptr(),firstnew1.n_rows);

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

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first1(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*first0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*first1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                // Resize and reduce over inputs
                firstnew0.reshape(state_space_size, input_space_size);
                firstnew1.reshape(state_space_size, input_space_size);
                if (is_reach) {
                    first0 = conv_to<colvec>::from(max(firstnew0, 1));
                    first1 = conv_to<colvec>::from(max(firstnew1, 1));
                    U_pos = index_max(firstnew1, 1);
                } else {
                    first0 = conv_to<colvec>::from(min(firstnew0, 1));
                    first1 = conv_to<colvec>::from(min(firstnew1, 1));
                    U_pos = index_min(firstnew1, 1);
                }
                firstnew0.reshape(state_space_size*input_space_size, 1);
                firstnew1.reshape(state_space_size*input_space_size, 1);

                if((approx_equal(first1, firstnew1.col(0), "absdiff", 1e-8)) and ((approx_equal(first0, firstnew0.col(0), "absdiff", 1e-8)))){
                    if (is_reach) {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    } else {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    }
                    break;
                }
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

        // Second phase with fixed controller from U_pos
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
            if (is_reach) {
                cout << "Max: " << max_diff << ", Min: " << min_diff << endl;
            } else {
                cout << "Max: " << max_diff << " Min: " << min_diff << endl;
            }

            bool use_min_direction_second = is_reach ? !IMDP_lower : IMDP_lower;

            if (!use_min_direction_second){
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);

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

                            size_t n = tempTmin.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1));
                            }

                            if (is_reach) {
                                glp_set_col_name(lp, n+1, "T");
                                if(tempTTmin(index) == tempTTmax(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, tempTTmin(index), tempTTmax(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                                glp_set_col_name(lp, n+2, "A");
                                if(tempATmin(index) == tempATmax(index)){
                                    glp_set_col_bnds(lp, n+2, GLP_FX, tempATmin(index), tempATmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+2, GLP_DB, tempATmin(index), tempATmax(index));
                                }
                                glp_set_obj_coef(lp, n+2, 0.0);
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(tempATmin(index) == tempATmax(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, tempATmin(index), tempATmax(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    if (is_reach) {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    } else {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    }
                    break;
                }
                second0 = secondnew0;
                second1 = secondnew1;

            }else{
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer0(secondnew0.memptr(),secondnew0.n_rows);
                    sycl::buffer<double> cdfBuffer1(secondnew1.memptr(),secondnew1.n_rows);

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

                            size_t n = tempTmin.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second1(i-1));
                            }

                            if (is_reach) {
                                glp_set_col_name(lp, n+1, "T");
                                if(tempTTmin(index) == tempTTmax(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, tempTTmin(index), tempTTmax(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                                glp_set_col_name(lp, n+2, "A");
                                if(tempATmin(index) == tempATmax(index)){
                                    glp_set_col_bnds(lp, n+2, GLP_FX, tempATmin(index), tempATmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+2, GLP_DB, tempATmin(index), tempATmax(index));
                                }
                                glp_set_obj_coef(lp, n+2, 0.0);
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(tempATmin(index) == tempATmax(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, tempATmin(index), tempATmax(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);

                            cdfAccessor0[index] = 0;
                            cdfAccessor1[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, i)*second0(i-1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, i)*second1(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            } else {
                                cdfAccessor0[index] += glp_get_col_prim(lp, n+1);
                                cdfAccessor1[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                if((approx_equal(second1, secondnew1, "absdiff", 1e-8)) and ((approx_equal(second0, secondnew0, "absdiff", 1e-8)))){
                    if (is_reach) {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is an absorbing state in the solution, try running the finite Horizon solution using this number of steps." << endl;
                    } else {
                        cout << "Bounds both converged after " << converge << " steps, but they did not converge to each other. It is likely there is a safe solution, try running the finite Horizon solution using this number of steps." << endl;
                    }
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
        if (is_reach) {
            controller.col(dim_x+dim_u) = first0;
            controller.col(dim_x+dim_u + 1) = second1;
        } else {
            controller.col(dim_x+dim_u) = ones(state_space_size) - first0;
            controller.col(dim_x+dim_u + 1) = ones(state_space_size) - second1;
        }
        for (size_t i = 0; i < state_space_size; ++i) {
            controller.row(i).cols(dim_x, dim_x + dim_u - 1) = input_space.row(U_pos(i));
        }
    }

    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/// infinite horizon reachability synthesis
void IMDP::infiniteHorizonReachController(bool IMDP_lower) {
    infiniteHorizonControllerImpl(IMDP_lower, true);
}


/// infinite horizon safety synthesis - DEPRECATED: Original implementation below replaced by infiniteHorizonSafeController wrapper
/// The following commented block contains the original infiniteHorizonReachController implementation
/// that has been replaced by infiniteHorizonControllerImpl(IMDP_lower, true) above.
/// This is kept for reference during validation, and should be deleted after verification.


/// infinite horizon safety synthesis
void IMDP::infiniteHorizonSafeController(bool IMDP_lower) {
    infiniteHorizonControllerImpl(IMDP_lower, false);
}


/// infinite horizon safety synthesis - DEPRECATED: Original implementation
/// that has been replaced by infiniteHorizonControllerImpl(IMDP_lower, false) above.
/// This is kept for reference during validation, and should be deleted after verification.

/// Internal implementation for finite horizon controller synthesis (reach and safe)
void IMDP::finiteHorizonControllerImpl(bool IMDP_lower, size_t timeHorizon, bool is_reach) {
    auto start = chrono::steady_clock::now();
    if (is_reach) {
        cout << "Finding control policy for finite horizon reach controller... " << endl;
    } else {
        cout << "Finding control policy for finite horizon safe controller... " << endl;
        cout << "Approximate memory required if stored (each): " << minTargetM.n_rows*sizeof(double)/1000000.0 << "Mb, " << minTargetM.n_rows*sizeof(double)/1000000000.0 << "Gb" << endl;
    }

    // Configuration: Reach uses n+2 columns (P + Target + Avoid), Safe uses n+1 (P + Avoid)
    // Both use same LP direction: GLP_MIN for IMDP_lower=true, GLP_MAX for IMDP_lower=false

    if(input_space_size == 0 && disturb_space_size == 0){
        // Initial value: zeros for reach, ones for safe
        vec first(state_space_size);
        vec firstnew(state_space_size);
        if (is_reach) {
            first.fill(0.0);
            firstnew.fill(0.0);
        } else {
            first.fill(1.0);
            firstnew.fill(1.0);
        }

        size_t k = 0;
        cout << "first loop iterations: " << endl;
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){ // for IMDP lower bound
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);

                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 0.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);
                            cdfAccessor[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();
                first = firstnew;
            }else{ // for IMDP upper bound
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);

                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);

                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 0.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);
                            cdfAccessor[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor[index] += glp_get_col_prim(lp, n+1);
                            }
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
            cout << "Lower bound found." << endl;
        }else{
            cout << "Upper bound found." << endl;
        }

        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = first;

    }else if(input_space_size == 0){
        // Disturbance only case - verification with disturbance
        vec first(state_space_size);
        mat firstnew(state_space_size*disturb_space_size, 1);
        if (is_reach) {
            first.fill(0.0);
            firstnew.fill(0.0);
        } else {
            first.fill(1.0);
            firstnew.fill(1.0);
        }

        size_t k = 0;
        cout << "first loop iterations: " << endl;
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);

                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);

                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 0.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);
                            cdfAccessor[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                // Resize and reduce over disturbances - worst case
                firstnew.reshape(state_space_size, disturb_space_size);
                first = conv_to<colvec>::from(min(firstnew, 1));
                firstnew.reshape(state_space_size*disturb_space_size, 1);
            }else{
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);

                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*disturb_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);

                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 0.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);
                            cdfAccessor[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                firstnew.reshape(state_space_size, disturb_space_size);
                first = conv_to<colvec>::from(min(firstnew, 1));
                firstnew.reshape(state_space_size*disturb_space_size, 1);
            }
            k++;
        }
        cout << endl;

        if (IMDP_lower){
            cout << "Lower bound found." << endl;
        }else{
            cout << "Upper bound found." << endl;
        }

        controller.set_size(state_space_size, dim_x + 2);
        controller.cols(0,dim_x-1) = state_space;
        controller.col(dim_x) = first;

    }else{
        // Input-based synthesis (controller synthesis with inputs)
        vec first(state_space_size);
        mat firstnew(state_space_size*input_space_size, 1);
        uvec U_pos(state_space_size);

        vec second(state_space_size);
        mat secondnew(state_space_size*input_space_size, 1);
        if (is_reach) {
            first.fill(0.0);
            firstnew.fill(0.0);
            second.fill(0.0);
            secondnew.fill(0.0);
        } else {
            first.fill(1.0);
            firstnew.fill(1.0);
            second.fill(1.0);
            secondnew.fill(1.0);
        }

        size_t k = 0;
        cout << "first loop iterations: " << endl;
        while (k < timeHorizon) {
            cout << "." << flush;
            if (IMDP_lower == true){
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);

                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);

                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 0.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);
                            cdfAccessor[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                // Resize and reduce over inputs
                firstnew.reshape(state_space_size, input_space_size);
                if (is_reach) {
                    first = conv_to<colvec>::from(max(firstnew, 1));
                    U_pos = index_max(firstnew, 1);
                } else {
                    first = conv_to<colvec>::from(min(firstnew, 1));
                    U_pos = index_min(firstnew, 1);
                }
                firstnew.reshape(state_space_size*input_space_size, 1);
            }else{
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer(firstnew.memptr(),firstnew.n_rows);

                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size*input_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);

                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);

                            size_t n = minTransitionM.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(minTransitionM.row(index)(i - 1) == maxTransitionM.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, minTransitionM.row(index)(i - 1), maxTransitionM.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, first(i-1));
                            }

                            if (is_reach) {
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
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(minAvoidM(index) == maxAvoidM(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, minAvoidM(index), maxAvoidM(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, minAvoidM(index), maxAvoidM(index));
                                }
                                glp_set_obj_coef(lp, n+1, 0.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);
                            cdfAccessor[index] = 0;

                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*first(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                firstnew.reshape(state_space_size, input_space_size);
                if (is_reach) {
                    first = conv_to<colvec>::from(max(firstnew, 1));
                    U_pos = index_max(firstnew, 1);
                } else {
                    first = conv_to<colvec>::from(min(firstnew, 1));
                    U_pos = index_min(firstnew, 1);
                }
                firstnew.reshape(state_space_size*input_space_size, 1);
            }
            k++;
        }
        cout << endl;

        if (IMDP_lower){
            cout << "control policy for lower bound found, finding upper bound." << endl;
        }else{
            cout << "control policy for upper bound found, finding lower bound." << endl;
        }

        // Second phase with fixed controller from U_pos
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
        k = 0;
        while (k < timeHorizon) {
            cout << "." << flush;
            if (!IMDP_lower){
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);

                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);
                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MIN);

                            size_t n = tempTmin.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1));
                            }

                            if (is_reach) {
                                glp_set_col_name(lp, n+1, "T");
                                if(tempTTmin(index) == tempTTmax(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, tempTTmin(index), tempTTmax(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                                glp_set_col_name(lp, n+2, "A");
                                if(tempATmin(index) == tempATmax(index)){
                                    glp_set_col_bnds(lp, n+2, GLP_FX, tempATmin(index), tempATmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+2, GLP_DB, tempATmin(index), tempATmax(index));
                                }
                                glp_set_obj_coef(lp, n+2, 0.0);
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(tempATmin(index) == tempATmax(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, tempATmin(index), tempATmax(index));
                                }
                                glp_set_obj_coef(lp, n+1, 0.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                secondnew.reshape(state_space_size, input_space_size);
                if (is_reach) {
                    second = conv_to<colvec>::from(max(secondnew, 1));
                } else {
                    second = conv_to<colvec>::from(min(secondnew, 1));
                }
                secondnew.reshape(state_space_size*input_space_size, 1);
            }else{
                sycl::queue queue;
                {
                    sycl::buffer<double> cdfBuffer(secondnew.memptr(),secondnew.n_rows);

                    queue.submit([&](sycl::handler& cgh) {
                        auto cdfAccessor = cdfBuffer.get_access<sycl::access::mode::discard_write>(cgh);

                        cgh.parallel_for<class minTarget_kernel>(sycl::range<1>(state_space_size), [=](sycl::item<1> item) {
                            size_t index = item.get_id(0);

                            glp_term_out(GLP_OFF);
                            glp_prob *lp;
                            lp = glp_create_prob();
                            glp_set_prob_name(lp, "SimpleLP");
                            glp_set_obj_dir(lp, GLP_MAX);

                            size_t n = tempTmin.row(index).n_cols;
                            size_t num_extra_cols = is_reach ? 2 : 1;
                            glp_add_cols(lp, n + num_extra_cols);

                            for (size_t i = 1; i <= n; ++i) {
                                glp_set_col_name(lp, i, ("P_" + to_string(i)).c_str());
                                if(tempTmin.row(index)(i - 1) == tempTmax.row(index)(i - 1)){
                                    glp_set_col_bnds(lp, i, GLP_FX, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }else{
                                    glp_set_col_bnds(lp, i, GLP_DB, tempTmin.row(index)(i - 1), tempTmax.row(index)(i - 1));
                                }
                                glp_set_obj_coef(lp, i, second(i-1));
                            }

                            if (is_reach) {
                                glp_set_col_name(lp, n+1, "T");
                                if(tempTTmin(index) == tempTTmax(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, tempTTmin(index), tempTTmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, tempTTmin(index), tempTTmax(index));
                                }
                                glp_set_obj_coef(lp, n+1, 1.0);
                                glp_set_col_name(lp, n+2, "A");
                                if(tempATmin(index) == tempATmax(index)){
                                    glp_set_col_bnds(lp, n+2, GLP_FX, tempATmin(index), tempATmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+2, GLP_DB, tempATmin(index), tempATmax(index));
                                }
                                glp_set_obj_coef(lp, n+2, 0.0);
                            } else {
                                glp_set_col_name(lp, n+1, "A");
                                if(tempATmin(index) == tempATmax(index)){
                                    glp_set_col_bnds(lp, n+1, GLP_FX, tempATmin(index), tempATmax(index));
                                }else{
                                    glp_set_col_bnds(lp, n+1, GLP_DB, tempATmin(index), tempATmax(index));
                                }
                                glp_set_obj_coef(lp, n+1, 0.0);
                            }

                            glp_add_rows(lp, 1);
                            glp_set_row_name(lp, 1, "Constraint");
                            glp_set_row_bnds(lp, 1, GLP_FX, 1.0, 1.0);
                            vector<int> ia = {0};
                            vector<int> ja(n + num_extra_cols + 1);
                            vector<double> ar(n + num_extra_cols + 1);
                            for (size_t i = 1; i <= n; ++i) {
                                ja[i] = i;
                                ar[i] = 1.0;
                            }
                            ja[n+1] = n+1;
                            ar[n+1] = 1.0;
                            if (is_reach) {
                                ja[n+2] = n+2;
                                ar[n+2] = 1.0;
                            }
                            glp_set_mat_row(lp, 1, n + num_extra_cols, &ja[0], &ar[0]);

                            glp_simplex(lp, nullptr);
                            cdfAccessor[index] = 0;
                            for (size_t i = 1; i <= n; ++i) {
                                cdfAccessor[index] += glp_get_col_prim(lp, i)*second(i-1);
                            }
                            if (is_reach) {
                                cdfAccessor[index] += glp_get_col_prim(lp, n+1);
                            }
                            glp_delete_prob(lp);
                        });
                    });
                }
                queue.wait_and_throw();

                secondnew.reshape(state_space_size, input_space_size);
                if (is_reach) {
                    second = conv_to<colvec>::from(max(secondnew, 1));
                } else {
                    second = conv_to<colvec>::from(min(secondnew, 1));
                }
                secondnew.reshape(state_space_size*input_space_size, 1);
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

    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count()/1000.0 << " seconds" << endl;
}

/// finite horizon reachability synthesis
void IMDP::finiteHorizonReachController(bool IMDP_lower, size_t timeHorizon) {
    finiteHorizonControllerImpl(IMDP_lower, timeHorizon, true);
}

/// finite horizon reachability synthesis - DEPRECATED: Original implementation
/// that has been replaced by finiteHorizonControllerImpl(IMDP_lower, timeHorizon, true) above.
/// This is kept for reference during validation, and should be deleted after verification.

/// finite horizon safety synthesis
void IMDP::finiteHorizonSafeController(bool IMDP_lower, size_t timeHorizon) {
    finiteHorizonControllerImpl(IMDP_lower, timeHorizon, false);
}

/// finite horizon safety synthesis - DEPRECATED: Original implementation
/// that has been replaced by finiteHorizonControllerImpl(IMDP_lower, timeHorizon, false) above.
/// This is kept for reference during validation, and should be deleted after verification.
