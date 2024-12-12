#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <nlopt.hpp>
#include <iomanip>
#include <hdf5/serial/hdf5.h>
#include <AdaptiveCpp/CL/sycl.hpp>
#include <chrono>
#include "MDP.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <armadillo>

namespace sycl = cl::sycl;
using namespace std;
using namespace arma;

/* MDP FUNCTIONS */

/// Constructor
MDP::MDP(const int x, const int u, const int w){
    dim_x = x;
    dim_u = u;
    dim_w = w;
}

///Function to turn {lb, ub, eta} into a discretised space using centering
void MDP::get_spaceC(mat& space, ivec& state_idx, const int& dim, const vec& lb, const vec& ub, const vec& eta) {
    vector<int> state_idx2(dim);
    vector<int> state_dim(dim);
    // Calculate the total state_dim
    int tot_states = 1;
    for (int i = dim-1; i >=0; --i) {
        state_idx2[i] = tot_states;
        state_dim[i] = static_cast<int>((ub(i) - lb(i)) / eta(i));
        tot_states *= state_dim[i];
    }
    // Create Armadillo matrix (initialize to zeros)
    space.set_size(tot_states, dim);
    // Create a SYCL queue to execute tasks
    sycl::queue queue(sycl::default_selector{});
    {
        // Get a buffer to access the matrix data
        sycl::buffer<double> buffer(space.memptr(),space.n_rows*space.n_cols);
        sycl::buffer<double> lb2(lb.memptr(),lb.n_elem);
        sycl::buffer<double> eta2(eta.memptr(),eta.n_elem);
        sycl::buffer<int> idx2(state_idx2.data(),dim);
        sycl::buffer<int> dim2(state_dim.data(),dim);
        // Submit a SYCL task to set the matrix elements in parallel
        queue.submit([&](sycl::handler& cgh) {
            auto acc = buffer.get_access<sycl::access::mode::discard_write>(cgh);
            auto acclb = lb2.get_access<sycl::access::mode::read>(cgh);
            auto acceta = eta2.get_access<sycl::access::mode::read>(cgh);
            auto accidx = idx2.get_access<sycl::access::mode::read>(cgh);
            auto accdim = dim2.get_access<sycl::access::mode::read>(cgh);
            // Define a parallel range using nd_range
            sycl::range<2> matrixRange(tot_states, dim);
            cgh.parallel_for<class SetMatrix>(matrixRange, [=](sycl::id<2> idx) {
                int x0 = idx[0];
                int x1 = idx[1];
                
                int index = x0 * dim + x1;
                
                int row = index%tot_states;
                int col = index/tot_states;
                
                acc[index] = acclb[col] + acceta[col]/2 +((row/accidx[col]) % accdim[col])*acceta[col];
            });
        });
    }
    // Wait for the SYCL queue to finish execution
    queue.wait_and_throw();
    
}


///Function to turn {lb, ub, eta} into a discretised space that is uncentered
void MDP::get_spaceU(mat& space, ivec& state_idx, const int& dim, const vec& lb, const vec& ub, const vec& eta) {
    vector<int> state_idx2(dim);
    vector<int> state_dim(dim);
    // Calculate the total state_dim
    int tot_states = 1;
    for (int i = dim-1; i >=0; --i) {
        state_idx2[i] = tot_states;
        state_dim[i] = static_cast<int>((ub(i) - lb(i)) / eta(i)) + 1;
        tot_states *= state_dim[i];
    }
    // Create Armadillo matrix (initialize to zeros)
    space.set_size(tot_states, dim);
    // Create a SYCL queue to execute tasks
    sycl::queue queue(sycl::default_selector{});
    {
        // Get a buffer to access the matrix data
        sycl::buffer<double> buffer(space.memptr(),space.n_rows*space.n_cols);
        sycl::buffer<double> lb2(lb.memptr(),lb.n_elem);
        sycl::buffer<double> eta2(eta.memptr(),eta.n_elem);
        sycl::buffer<int> idx2(state_idx2.data(),dim);
        sycl::buffer<int> dim2(state_dim.data(),dim);
        // Submit a SYCL task to set the matrix elements in parallel
        queue.submit([&](sycl::handler& cgh) {
            auto acc = buffer.get_access<sycl::access::mode::discard_write>(cgh);
            auto acclb = lb2.get_access<sycl::access::mode::read>(cgh);
            auto acceta = eta2.get_access<sycl::access::mode::read>(cgh);
            auto accidx = idx2.get_access<sycl::access::mode::read>(cgh);
            auto accdim = dim2.get_access<sycl::access::mode::read>(cgh);
            // Define a parallel range using nd_range
            sycl::range<2> matrixRange(tot_states, dim);
            cgh.parallel_for<class SetMatrix>(matrixRange, [=](sycl::id<2> idx) {
                int x0 = idx[0];
                int x1 = idx[1];
                
                int index = x0 * dim + x1;
                
                int row = index%tot_states;
                int col = index/tot_states;
                
                acc[index] = acclb[col] + ((row/accidx[col]) % accdim[col])*acceta[col];
            });
        });
    }
    // Wait for the SYCL queue to finish execution
    queue.wait_and_throw();
    
}

/// Functions to separate state_space, target_space and avoid_space
void MDP::separate(mat& base_space, const function<bool(const vec&)>& target_condition, mat& target_set, const function<bool(const vec&)>& avoid_condition, mat& avoid_set) {
    cout << "Separating Target and Avoid sets from state space." << endl;
    // Iterate through rows of matrix base_space to find target_set and avoid_set
    avoid_set.set_size(0,dim_x);
    target_set.set_size(0,dim_x);
    for (int i = 0; i < base_space.n_rows; ++i) {
        // Check the boolean condition
        if (avoid_condition(base_space.row(i).t())) {
            // If the condition is true, add the index to avoid_set
            avoid_set.insert_rows(avoid_set.n_rows, base_space.row(i));
            // Remove selected row from matrix base_space
            base_space.shed_row(i);
            --i;
        } else if (target_condition(base_space.row(i).t())) {
            // If the condition is true, add the index to target_set
            target_set.insert_rows(target_set.n_rows, base_space.row(i));
            // Remove selected row from matrix base_space
            base_space.shed_row(i);
            --i;
        }
    }
}

/// Functions to separate state_space and either target_space or avoid_space
void MDP::separate(mat& base_space, const function<bool(const vec&)>& separate_condition, mat& separate_set) {
    separate_set.set_size(0,dim_x);
    // Iterate through rows of matrix base_space to find separate_set
    for (int i = 0; i < base_space.n_rows; ++i) {
        // Check the boolean condition
        if (separate_condition(base_space.row(i).t())) {
            // If the condition is true, add the index to separate_set
            separate_set.insert_rows(separate_set.n_rows, base_space.row(i));
            // Remove selected row from matrix base_space
            base_space.shed_row(i);
            --i;
        }
    }
}

/* State Space Functions */

///Getters and Setters for State Space
void MDP::setStateSpace(vec lb, vec ub, vec eta){
    if (lb.size() == dim_x && ub.size() == dim_x && eta.size() == dim_x){
        cout << "State space is of correct dimension, saving state space." << endl;
        get_spaceU(state_space, ss_idx, dim_x, lb, ub, eta); //TODO: work out if centering or not!
        state_space_size = state_space.n_rows;
        cout << "State space size: " << state_space_size << endl;
        ss_lb = lb;
        ss_ub = ub;
        ss_eta = eta;
    }else{
        cout << "Error: state space dimensions don't match." << endl;
    }
}

mat MDP::getStateSpace(){
    return state_space;
}

void MDP::saveStateSpace(){
    if (state_space.empty()){
        cout << "State space is empty, can't save file." << endl;
    }else{
        state_space.save("ss.h5",hdf5_binary);
        cout << "saved in ss.h5." << endl;
    }
}

void MDP::loadStateSpace(string filename){
    bool ok = state_space.load(filename);
    if (ok == false){
        cout << "Issue loading state_space!" << endl;
    }else{
        state_space_size = state_space.n_rows;
        cout << "State space loaded" << endl;
    }
}

/* Input Space Functions */

/// Getters and Setters for Input Space
void MDP::setInputSpace(vec lb, vec ub, vec eta){
    if (lb.size() == dim_u && ub.size() == dim_u && eta.size() == dim_u){
        cout << "Input space is of correct dimension, saving input space." << endl;
        get_spaceU(input_space, is_idx, dim_u, lb, ub, eta);
        input_space_size = input_space.n_rows;
        cout << "Input space size: " << input_space_size << endl;
        is_lb = lb;
        is_ub = ub;
        is_eta = eta;
    }else{
        cout << "Error: input space dimensions don't match." << endl;
    }
}

mat MDP::getInputSpace(){
    return input_space;
}

void MDP::saveInputSpace(){
    if (input_space.empty()){
        cout << "Input space is empty, can't save file." << endl;
    }else{
        input_space.save("is.h5",hdf5_binary);
        cout << "saved in is.h5." << endl;
    }
}

void MDP::loadInputSpace(string filename){
    bool ok = input_space.load(filename);
    if (ok == false){
        cout << "Issue loading input space!" << endl;
    }else{
        input_space_size = input_space.n_rows;
        cout << "input space loaded" << endl;
    }
}

/* Disturb Space Functions*/

///Getters and Setters for Disturb Space
void MDP::setDisturbSpace(vec lb, vec ub, vec eta){
    if (lb.size() == dim_w && ub.size() == dim_w && eta.size() == dim_w){
        cout << "Disturb space is of correct dimension, saving disturb space." << endl;
        get_spaceU(disturb_space, ws_idx, dim_w, lb, ub, eta);
        disturb_space_size = disturb_space.n_rows;
        cout << "Disturb space size: " << disturb_space_size << endl;
        ws_lb = lb;
        ws_ub = ub;
        ws_eta = eta;
    }else{
        cout << "Error: disturb space dimensions don't match." << endl;
    }
}
mat MDP::getDisturbSpace(){
    return disturb_space;
}

void MDP::saveDisturbSpace(){
    if (disturb_space.empty()){
        cout << "Disturb space is empty, can't save file." << endl;
    }else{
        disturb_space.save("ws.h5",hdf5_binary);
        cout << "saved in ws.h5." << endl;
    }
}

void MDP::loadDisturbSpace(string filename){
    bool ok = disturb_space.load(filename);
    if (ok == false){
        cout << "Issue loading disturb space!" << endl;
    }else{
        disturb_space_size = disturb_space.n_rows;
        cout << "loaded disturbance space successfully." << endl;
    }
}

/* Target Space Functions */

///Getters and Setters for Target Space
void MDP::setTargetSpace(const function<bool(const vec&)>& separate_condition, bool remove){
    if (state_space.empty()){
        cout << "State space is empty, can't create target." << endl;
    }else if(remove){
        cout << "Setting target region... ";
        separate(state_space, separate_condition, target_space);
        cout << "Complete." << endl;
        state_space_size = state_space.n_rows;
        cout << "Updated state space size: " << state_space_size << endl;
        cout << "Target space size: " << target_space.n_rows << endl;
    }else{
        if(filter.is_vec()){
            filter.zeros(state_space_size);
        }
        filterTarget(state_space, separate_condition);
    }
}

mat MDP::getTargetSpace(){
    return target_space;
}

void MDP::saveTargetSpace(){
    if (target_space.empty()){
        cout << "Target space is empty, can't save file." << endl;
    }else{
        target_space.save("ts.h5",hdf5_binary);
        cout << "saved in ts.h5." << endl;
    }
}

void MDP::loadTargetSpace(string filename){
    bool ok = target_space.load(filename);
    if (ok == false){
        cout << "Issue loading target space!" << endl;
    }else{
        cout << "loaded target space" << endl;
    }
}

/* Avoid Space Functions*/

///Getters and Setters for Avoid Space
void MDP::setAvoidSpace(const function<bool(const vec&)>& separate_condition, bool remove){
    if (state_space.empty()){
        cout << "State space is empty, can't create target." << endl;
    }else if(remove){
        cout << "Setting avoid region... ";
        separate(state_space, separate_condition, avoid_space);
        cout << "Complete." << endl;
        state_space_size = state_space.n_rows;
        cout << "Updated state space size: " << state_space_size << endl;
        cout << "Avoid space size: " << avoid_space.n_rows << endl;
    }else{
        if(filter.is_vec()){
            filter.zeros(state_space_size);
        }
        filterAvoid(state_space, separate_condition);
    }
}

mat MDP::getAvoidSpace(){
    return avoid_space;
}

void MDP::saveAvoidSpace(){
    if (avoid_space.empty()){
        cout << "Avoid space is empty, can't save file." << endl;
    }else{
        avoid_space.save("as.h5",hdf5_binary);
        cout << "saved in as.h5." << endl;
    }
}

void MDP::loadAvoidSpace(string filename){
    bool ok = avoid_space.load(filename);
    if (ok == false){
        cout << "Issue loading avoid space!" << endl;
    }else{
        cout << "loaded avoid space" << endl;
    }
}

/* Joint Target and Avoid Space Functions */

///Set Target and Avoid Spaces together
void MDP::setTargetAvoidSpace(const function<bool(const vec&)>& target_condition,const function<bool(const vec&)>& avoid_condition, bool remove){
    if (state_space.empty()){
        cout << "State space is empty, can't create target and avoid regions." << endl;
    }else if(remove){
        cout << "Setting target and avoid regions... ";
        separate(state_space, target_condition, target_space, avoid_condition, avoid_space);
        cout << "Complete." << endl;
        state_space_size = state_space.n_rows;
        cout << "Updated state space size: " << state_space_size << endl;
        cout << "Avoid space size: " << avoid_space.n_rows << endl;
        cout << "Target space size: " << target_space.n_rows << endl;
    }else{
        filter.zeros(state_space_size);
        filterTargetAvoid(state_space, target_condition, avoid_condition);
    }
}

/* Dynamics, Noise and Other Parameters */

///Setters for Dynamics
void MDP::setDynamics(const function<vec(const vec&, const vec&, const vec&)> d){
    if(dim_x != 0 && dim_u != 0 && dim_w != 0){
        dynamics3 = d;
    }else{
        cout << "Provided dynamics function with 3 parameters which doesn't match MDP description." << endl;
    }
}
void MDP::setDynamics(const function<vec(const vec&, const vec&)> d){
    if((dim_x != 0 && dim_u != 0 && dim_w == 0) || (dim_x != 0 && dim_w != 0 && dim_u == 0)){
        dynamics2 = d;
    }else{
        cout << "Provided dynamics function with 2 parameters which doesn't match MDP description." << endl;
    }
}
void MDP::setDynamics(const function<vec(const vec&)> d){
    dynamics1 = d;
    if(dim_x != 0 && dim_u == 0 && dim_w == 0){
        dynamics1 = d;
    }else{
        cout << "Provided dynamics function with 1 parameter which doesn't match MDP description." << endl;
    }
}

///Setters for NoiseType and Properties
void MDP::setNoise(NoiseType n, bool ind){
    noise = n;
    if(ind != true){
        cout<<"Poor noise choise set, for offdiagonal noise you need to choose a number of monte carlo samples and include this as an additional function parameter: setNoise(NoiseType n, bool ind, size_t monte_carlo_samples)."<<endl;
        
    }
    diagonal = ind;
}
void MDP::setNoise(NoiseType n, bool ind, size_t monte_carlo_samples){
    noise = n;
    diagonal = ind;
    calls = monte_carlo_samples;
}

///Setter for Inverse Covariance and Determinant
void MDP::setInvCovDet(mat inv_cov, double det){
    inv_covariance_matrix = inv_cov;
    covariance_matrix_determinant = det;
}

///Setter for Standard Deviation
void MDP::setStdDev(vec sig){
    sigma = sig;
}

///Setter for Stopping Condition
void MDP::setStoppingCondition(double eps){
    epsilon = eps;
}

///Getter for Inverse Covariance matrix
mat MDP::getInvCov(){
    return inv_covariance_matrix;
}

///Getter for Determinant
double MDP::getDet(){
    return covariance_matrix_determinant;
}

///Getter for Standard Deviation
vec MDP::getStdDev(){
    return sigma;
}

///Setter for Custom Distribution
void MDP::setCustomDistribution(double (*c)(double*, size_t, void*), size_t monte_carlo_samples){
    calls = monte_carlo_samples;
    customPDF = c;
    cout << "custom PDF stored and samples stored" << endl;
}

/* Filter Functions (probably not used unless not removing states from state space) */
void MDP::filterAvoid(mat& base_space, const function<bool(const vec&)>& separate_condition) {
    int count = 0;
    // Iterate through rows of matrix base_space to find separate_set
    for (int i = 0; i < base_space.n_rows; ++i) {
        // Check the boolean condition
        if (separate_condition(base_space.row(i).t())) {
            // If the condition is true, add the index to separate_set
            filter(i) = -1;
            count++;
        }
    }
    cout << "Avoid space size: " << count << endl;
}

void MDP::filterTarget(mat& base_space, const function<bool(const vec&)>& separate_condition) {
    int count = 0;
    // Iterate through rows of matrix base_space to find separate_set
    for (int i = 0; i < base_space.n_rows; ++i) {
        // Check the boolean condition
        if (separate_condition(base_space.row(i).t())) {
            // If the condition is true, add the index to separate_set
            filter(i) = 1;
            count++;
        }
    }
    cout << "Target space size: " << count << endl;
}

void MDP::filterTargetAvoid(mat& base_space, const function<bool(const vec&)>& target_condition, const function<bool(const vec&)>& avoid_condition) {
    int countA = 0;
    int countT = 0;
    // Iterate through rows of matrix base_space to find separate_set
    for (int i = 0; i < base_space.n_rows; ++i) {
        // Check the boolean condition
        if (target_condition(base_space.row(i).t())) {
            // If the condition is true, add the index to separate_set
            filter(i) = 1;
            countT++;
        }else if(avoid_condition(base_space.row(i).t())){
            filter(i) = -1;
            countA++;
        }
    }
    cout << "Avoid space size: " << countA << endl;
    cout << "Target space size: " << countT << endl;
}

/* Load and Save Transition Matrices */
///Save minimal transition matrix
void MDP::saveTransitionMatrix(){
    if (TransitionM.empty()){
        cout << "Transition Matrix is empty, can't save file." << endl;
    }else{
        TransitionM.save("tm.h5", hdf5_binary);
    }
}

///Load minimal transition matrix
void MDP::loadTransitionMatrix(string filename){
    bool ok = TransitionM.load(filename);
    if (ok == false){
        cout << "Issue loading transition matrix!" << endl;
    }
}

///Save maximal target transition vector
void MDP::saveTargetTransitionVector(){
    if (TargetM.empty()){
        cout << "Target Transition Vector is empty, can't save file." << endl;
    }else{
        TargetM.save("ttm.h5", hdf5_binary);
    }
}

///Load maximal target transition vector
void MDP::loadTargetTransitionVector(string filename){
    bool ok = TargetM.load(filename);
    if (ok == false){
        cout << "Issue loading target transition Vector!" << endl;
    }
}

///Save maximal avoid transition vector
void MDP::saveAvoidTransitionVector(){
    if (AvoidM.empty()){
        cout << "Avoid Transition Vector is empty, can't save file." << endl;
    }else{
        AvoidM.save("atm.h5", hdf5_binary);
    }
}

///Load maximal avoid transition vector
void MDP::loadAvoidTransitionVector(string filename){
    bool ok = AvoidM.load(filename);
    if (ok == false){
        cout << "Issue loading avoid transition Vector!" << endl;
    }
}