#ifndef MDP_H
#define MDP_H

#include <armadillo>

using namespace std;
using namespace arma;

/* MDP Parameters */

/// Type of Noise
enum class NoiseType {NORMAL, CUSTOM};

/// custom Parameter Struct used for CustomPDF
struct customParams {
    vec mean;
    vec state_start;
    function<vec(const vec&)> dynamics1;
    function<vec(const vec&,const vec&)> dynamics2;
    function<vec(const vec&,const vec&,const vec&)> dynamics3;
    vec input;
    vec disturb;
    vec lb;
    vec ub;
    vec eta;
};

/* MDP class */
class MDP {
    
    /* Protected Variables */
    
protected:
    /// Dimensions
    size_t dim_x;
    size_t dim_u;
    size_t dim_w;
    
    /// States
    mat state_space;
    vec ss_lb;
    vec ss_ub;
    vec ss_eta;
    ivec ss_idx;
    vec filter;
    
    /// Inputs
    mat input_space;
    vec is_lb;
    vec is_ub;
    vec is_eta;
    ivec is_idx;
    
    /// Disturbances
    mat disturb_space;
    vec ws_lb;
    vec ws_ub;
    vec ws_eta;
    ivec ws_idx;
    
    ///Size of spaces
    size_t state_space_size = 0;
    size_t input_space_size = 0;
    size_t disturb_space_size = 0;
    
    ///Target and Avoid Spaces
    mat target_space;
    mat avoid_space;
    
    ///Noise and Integration Parameters
    bool diagonal;
    NoiseType noise;
    vec sigma;
    mat inv_covariance_matrix;
    double covariance_matrix_determinant;
    size_t calls;
    
    ///Dynamics
    function<vec(const vec&, const vec& , const vec&)> dynamics3;
    function<vec(const vec&, const vec&)> dynamics2;
    function<vec(const vec&)> dynamics1;

    function<double(double *x, size_t dim, void *params)> customPDF;

    ///Stopping Condition
    double epsilon = 0.00001;
    
    /* Private Functions */
    
private:
    ///Function to turn {lb, ub, eta} into a discretized space
    void get_spaceC(mat& space, ivec& state_idx, const int& dim, const vec& lb, const vec& ub, const vec& eta);
    void get_spaceU(mat& space, ivec& state_idx, const int& dim, const vec& lb, const vec& ub, const vec& eta);
    
    ///Functions to seperate state_space, target_space and avoid_space
    void separate(mat& base_space, const function<bool(const vec&)>& target_condition, mat& target_set, const function<bool(const vec&)>& avoid_condition, mat& avoid_set);
    void separate(mat& base_space, const function<bool(const vec&)>& separate_condition, mat& separate_set);
    
    ///Functions to filter if not removing states (unlikely to be used).
    void filterTargetAvoid(mat& base_space, const function<bool(const vec&)>& target_condition, const function<bool(const vec&)>& avoid_condition);
    void filterTarget(mat& base_space, const function<bool(const vec&)>& separate_condition);
    void filterAvoid(mat& base_space, const function<bool(const vec&)>& separate_condition);
    
    /* Public Functions */
    
public:
    /// Constructor
    MDP(const int x, const int u, const int w);
    
    ///Getters and Setters for State Space
    void setStateSpace(vec lb, vec ub, vec eta);
    mat getStateSpace();
    void saveStateSpace();
    void loadStateSpace(string filename);
    
    ///Getters and Setters for Input Space
    void setInputSpace(vec lb, vec ub, vec eta);
    mat getInputSpace();
    void saveInputSpace();
    void loadInputSpace(string filename);
    
    ///Getters and Setters for Disturb Space
    void setDisturbSpace(vec lb, vec ub, vec eta);
    mat getDisturbSpace();
    void saveDisturbSpace();
    void loadDisturbSpace(string filename);
    
    ///Getters and Setters for Target Space
    void setTargetSpace(const function<bool(const vec&)>& separate_condition, bool remove);
    mat getTargetSpace();
    void saveTargetSpace();
    void loadTargetSpace(string filename);
    
    ///Getters and Setters for Avoid Space
    void setAvoidSpace(const function<bool(const vec&)>& separate_condition, bool remove);
    mat getAvoidSpace();
    void saveAvoidSpace();
    void loadAvoidSpace(string filename);
    
    ///Set Target and Avoid Spaces together
    void setTargetAvoidSpace(const function<bool(const vec&)>& target_condition,const function<bool(const vec&)>& avoid_condition, bool remove);
    
    ///Setters for Dynamics
    void setDynamics(const function<vec(const vec&, const vec&, const vec&)> d);
    void setDynamics(const function<vec(const vec&, const vec&)> d);
    void setDynamics(const function<vec(const vec&)> d);
    
    ///Setters for Noise and Integration Parameters
    void setInvCovDet(mat inv_cov, double det);
    void setStdDev(vec sig);
    void setNoise(NoiseType n, bool diagonal = true);
    void setNoise(NoiseType n, bool diagonal, size_t monte_carlo_samples);
    void setCustomDistribution(const function<double(double *x, size_t dim, void *params)> c,size_t monte_carlo_samples);
    
    ///Getters for Noise Parameters
    mat getInvCov();
    double getDet();
    vec getStdDev();
    
    ///Setter for stopping condition
    void setStoppingCondition(double eps);
};
#endif
