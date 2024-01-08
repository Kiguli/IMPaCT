# IMPaCT Setup

In this file, I hope to provide all the necessary details required for running the case studies and also creating your own case studies to be used by **IMPaCT**. Each case study is run via a configuration file such as `robot2D.cpp` in the example [ex_2Drobot-R-U](./examples/ex_2Drobot-R-U). **IMPaCT** is designed to be as user friendly as possible, particularly for those who are maybe less familiar with programming. The trickiest part therefore is likely to be with the configuration of the Makefiles - I leave the discussions around this until the very end of this file.

In addition, the functions are named primarily for the synthesis of interval Markov decision processes (IMDPs) but the functions are also useful for verification of interval Markov chains (IMCs). The tool will automatically detect when there is no input present within the system formulation and return to the user a lookup table with the verification satisfaction probabilities instead of the usual lookup table controller with both optimal actions and satisfaction probabilities. This can be seen in some of the examples such as the [7D BAS Verification Example](./examples/ex_7DBAS-S).

**IMPaCT** is also very flexible as it can enable the loading in of transition matrices that have been computed elsewhere, in order to be used for either infinite-time horizon or finite-time horizon synthesis within **IMPaCT**. This is powerful as it enables **IMPaCT** to be useful for data-driven methods.

Please compare the instructions in the setup with the examples given in the repository in order to best understand what is going on. The majority of examples provide the code for both infinite-time horizon and finite-time horizon with one choice commented out. Below is a list of the type of case studies that the user may wish to use and an example demonstrating that type of case study:
- Verification: [ex_7DBAS-S](./examples/ex_7DBAS-S)
- Reachability *without* Disturbance: [ex_2Drobot-R-U](./examples/ex_2Drobot-R-U)
- Reachability *with* Disturbance: [ex_2Drobot-R-D](./examples/ex_2Drobot-R-D)
- Reach-while-Avoid *without* Disturbance: [ex_2Drobot-RA-U](./examples/ex_2Drobot-RA-U)
- Reach-while-Avoid *with* Disturbance: [ex_2Drobot-RA-U](./examples/ex_2Drobot-RA-D)
- Safety: [ex_4DBAS-S](./examples/ex_4DBAS-S)
- multivariate normal distributions: [ex_multivariateNormalPDF](./examples/ex_multivariateNormalPDF)
- custom distributions: [ex_customPDF](./examples/ex_customPDF)
- Loading source files into **IMPaCT** for synthesis: [ex_load_reach](./examples/ex_load_reach)
- Dealing with Absorbing states: [ex_load_safe](./examples/ex_load_safe)

# Table of Contents
- [Running an Example](#running-an-example)
- [Constructing IMDP/IMC](#constructing-imdpimc)
- [Noise Distributions](#noise-distributions)
- [State Space and Specification](#state-space-and-specification)
- [Dynamics](#dynamics)
- [Compute Abstraction](#compute-abstraction)
- [Verification and Synthesis](#verification-and-synthesis)
- [Loading and Saving Files](#loading-and-saving-files)
- [Makefiles](#makefiles)

# Running an Example

In general one should be able to go into a folder of any of the examples where there will be some `.cpp` file e.g. `example.cpp` which contains the configuration file for the case study and also a make file which will compile and then run the code. To run the case study, open up a terminal or command line interface and run:

`make` which will compile the code followed by `./example` which will execute the new file. The name of the executable file will match the file `*.cpp` but without the `.cpp` part at the end.

When looking at the output, its possible there appear lots of debug warnings, these can be removed by running `export ACPP_DEBUG_LEVEL=0`.

# Constructing IMDP/IMC

First, it is important to always define an IMDP object for both IMDP/IMC problems using:

`IMDP(const int x, const int u, const int w);`

When the first parameter `x` is the dimensions of the state space, the second parameter `u` is the dimension of the input space, and the third parameter `w` is the dimension of the disturbance space. When `u=0`, the problem is considered an IMC verification problem and the tool will automatically adjust all the following functions to be applicable for verification instead of synthesis.

### Demonstration
For the setup we will consider that we label this object `imdp`, so for our new implementation we should call:

`IMDP imdp(x,u,w);`

This object `imdp` will then be calling all the future functions by using a `.` for the functions that exist within the object class:
e.g. 

`imdp.setNoiseType(NORMAL);` 

# Noise Distributions

**IMPaCT** considers by default normal distributions and has built-in functionality for both diagonal covariance matrices (*the default*) and full covariance matrices with nonzero offdiagonal values. **IMPaCT** also enables the use of custom distributions if the user provides the PDF function that will be integrated inside of the tool. The integration is computed via Monte Carlo integration due to its scalability against other implementations, therefore the user is expected to provide the number of samples for this integration for both the multivariate normal distribution and custom distribution environments. For the normal distribution with diagonal covariance matrix, the product of the closed-form cumulative distribution function of each dimension independently is calculated. This also improves the computation time for the abstractions.

There is an `enum` called `NoiseType` which is used to describe the two noise distribution options:

`enum class NoiseType {NORMAL, CUSTOM};`

The NoiseType can then be set for the IMDP object by using either:

`void setNoise(NoiseType n, bool diagonal = true);`

`void setNoise(NoiseType n, bool diagonal, size_t monte_carlo_samples);`

Other parameters can then be set that provide further details about each noise distribution.

For normal distributions with diagonal covariance matrix, the standard deviation is set using: 

`void setStdDev(vec sig);`

For multivariate noise with full covariance matrix the inverse covariance matrix and the determinant are set using:

`void setInvCovDet(mat inv_cov, double det);`

For custom distributions the file `./src/custom.cpp` should be editted appropriately [here](./src/custom.cpp) (the default is a multivariate normal distribution) which includes the PDF of the custom distribution. Some useful parameters that may be needed can be used by the function which come from the `struct` called `customParams` including the state being passed to the customPDF function (`state_start`), the dynamics of the system (`dynamics1`, `dynamics2` or `dynamics3` depending on number of parameters in the IMDP), the input (`input`), the disturbance (`disturb`), the lower bound and upper bound of the integration (`lb` and `ub` respectively), the discretization parameter (`eta`), and the result of running the dynamics on the current state, input, disturbance triple (`mean`).

The number of samples in the integration can then be set by:

`void setCustomDistribution(size_t monte_carlo_samples);`

### Demonstration 

For normal distribution with diagonal covariance:

`imdp.setNoise(NoiseType::NORMAL);`

`imdp.setStdDev(vec sig);`

For normal distribution with full covariance with 1000 Monte Carlo samples:

`imdp.setNoise(NoiseType::NORMAL, false, 1000);`

`imdp.setInvCovDet(mat inv_cov, double det);`

For custom distribution with 1000 Monte Carlo samples (after changing `./src/custom.cpp`):

`imdp.setNoise(NoiseType::CUSTOM)`

`imdp.setCustomDistribution(1000);`

or simply

`imdp.setNoise(NoiseType::CUSTOM, false, 1000);`

# State Space and Specification

The state space, input space, and disturbance space can be defined each by a hyper-rectangle. Three vectors are defined each with length equal to the number of dimensions of the space. Vector `lb` is each lower bound of each dimension, `ub` is the upper bounds of each dimension, and `eta` is the discretization parameter of each dimension. If either or both of the input space and disturbance space are not present in the system then the functions calls can be simply omitted from the configuration file.

`void setStateSpace(vec lb, vec ub, vec eta);`

`void setInputSpace(vec lb, vec ub, vec eta);`

`void setDisturbSpace(vec lb, vec ub, vec eta);`

When considering the specifications, the defined state space is considered by default to be the safe region of the system, for safety specifications no further steps are therefore necessary. For reachability and reach-while-avoid specifications, if the state space contains any states that should be considered target region states or avoid region states, a boolean function should be used to define these states and then the following functions will seperate the states from the state space and move them to a new space called the target space or avoid space dependent on the description.

`void setTargetSpace(const function<bool(const vec&)>& separate_condition, bool remove);`

`void setAvoidSpace(const function<bool(const vec&)>& separate_condition, bool remove);`

When both target and avoid states are present both these steps can be done independently, or a function to do both simultaneously is provided:

`void setTargetAvoidSpace(const function<bool(const vec&)>& target_condition,const function<
bool(const vec&)>& avoid_condition, bool remove);`

Optionally the states that make the target or avoid region do not need to be removed from the state space (set `remove=false`, there may be some circumstances where this is a useful feature, e.g. to check how many states fulfill a certain property.

### Demonstration

As a simple example, a state space with 2 dimensions can be defined where each vector is first defined seperately and then passed the the `IMDP` object.

`const vec ss_lb = {-10, -10};`

`const vec ss_ub = {10, 10};`

`const vec ss_eta = {1, 1};`

`imdp.setStateSpace(ss_lb, ss_ub, ss_eta);`

The boolean function for the target region can be defined as such where the target region is described by a square where the lower left coordinate is (5,5) and the upper right coordinate is (8,8).

```
auto target_condition = [](const vec& ss) { 
return (ss[0] >= 5.0 && ss[0] <= 8.0) && (ss[1] >= 5.0 && ss[1] <= 8.0);
};
```

This boolean function is then passed to the `IMDP` object and the states are removed from the state space and added to the target space.

`imdp.setTargetSpace(target_condition, true);`

# Dynamics

The dynamics of the system are designed and passed to the `IMDP` object in a similar way to the boolean target space. A function should be described where the number of parameters for the function matches the number of parameters of the `IMDP` object.

`function<vec(const vec&, const vec& , const vec&)> dynamics3;`

`function<vec(const vec&, const vec&)> dynamics2;`

`function<vec(const vec&)> dynamics1;`

### Demonstration

The following dynamics describe an `IMDP` object with a state space and an input space but no disturbance space. Notice, there are only two parameters in the function definition.

```
auto dynamics = [](const vec& x, const vec& u) -> vec {
    vec xx(dim_x);
    xx[0] = x[0] + 2*u[0]*cos(u[1]);
    xx[1] = x[1] + 2*u[0]*sin(u[1]);
    return xx;
    };
```

The dynamics can then be added using (the function detects automatically the number of parameters):

 `imdp.setDynamics(dynamics);`

# Compute Abstraction

IMDP abstraction consists of a nonlinear optimization for each state to state transition within the system. This occurs twice, once for the minimal transition probabilities and the second time for the maximal transition probabilities.

Firstly, the algorithm used for the nonlinear optimization can be set using the function:

`void setAlgorithm(nlopt::algorithm alg);`

where the default choice is `nlopt::LN_SBPLX` but any other algorithm can be used, see [NLopt algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/) for more options.

At this point the abstraction step for each of the matrices can occur. It is always necessary to run:

`void minTransitionMatrix();`

`void maxTransitionMatrix();`

`void minAvoidTransitionVector();`

`void maxAvoidTransitionVector();`

as these algorithms do the abstraction for the transitions inside of the state space and also the transitions outside of the state space (even if the avoid region is empty). Optionally, if a target region has been set for reachability or reach-while-avoid specifications then the following functions should also be called:

`void minTargetTransitionVector();`

`void maxTargetTransitionVector();`

As a nice implementation trick, for each of the transition probabilities that is calculated, if any matrix/vector element is zero for the maximal case, then it is impossible for the minimal case to not also be zero. This can provide a beneficial speedup of implementation to avoid needing to run the nonlinear optimization step in those cases. We can do this low-cost abstraction for the transition matrix and for the target transition vector using:

`void transitionMatrixBounds();`

`void targetTransitionVectorBounds();`

The more sparse the transition matrices are, the greater the performance this computational trick will provide the user.

# Verification and Synthesis

We provide two different synthesis algorithms for the different specifications dependent on whether the specifications are over an infinite-time horizon or a finite-time horizon. For infinite-time horizon the *interval iteration* algorithm is implemented which provides guarantees on convergence of the systems, unlike the more common *value iteration*, see  for more details. For the finite-time horizon specifications, we use the *value iteration* algorithm as convergence is not required and it is computationally more efficient, requiring only one Bellman equation instead of two, again we refer to [our paper](./IMPaCT-Paper_arXiv.pdf) for details.

Both algorithms are implemented for safety problems and also for reachability/reach-while-avoid problems. The functions automatically detect which of reachability and reach-while-avoid specifications are being used dependent on the avoid region being an empty set or not. Safety algorithms are implemented in a slightly different way to reachability, and so seperate functions are necessary.

For all the verification and synthesis functions, a boolean `IMDP_lower` needs to be selected that chooses either a pessimistic (`true`) or optimistic (`false`) control strategy. Pessimistic controllers considering the worst case noise and the satisfaction probability lower bound when finding the optimal control inputs, before fixing the input to find the satisfaction probability upper bound. Optimistic control policies consider the reverse. For finite-time horizon verification and synthesis, a second parameter for the number of time steps to consider is also required.

Finally, once again we highlight that the tool automatically detects if there is an input present in the system to whether it provides a lookup table controller or a lookup table for the verification probabilities of each state. The same functions are used in both cases.

`void infiniteHorizonReachController(bool IMDP_lower);`

`void infiniteHorizonSafeController(bool IMDP_lower);`

`void finiteHorizonReachController(bool IMDP_lower, size_t timeHorizon);`

`void finiteHorizonSafeController(bool IMDP_lower, size_t timeHorizon);`

# Loading and Saving Files

As mentioned briefly before, **IMPaCT** is very flexible and enables the user to compute some parts of the abstraction, verification and/or synthesis elsewhere and load these into **IMPaCT** for the remaining steps. **IMPaCT** loads and saves files each in a seperate [HDF5](https://www.neonscience.org/resources/learning-hub/tutorials/about-hdf5#:~:text=Supports%20Large%2C%20Complex%20Data%3A%20HDF5,%2C%20heterogeneous%2C%20and%20complex%20datasets.) file. The field parameter is by default called `dataset`. The [HDF5](https://www.neonscience.org/resources/learning-hub/tutorials/about-hdf5#:~:text=Supports%20Large%2C%20Complex%20Data%3A%20HDF5,%2C%20heterogeneous%2C%20and%20complex%20datasets.) format is especially useful as it natively supported by many applications and languages such as [HDF5 for MATLAB](https://uk.mathworks.com/help/matlab/hdf5-files.html), [HDF5 for Python](https://docs.h5py.org/en/stable/), [HDF5 for R](https://www.bioconductor.org/packages/devel/bioc/vignettes/rhdf5/inst/doc/rhdf5.html), etc.

In addition, loading the transition matrix files that have been computed using data, or other methods, means **IMPaCT** is flexible beyond just model-based system analysis and controller design. 

The following functions can be used to load and save the different components of **IMPaCT** to HDF5 files. Use the links above to see how they can be used in the other tools. Also check out example [ex_load_reach](./examples/ex_load_reach) which shows how the matrices and vectors can be loaded into **IMPaCT** for synthesis and verification. 

`void loadStateSpace(string filename);`

`void loadInputSpace(string filename);`

`void loadDisturbSpace(string filename);`

`void loadTargetSpace(string filename);`

`void loadAvoidSpace(string filename);`

`void loadMinTargetTransitionVectorx(string filename);`

`void loadMaxTargetTransitionVector(string filename);`

`void loadMinAvoidTransitionVector(string filename);`

`void loadMaxAvoidTransitionVector(string filename);`

`void loadMinTransitionMatrix(string filename);`

`void loadMaxTransitionMatrix(string filename);`

`void loadController(string filename);`

`void saveStateSpace();`

`void saveInputSpace();`

`void saveDisturbSpace();`

`void saveTargetSpace();`

`void saveAvoidSpace();`

`void saveMinTargetTransitionVector();`

`void saveMaxTargetTransitionVector();`

`void saveMinAvoidTransitionVector();`

`void saveMaxAvoidTransitionVector();`

`void saveMinTransitionMatrix();`

`void saveMaxTransitionMatrix();`

`void saveController();`

# Makefiles

Makefiles always seem to be generally a bit tricky and frustrating when it comes to code, if you encounter any issues after installing the pre-requisites with running an example from **IMPaCT**, it is likely the makefile is the issue. In general, a handy guide for using Makefiles can be found [here](https://opensource.com/article/18/8/what-how-makefile). Simply, the Makefile tries to compile the code to create an executable file that you can then run the get the synthesis or verification results for your system. The Makefile combines all the various different compilers, libraries and files together so that you do not have to manually write the compilation instructions yourself, e.g.:

```
acpp  robot2D.cpp ../../src/IMDP.cpp ../../src/MDP.cpp --acpp-targets="omp" -O3 -lnlopt -lm -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lglpk -lgsl -lgslcblas -DH5_USE_110_API -larmadillo -o robot2D
```

The Makefile also includes the instructions for when the user runs `make clean` to remove the previous executable files that were created.

- `acpp` is the compiler that comes from [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) using this compiler also requires the `--acpp-targets="omp"` which selects OpenMP as the backend for parallelism. Note instead of these commands we have experienced it also working with `syclcc` as the compiler without the targets command. If you are having issues, try both.
- `robot2D.cpp` or equivalent is the source file to be compiled, this should be your configuration file.
- `../../src/IMDP.cpp` and `../../src/MDP.cpp` are the other sources files required to build the program and contain all the functions that are used and the class descriptions. Make sure these accurately point to the `src` folder from the location the configuration file is. The default location assumes the command `make` is called from inside one of the examples inside the `examples` folder.
- `-O3` is a flag that indicates the highest level of optimization
- `-l` signifiers a linker flag. So, `-lnlopt -lm -lhdf5 -lglpk -lgsl -lgslcblas -larmadillo` link to `NLopt`, `math`, `HDF5`, `GLPK`, `GSL`, `gslcblas`, and `Armadillo` libraries respectively.
- `-I` specifies an include directory and `-L` signifies a library directory, so `-I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial` specifies explicitly to include and link to the HDF5 library at this location. You can change these as necessary (e.g. `-lhdf5` may find the correct directories immediately, or other libraries may not be found and require including and linking via `-I` and `-L`.
- `-D` is a preprocessor definition so `-DH5_USE_110_API` indicates to use `HDF5`'s 1.10 API.
- Finally, `-o` indicates the name of the output file after compiling. The Makefile will automatically match the configuration file name to the executable file name, so for configuration file `robot2D.cpp` the executable via `-o robot2D` is called `robot2D`.

To run the make file simply run `make` into the command line of the terminal pointing to the folder the configuration file is in, then type `./name` where name is the appropriate name for the executable.

We hope this file has provided comprehensive details for the setup and running of our tool, please contact us with any issues or corrections based on your experiences of using **IMPaCT**.
