# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**IMPaCT** (Interval MDP Parallel Construction for Controller Synthesis of Large-Scale Stochastic Systems) is a C++ tool for parallelized verification and controller synthesis of large-scale stochastic systems. It constructs:
- **IMCs** (Interval Markov Chains) for verification
- **IMDPs** (Interval Markov Decision Processes) for controller synthesis

The tool leverages SYCL/AdaptiveCpp for adaptive parallelism across CPUs and GPUs, enabling high-performance computing for systems that suffer from state-explosion problems.

## Build and Development Commands

### Compiling and Running Examples

```bash
# Navigate to an example directory
cd examples/ex_2Drobot-R-U/

# Compile the code
make

# Run the executable (name matches the .cpp filename)
./robot2D

# Clean build artifacts
make clean

# Suppress verbose debug output (optional)
export ACPP_DEBUG_LEVEL=0
```

### Compiler

The project uses `acpp` (AdaptiveCpp SYCL compiler) with OpenMP for CPU parallelization or GPU-specific targets.

**Standard Makefile pattern:**
```makefile
CC = acpp
CFLAGS = --acpp-targets="omp" -O3 -lnlopt -lm -lhdf5 -lglpk -lgsl -lgslcblas -larmadillo
%: %.cpp ../../src/IMDP.cpp ../../src/MDP.cpp
    $(CC) $^ $(CFLAGS) -o $@
```

**GPU variant** (replace `IMDP.cpp` with `GPU_synthesis.cpp`):
```makefile
%: %.cpp ../../src/GPU_synthesis.cpp ../../src/MDP.cpp
    $(CC) $^ $(CFLAGS) -o $@
```

## Code Architecture

### Class Hierarchy

```
MDP (base class: src/MDP.h, src/MDP.cpp)
  └── IMDP (derived class: src/IMDP.h, src/IMDP.cpp)
       └── GPU variant (src/GPU_synthesis.cpp)
```

### Configuration Pattern

Each example is a standalone `.cpp` file that:
1. Creates an `IMDP` object with dimensions
2. Defines state/input/disturbance spaces
3. Specifies target/avoid regions
4. Registers system dynamics and noise models
5. Computes abstraction (transition bounds)
6. Performs synthesis or verification

See [examples/ex_2Drobot-R-U/robot2D.cpp](examples/ex_2Drobot-R-U/robot2D.cpp) for a well-commented reference implementation.

### Three-Phase Workflow

**Phase 1: Space Definition**
```cpp
IMDP mdp(dim_x, dim_u, dim_w);  // x=state, u=input, w=disturbance
mdp.setStateSpace(lb, ub, eta);  // lower bound, upper bound, discretization step
mdp.setInputSpace(lb, ub, eta);
```

**Phase 2: Abstraction Construction**
```cpp
mdp.setDynamics([](const vec& x, const vec& u) -> vec { /* dynamics */ });
mdp.setNoise(NoiseType::NORMAL);
mdp.setStdDev(sigma);
mdp.transitionMatrixBounds();  // Compute min/max transition matrices
```

**Phase 3: Synthesis/Verification**
```cpp
mdp.infiniteHorizonReachController(true);  // true = pessimistic
mdp.saveController();  // Save to HDF5
```

### Automatic Mode Detection

The tool automatically detects the problem type based on input dimensions:
- **IMC (verification)**: `dim_u = 0` → Returns satisfaction probabilities
- **IMDP (synthesis)**: `dim_u > 0` → Returns controller lookup table with optimal actions

### Key Design Patterns

**Function Pointers for Dynamics**: User-defined system dynamics are passed as lambda functions and stored as `std::function` objects. Supports 1, 2, or 3 parameter variants:
```cpp
// 2-parameter: f(x, u)
auto dynamics = [](const vec& x, const vec& u) -> vec { return x + u; };
mdp.setDynamics(dynamics);

// 3-parameter: f(x, u, w) for disturbances
auto dynamics = [](const vec& x, const vec& u, const vec& w) -> vec { return x + u + w; };
mdp.setDynamics(dynamics);
```

**Specification via Lambdas**: Target and avoid regions defined using boolean functions:
```cpp
mdp.setTargetSpace([](const vec& ss) {
    return ss[0] >= 5 && ss[0] <= 8 && ss[1] >= 5 && ss[1] <= 8;
}, true);  // true = separate (remove) target states from base space
```

**Intermediate Results via HDF5**: Save/load abstraction matrices to avoid expensive recomputation:
```cpp
mdp.saveMinTransitionMatrix();
mdp.loadMinTransitionMatrix();
```

**"Separate" Operation**: When `setTargetSpace(condition, true)` or `setAvoidSpace(condition, true)` is called with `true`, those states are removed from the base state space before synthesis. This shapes the controller domain.

## Examples Organization

### Small Examples (Quick Testing on Personal Computers)
- [ex_2Drobot-R-U](examples/ex_2Drobot-R-U/) - 2D robot reachability (no disturbance)
- [ex_2Drobot-R-D](examples/ex_2Drobot-R-D/) - 2D robot reachability (with disturbance)
- [ex_4DBAS-S](examples/ex_4DBAS-S/) - 4D building automation (safety)

### Large Examples (Performance Testing)
- [ex_3Dvehicle-RA](examples/ex_3Dvehicle-RA/) - 3D autonomous vehicle reach-while-avoid
- [ex_3Droom-S](examples/ex_3Droom-S/) - 3D room temperature (safety)
- [ex_5Droom-S](examples/ex_5Droom-S/) - 5D room model
- [ex_7DBAS-S](examples/ex_7DBAS-S/) - 7D building automation (verification)
- [ex_14Dstochy-S](examples/ex_14Dstochy-S/) - 14D stochastic system

### Example Types by Problem Class

| Problem Type | Example | Notes |
|--------------|---------|-------|
| Verification (no inputs) | [ex_7DBAS-S](examples/ex_7DBAS-S/) | `dim_u = 0`, returns probabilities |
| Reachability (no disturbance) | [ex_2Drobot-R-U](examples/ex_2Drobot-R-U/) | `-U` suffix = undisturbed |
| Reachability (with disturbance) | [ex_2Drobot-R-D](examples/ex_2Drobot-R-D/) | `-D` suffix = disturbed, best commented example |
| Reach-while-avoid | [ex_2Drobot-RA-U](examples/ex_2Drobot-RA-U/), [ex_2Drobot-RA-D](examples/ex_2Drobot-RA-D/) | `-RA` suffix |
| Safety | [ex_4DBAS-S](examples/ex_4DBAS-S/) | `-S` suffix |
| Multivariate normal PDF | [ex_multivariateNormalPDF](examples/ex_multivariateNormalPDF/) | Full covariance matrices |
| Custom distributions | [ex_customPDF](examples/ex_customPDF/) | User-defined PDFs in `src/custom.cpp` |
| GPU acceleration | [ex_GPU](examples/ex_GPU/) | Links `GPU_synthesis.cpp` |
| Loading pre-computed data | [ex_load_reach](examples/ex_load_reach/), [ex_load_safe](examples/ex_load_safe/) | HDF5 import for data-driven methods |

## Configuration File Structure

A typical configuration file follows this pattern:

```cpp
#include "../../src/IMDP.h"

// 1. Define dimensions and parameters
const int dim_x = 2, dim_u = 2, dim_w = 0;
const vec ss_lb = {-10, -10}, ss_ub = {10, 10}, ss_eta = {1, 1};
const vec is_lb = {-1, -1}, is_ub = {1, 1}, is_eta = {0.2, 0.2};

// 2. Define target region
auto target = [](const vec& ss) { return ss[0] >= 5 && ss[0] <= 8; };

// 3. Define dynamics
auto dynamics = [](const vec& x, const vec& u) -> vec {
    return {x[0] + 2*u[0]*cos(u[1]), x[1] + 2*u[0]*sin(u[1])};
};

int main() {
    // 4. Create IMDP object
    IMDP mdp(dim_x, dim_u, dim_w);

    // 5. Set spaces
    mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
    mdp.setInputSpace(is_lb, is_ub, is_eta);
    mdp.setTargetSpace(target, true);  // true = separate target states

    // 6. Configure noise
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::NORMAL);
    mdp.setStdDev({0.866, 0.866});

    // 7. Compute abstraction
    mdp.transitionMatrixBounds();
    mdp.targetTransitionVectorBounds();

    // 8. Synthesize controller
    mdp.infiniteHorizonReachController(true);  // true = pessimistic (min probability)
    // OR for finite horizon:
    // mdp.finiteHorizonReachController(10, true);  // 10 time steps

    // 9. Save results
    mdp.saveController();
}
```

### Noise Distribution Options

**Normal (diagonal covariance - fastest)**:
```cpp
mdp.setNoise(NoiseType::NORMAL);
mdp.setStdDev({sigma_x, sigma_y});  // Standard deviations per dimension
```

**Multivariate normal (full covariance)**:
```cpp
mdp.setNoise(NoiseType::NORMAL, false, 10000);  // false = non-diagonal, 10k Monte Carlo samples
mdp.setInvCovDet(inv_cov_matrix, determinant);
```

**Custom distributions**:
```cpp
mdp.setNoise(NoiseType::CUSTOM, true, 10000);  // Monte Carlo integration
// Define PDF in src/custom.cpp
```

## Important Technical Notes

### No Formal Test Framework
- Examples serve as integration tests
- Run small examples (ex_2Drobot-R-U) to verify installation
- Each example is self-contained with its own Makefile

### Output Format
- Results saved to HDF5 format (Armadillo library integration)
- Read with MATLAB: `h5read('file.h5', '/dataset')`
- Read with Python: `import h5py; f = h5py.File('file.h5', 'r')`
- Read with R: `library(rhdf5); h5read('file.h5', '/dataset')`
- Post-processing scripts in [misc/](misc/) directory

### YouTube Tutorials
Comprehensive video guides available:
- [Installation on Ubuntu VM](https://www.youtube.com/watch?v=wwfP2ErgLcM)
- [Configuration files](https://www.youtube.com/watch?v=rsU6fZU_O4c)
- [Makefile setup](https://www.youtube.com/watch?v=6kzuQC_X9WQ)
- [Full playlist](https://www.youtube.com/playlist?list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL)

### Core Libraries
- **AdaptiveCpp**: SYCL implementation for CPU/GPU parallelism
- **Armadillo**: Linear algebra (with HDF5 support)
- **NLopt**: Nonlinear optimization for transition bound computation
- **GSL**: Monte Carlo integration for custom distributions
- **GLPK**: Linear programming solver

### Numerical Constants
- Epsilon for interval iteration convergence: `0.00001` (hard-coded in MDP class)
- Can be adjusted in [src/MDP.h](src/MDP.h) if needed

## Additional Resources

For comprehensive documentation, see:
- [README.md](README.md) - Project overview and examples
- [setup.md](setup.md) - Detailed configuration guide (600+ lines)
- [installation.md](installation.md) - Installation instructions for macOS, Linux, Windows
- [Docker_instructions.md](Docker_instructions.md) - Docker usage
- [Artifact-Evaluation-Instructions.pdf](Artifact-Evaluation-Instructions.pdf) - Reproducing paper results

## Citation

Wooding, B., & Lavaei, A. (2024). IMPaCT: Interval MDP Parallel Construction for Controller Synthesis of Large-Scale Stochastic Systems. In *International Conference on Quantitative Evaluation of Systems and Formal Modeling and Analysis of Timed Systems* (pp. 249-267). Springer.
