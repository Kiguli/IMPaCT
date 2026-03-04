===========
Quick Start
===========

This guide walks through a complete example: synthesizing a controller for a 2D robot
to reach a target region. This corresponds to the ``ex_2Drobot-R-U`` example.

Prerequisites
=============

Ensure IMPaCT is installed (see :doc:`installation`) or use :doc:`Docker <docker>`.

Configuration File
==================

Each IMPaCT case study is defined in a C++ configuration file. Create a file
(e.g., ``robot2D.cpp``) with the following structure.

**1. Include headers and set dimensions:**

.. code-block:: cpp

   #include "../../src/IMDP.h"
   #include <armadillo>

   using namespace std;
   using namespace arma;

   const int dim_x = 2;  // state dimensions
   const int dim_u = 2;  // input dimensions
   const int dim_w = 0;  // disturbance dimensions (0 = none)

**2. Define the spaces:**

Each space is a hyper-rectangle defined by lower bounds, upper bounds, and discretization step sizes:

.. code-block:: cpp

   // State space: [-10, 10] x [-10, 10] with step size 1
   const vec ss_lb = {-10, -10};
   const vec ss_ub = {10, 10};
   const vec ss_eta = {1, 1};

   // Input space
   const vec is_lb = {-1, -1};
   const vec is_ub = {1, 1};
   const vec is_eta = {0.2, 0.2};

**3. Define noise and target region:**

.. code-block:: cpp

   // Standard deviation of Gaussian noise per dimension
   const vec sigma = {sqrt(1/1.3333), sqrt(1/1.3333)};

   // Target region: square from (5,5) to (8,8)
   auto target_condition = [](const vec& ss) {
       return (ss[0] >= 5.0 && ss[0] <= 8.0) &&
              (ss[1] >= 5.0 && ss[1] <= 8.0);
   };

**4. Define dynamics:**

.. code-block:: cpp

   auto dynamics = [](const vec& x, const vec& u) -> vec {
       vec xx(dim_x);
       xx[0] = x[0] + 2*u[0]*cos(u[1]);
       xx[1] = x[1] + 2*u[0]*sin(u[1]);
       return xx;
   };

**5. Main function --- create, abstract, and synthesize:**

.. code-block:: cpp

   int main() {
       // Create IMDP object
       IMDP mdp(dim_x, dim_u, dim_w);

       // Set up spaces
       mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
       mdp.setInputSpace(is_lb, is_ub, is_eta);
       mdp.setTargetSpace(target_condition, true);

       // Set dynamics and noise
       mdp.setDynamics(dynamics);
       mdp.setNoise(NoiseType::NORMAL);
       mdp.setStdDev(sigma);

       // Compute abstraction
       mdp.targetTransitionVectorBounds();
       mdp.minAvoidTransitionVector();
       mdp.maxAvoidTransitionVector();
       mdp.transitionMatrixBounds();

       // Synthesize controller (true = pessimistic)
       mdp.infiniteHorizonReachController(true);

       // Save results
       mdp.saveController();
       return 0;
   }

Build and Run
=============

.. code-block:: bash

   cd examples/ex_2Drobot-R-U
   make
   ./robot2D

The tool will output progress for the abstraction and synthesis steps. Results are
saved as HDF5 files in the same directory.

.. tip::

   Suppress debug output with ``export ACPP_DEBUG_LEVEL=0`` before running.

Output Files
============

IMPaCT saves results in HDF5 format (``.h5`` files):

- ``ss.h5`` --- state space
- ``is.h5`` --- input space
- ``ts.h5`` --- target space
- ``minTM.h5``, ``maxTM.h5`` --- transition matrix bounds
- ``controller.h5`` --- synthesized controller

These can be loaded in MATLAB, Python, or R for visualization.
See the ``misc/`` folder for plotting utilities.

What's Next?
============

- :doc:`/user-guide/configuration` --- Full guide to defining systems
- :doc:`/user-guide/synthesis` --- Synthesis algorithms and options
- :doc:`/examples/index` --- All example case studies
- :doc:`/api/imdp` --- Complete IMDP API reference
