=============
Configuration
=============

Each IMPaCT case study is defined in a C++ configuration file (e.g., ``robot2D.cpp``).
This page explains every configurable component.

IMDP Object
============

Create an IMDP object specifying the dimensions of each space:

.. code-block:: cpp

   IMDP imdp(dim_x, dim_u, dim_w);

- ``dim_x`` --- state space dimensions
- ``dim_u`` --- input space dimensions (set to ``0`` for IMC verification)
- ``dim_w`` --- disturbance space dimensions (set to ``0`` if none)

When ``dim_u = 0``, IMPaCT automatically treats the problem as IMC verification
and returns satisfaction probabilities instead of a controller.

State, Input, and Disturbance Spaces
=====================================

Each space is a hyper-rectangle defined by three vectors of length equal to the
number of dimensions:

.. code-block:: cpp

   void setStateSpace(vec lb, vec ub, vec eta);
   void setInputSpace(vec lb, vec ub, vec eta);
   void setDisturbSpace(vec lb, vec ub, vec eta);

- ``lb`` --- lower bound of each dimension
- ``ub`` --- upper bound of each dimension
- ``eta`` --- discretization step size of each dimension

If input or disturbance spaces are not present, simply omit the call.

**Example:**

.. code-block:: cpp

   const vec ss_lb = {-10, -10};
   const vec ss_ub = {10, 10};
   const vec ss_eta = {1, 1};
   imdp.setStateSpace(ss_lb, ss_ub, ss_eta);

Target and Avoid Regions
========================

By default, all states are considered safe. For reachability or reach-avoid
specifications, define target and/or avoid regions using boolean functions:

.. code-block:: cpp

   void setTargetSpace(const function<bool(const vec&)>& condition, bool remove);
   void setAvoidSpace(const function<bool(const vec&)>& condition, bool remove);
   void setTargetAvoidSpace(
       const function<bool(const vec&)>& target_condition,
       const function<bool(const vec&)>& avoid_condition,
       bool remove);

Set ``remove = true`` to move matching states out of the state space (typical usage).
Set ``remove = false`` to keep them in the state space (useful for counting states).

**Example:**

.. code-block:: cpp

   // Target: square region from (5,5) to (8,8)
   auto target_condition = [](const vec& ss) {
       return (ss[0] >= 5.0 && ss[0] <= 8.0) &&
              (ss[1] >= 5.0 && ss[1] <= 8.0);
   };
   imdp.setTargetSpace(target_condition, true);

Dynamics
========

Define system dynamics as a lambda function. The number of parameters must match
the IMDP object dimensions:

.. code-block:: cpp

   // 3 parameters: state, input, disturbance
   void setDynamics(function<vec(const vec&, const vec&, const vec&)> d);

   // 2 parameters: state, input (no disturbance)
   void setDynamics(function<vec(const vec&, const vec&)> d);

   // 1 parameter: state only (autonomous system)
   void setDynamics(function<vec(const vec&)> d);

IMPaCT detects the number of parameters automatically.

**Example:**

.. code-block:: cpp

   auto dynamics = [](const vec& x, const vec& u) -> vec {
       vec xx(dim_x);
       xx[0] = x[0] + 2*u[0]*cos(u[1]);
       xx[1] = x[1] + 2*u[0]*sin(u[1]);
       return xx;
   };
   imdp.setDynamics(dynamics);

Noise Distributions
===================

IMPaCT supports three noise configurations:

**Normal distribution with diagonal covariance (default):**

.. code-block:: cpp

   imdp.setNoise(NoiseType::NORMAL);
   imdp.setStdDev(sigma);  // vec of standard deviations

Uses the closed-form CDF product, which is the fastest option.

**Multivariate normal with full covariance:**

.. code-block:: cpp

   imdp.setNoise(NoiseType::NORMAL, false, 1000);
   imdp.setInvCovDet(inv_cov, det);  // inverse covariance + determinant

Uses Monte Carlo integration with the specified number of samples.

**Custom distribution:**

Edit ``src/custom.cpp`` to define your PDF, then:

.. code-block:: cpp

   imdp.setNoise(NoiseType::CUSTOM, false, 1000);

The custom PDF receives a ``customParams`` struct with access to the current state,
dynamics, input, disturbance, integration bounds, and discretization parameter.

Optimization Algorithm
======================

The nonlinear optimization algorithm for abstraction can be changed:

.. code-block:: cpp

   imdp.setAlgorithm(nlopt::LN_SBPLX);  // default

See the `NLopt documentation <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_
for other algorithms (e.g., ``LN_COBYLA``).

Stopping Condition
==================

Set the convergence threshold for infinite-horizon synthesis:

.. code-block:: cpp

   imdp.setStoppingCondition(0.00001);  // default
