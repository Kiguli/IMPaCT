===========================
MDP (``class MDP``)
===========================

.. cpp:class:: MDP

   Base class for Markov Decision Processes. Manages state, input, and disturbance
   spaces, system dynamics, noise distributions, and transition data.

   Defined in ``src/MDP.h``.

   Constructor
   -----------

   .. cpp:function:: MDP(const int x, const int u, const int w)

      Create an MDP with the specified dimensions.

      :param x: Number of state space dimensions.
      :param u: Number of input space dimensions (``0`` for no inputs).
      :param w: Number of disturbance space dimensions (``0`` for no disturbance).

   State Space
   -----------

   .. cpp:function:: void setStateSpace(vec lb, vec ub, vec eta)

      Define the state space as a discretized hyper-rectangle.

      :param lb: Lower bounds vector (length ``dim_x``).
      :param ub: Upper bounds vector (length ``dim_x``).
      :param eta: Discretization step size vector (length ``dim_x``).

   .. cpp:function:: mat getStateSpace()

      :returns: Matrix where each row is a discretized state.

   .. cpp:function:: void saveStateSpace()

      Save the state space to ``ss.h5`` in HDF5 format.

   .. cpp:function:: void loadStateSpace(string filename)

      Load a state space from an HDF5 file.

   Input Space
   -----------

   .. cpp:function:: void setInputSpace(vec lb, vec ub, vec eta)

      Define the input space as a discretized hyper-rectangle.

   .. cpp:function:: mat getInputSpace()

      :returns: Matrix where each row is a discretized input.

   .. cpp:function:: void saveInputSpace()

      Save the input space to ``is.h5``.

   .. cpp:function:: void loadInputSpace(string filename)

      Load an input space from an HDF5 file.

   Disturbance Space
   -----------------

   .. cpp:function:: void setDisturbSpace(vec lb, vec ub, vec eta)

      Define the disturbance space as a discretized hyper-rectangle.

   .. cpp:function:: mat getDisturbSpace()

      :returns: Matrix where each row is a discretized disturbance.

   .. cpp:function:: void saveDisturbSpace()

      Save the disturbance space to ``ws.h5``.

   .. cpp:function:: void loadDisturbSpace(string filename)

      Load a disturbance space from an HDF5 file.

   Target Space
   ------------

   .. cpp:function:: void setTargetSpace(const function<bool(const vec&)>& condition, bool remove)

      Separate states matching the condition into a target region.

      :param condition: Boolean function evaluated on each state.
      :param remove: If ``true``, remove matching states from the state space.

   .. cpp:function:: mat getTargetSpace()

      :returns: Matrix of target states.

   .. cpp:function:: void saveTargetSpace()

      Save the target space to ``ts.h5``.

   .. cpp:function:: void loadTargetSpace(string filename)

      Load a target space from an HDF5 file.

   Avoid Space
   -----------

   .. cpp:function:: void setAvoidSpace(const function<bool(const vec&)>& condition, bool remove)

      Separate states matching the condition into an avoid region.

      :param condition: Boolean function evaluated on each state.
      :param remove: If ``true``, remove matching states from the state space.

   .. cpp:function:: mat getAvoidSpace()

      :returns: Matrix of avoid states.

   .. cpp:function:: void saveAvoidSpace()

      Save the avoid space to ``as.h5``.

   .. cpp:function:: void loadAvoidSpace(string filename)

      Load an avoid space from an HDF5 file.

   Combined Target and Avoid
   -------------------------

   .. cpp:function:: void setTargetAvoidSpace(const function<bool(const vec&)>& target_condition, const function<bool(const vec&)>& avoid_condition, bool remove)

      Set both target and avoid regions simultaneously.

   Dynamics
   --------

   .. cpp:function:: void setDynamics(function<vec(const vec&, const vec&, const vec&)> d)

      Set dynamics with state, input, and disturbance parameters.

   .. cpp:function:: void setDynamics(function<vec(const vec&, const vec&)> d)

      Set dynamics with state and input parameters.

   .. cpp:function:: void setDynamics(function<vec(const vec&)> d)

      Set dynamics with state parameter only (autonomous system).

   Noise Configuration
   -------------------

   .. cpp:function:: void setNoise(NoiseType n, bool diagonal = true)

      Set the noise distribution type.

      :param n: ``NoiseType::NORMAL`` or ``NoiseType::CUSTOM``.
      :param diagonal: If ``true``, use diagonal covariance (default).

   .. cpp:function:: void setNoise(NoiseType n, bool diagonal, size_t monte_carlo_samples)

      Set noise with Monte Carlo integration.

      :param monte_carlo_samples: Number of samples for MC integration.

   .. cpp:function:: void setStdDev(vec sig)

      Set standard deviations for diagonal normal distribution.

   .. cpp:function:: void setInvCovDet(mat inv_cov, double det)

      Set the inverse covariance matrix and determinant for full covariance normal.

   .. cpp:function:: void setCustomDistribution(double (*c)(double*, size_t, void*), size_t monte_carlo_samples)

      Set a custom probability distribution function.

   .. cpp:function:: mat getInvCov()

      :returns: The inverse covariance matrix.

   .. cpp:function:: double getDet()

      :returns: The covariance matrix determinant.

   .. cpp:function:: vec getStdDev()

      :returns: The standard deviation vector.

   Transition Data
   ---------------

   .. cpp:function:: void saveTransitionMatrix()

      Save the transition matrix to HDF5.

   .. cpp:function:: void loadTransitionMatrix(string filename)

      Load a transition matrix from HDF5.

   .. cpp:function:: void saveTargetTransitionVector()

      Save the target transition vector.

   .. cpp:function:: void loadTargetTransitionVector(string filename)

      Load a target transition vector.

   .. cpp:function:: void saveAvoidTransitionVector()

      Save the avoid transition vector.

   .. cpp:function:: void loadAvoidTransitionVector(string filename)

      Load an avoid transition vector.

   .. cpp:function:: void trackMDP(bool store)

      Enable or disable intermediate MDP data storage.

   Stopping Condition
   ------------------

   .. cpp:function:: void setStoppingCondition(double eps)

      Set the convergence threshold for infinite-horizon synthesis.

      :param eps: Convergence epsilon (default: ``0.00001``).

Enums and Structs
-----------------

.. cpp:enum-class:: NoiseType

   .. cpp:enumerator:: NORMAL

      Gaussian noise (diagonal or full covariance).

   .. cpp:enumerator:: CUSTOM

      User-defined distribution via ``src/custom.cpp``.

.. cpp:struct:: customParams

   Parameters passed to custom PDF functions.

   .. cpp:member:: vec mean

      Result of applying dynamics to the current state.

   .. cpp:member:: vec state_start

      Current state being evaluated.

   .. cpp:member:: function<vec(const vec&)> dynamics1
   .. cpp:member:: function<vec(const vec&, const vec&)> dynamics2
   .. cpp:member:: function<vec(const vec&, const vec&, const vec&)> dynamics3

      System dynamics functions.

   .. cpp:member:: vec input

      Current input.

   .. cpp:member:: vec disturb

      Current disturbance.

   .. cpp:member:: vec lb
   .. cpp:member:: vec ub

      Integration bounds.

   .. cpp:member:: vec eta

      Discretization parameter.
