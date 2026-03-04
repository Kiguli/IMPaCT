=============================
IMDP (``class IMDP``)
=============================

.. cpp:class:: IMDP : public MDP

   Interval MDP class for abstraction, controller synthesis, and data management.
   Inherits all functionality from :cpp:class:`MDP`.

   Defined in ``src/IMDP.h`` with implementations in ``src/IMDP.cpp``.

   Constructor
   -----------

   Inherits the :cpp:class:`MDP` constructor:

   .. code-block:: cpp

      IMDP imdp(dim_x, dim_u, dim_w);

   Optimization Algorithm
   ----------------------

   .. cpp:function:: void setAlgorithm(nlopt::algorithm alg)

      Set the nonlinear optimization algorithm used during abstraction.

      :param alg: An NLopt algorithm (default: ``nlopt::LN_SBPLX``).

      See `NLopt Algorithms <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_
      for all options.

   Abstraction
   ===========

   Transition Matrix
   -----------------

   .. cpp:function:: void minTransitionMatrix()

      Compute the minimal transition probability matrix for state-to-state transitions.

   .. cpp:function:: void maxTransitionMatrix()

      Compute the maximal transition probability matrix for state-to-state transitions.

   .. cpp:function:: void transitionMatrixBounds()

      Compute both min and max transition matrices efficiently.
      Exploits sparsity: if max is zero, min is skipped. Use this instead of
      calling ``minTransitionMatrix()`` and ``maxTransitionMatrix()`` separately.

   Target Transition Vector
   ------------------------

   .. cpp:function:: void minTargetTransitionVector()

      Compute the minimal transition probability vector to the target region.

   .. cpp:function:: void maxTargetTransitionVector()

      Compute the maximal transition probability vector to the target region.

   .. cpp:function:: void targetTransitionVectorBounds()

      Compute both min and max target vectors efficiently with sparsity exploitation.

   Avoid Transition Vector
   -----------------------

   .. cpp:function:: void minAvoidTransitionVector()

      Compute the minimal transition probability vector to outside the state space
      (including the avoid region).

   .. cpp:function:: void maxAvoidTransitionVector()

      Compute the maximal transition probability vector to outside the state space.

   .. note::

      Avoid vectors are always required, even when no avoid region is defined, as they
      capture transitions out of the state space.

   Controller Synthesis
   ====================

   All synthesis functions take ``IMDP_lower``:

   - ``true`` --- pessimistic (worst-case noise first, then fix for upper bound)
   - ``false`` --- optimistic (best-case noise first, then fix for lower bound)

   Standard Synthesis
   ------------------

   .. cpp:function:: void infiniteHorizonReachController(bool IMDP_lower)

      Synthesize a reachability (or reach-avoid) controller over an infinite horizon
      using the interval iteration algorithm.

   .. cpp:function:: void infiniteHorizonSafeController(bool IMDP_lower)

      Synthesize a safety controller over an infinite horizon.

   .. cpp:function:: void finiteHorizonReachController(bool IMDP_lower, size_t timeHorizon)

      Synthesize a reachability (or reach-avoid) controller for a finite number of steps
      using value iteration.

   .. cpp:function:: void finiteHorizonSafeController(bool IMDP_lower, size_t timeHorizon)

      Synthesize a safety controller for a finite number of steps.

   Sorted Synthesis (GPU-Compatible)
   ---------------------------------

   Sorted variants eliminate the GLPK dependency and support GPU execution
   (implemented in ``src/GPU_synthesis.cpp``):

   .. cpp:function:: void infiniteHorizonReachControllerSorted(bool IMDP_lower)
   .. cpp:function:: void infiniteHorizonSafeControllerSorted(bool IMDP_lower)
   .. cpp:function:: void finiteHorizonReachControllerSorted(bool IMDP_lower, size_t timeHorizon)
   .. cpp:function:: void finiteHorizonSafeControllerSorted(bool IMDP_lower, size_t timeHorizon)

   .. cpp:function:: void finiteHorizonReachControllerSortedStoreMDP(bool IMDP_lower, size_t timeHorizon)

      Sorted finite-horizon reachability with intermediate MDP data storage.

   Save Functions
   ==============

   .. cpp:function:: void saveMinTransitionMatrix()
   .. cpp:function:: void saveMaxTransitionMatrix()
   .. cpp:function:: void saveMinTargetTransitionVector()
   .. cpp:function:: void saveMaxTargetTransitionVector()
   .. cpp:function:: void saveMinAvoidTransitionVector()
   .. cpp:function:: void saveMaxAvoidTransitionVector()
   .. cpp:function:: void saveController()

      Save the synthesized controller to ``controller.h5``.

   Load Functions
   ==============

   .. cpp:function:: void loadMinTransitionMatrix(string filename)
   .. cpp:function:: void loadMaxTransitionMatrix(string filename)
   .. cpp:function:: void loadMinTargetTransitionVector(string filename)
   .. cpp:function:: void loadMaxTargetTransitionVector(string filename)
   .. cpp:function:: void loadMinAvoidTransitionVector(string filename)
   .. cpp:function:: void loadMaxAvoidTransitionVector(string filename)
   .. cpp:function:: void loadController(string filename)

      Load a previously saved controller from HDF5.
