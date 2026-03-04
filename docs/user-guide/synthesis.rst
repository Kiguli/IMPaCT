==========================
Verification and Synthesis
==========================

After configuring the system and computing the abstraction, IMPaCT performs
formal verification or controller synthesis using interval iteration algorithms.

Abstraction
===========

The abstraction computes transition probability bounds for every state-to-state pair.
These are always required:

.. code-block:: cpp

   // Transition matrices (state-to-state)
   imdp.minTransitionMatrix();
   imdp.maxTransitionMatrix();

   // Avoid transition vectors (state-to-outside)
   imdp.minAvoidTransitionVector();
   imdp.maxAvoidTransitionVector();

For reachability or reach-avoid specifications, also compute target vectors:

.. code-block:: cpp

   imdp.minTargetTransitionVector();
   imdp.maxTargetTransitionVector();

Low-Cost Abstraction
--------------------

For sparse systems, a computational trick can speed up the abstraction: if the
maximal transition probability is zero, the minimal must also be zero. Use these
functions to exploit this:

.. code-block:: cpp

   imdp.transitionMatrixBounds();       // replaces separate min/max calls
   imdp.targetTransitionVectorBounds(); // replaces separate min/max calls

The sparser the system, the greater the speedup.

Controller Synthesis
====================

IMPaCT provides synthesis for two specification types across two time horizons.

All functions take a boolean ``IMDP_lower`` parameter:

- ``true`` --- **pessimistic** strategy (worst-case noise, lower bound probabilities)
- ``false`` --- **optimistic** strategy (best-case noise, upper bound probabilities)

Infinite-Horizon
-----------------

Uses the **interval iteration** algorithm with convergence guarantees:

.. code-block:: cpp

   imdp.infiniteHorizonReachController(true);   // reachability / reach-avoid
   imdp.infiniteHorizonSafeController(true);     // safety

Reach-avoid is detected automatically when an avoid region has been defined.

Finite-Horizon
--------------

Uses **value iteration** for a specified number of time steps:

.. code-block:: cpp

   imdp.finiteHorizonReachController(true, 10);  // 10 time steps
   imdp.finiteHorizonSafeController(true, 10);

Sorted Synthesis (GPU-Compatible)
=================================

Sorted variants remove the dependency on the GLPK library and enable GPU execution:

.. code-block:: cpp

   imdp.infiniteHorizonReachControllerSorted(true);
   imdp.infiniteHorizonSafeControllerSorted(true);
   imdp.finiteHorizonReachControllerSorted(true, 10);
   imdp.finiteHorizonSafeControllerSorted(true, 10);

These functions are implemented in ``src/GPU_synthesis.cpp``. To use them on a GPU,
update the Makefile (see :doc:`/getting-started/gpu`).

.. note::

   GPU synthesis requires pre-computed abstraction matrices loaded from HDF5 files.
   Compute the abstraction on CPU first, then load for GPU synthesis.
   See :doc:`io` for save/load details.

IMC Verification
================

When the input dimension is ``0`` (no control inputs), IMPaCT automatically
performs **verification** instead of synthesis. The same functions are used ---
the output is a lookup table of satisfaction probabilities rather than a controller.

Example:

.. code-block:: cpp

   IMDP imdp(7, 0, 0);  // 7D system, no inputs, no disturbance
   // ... set state space, dynamics, noise ...
   imdp.infiniteHorizonSafeController(true);  // returns probabilities
