=============
GPU Synthesis
=============

GPU-accelerated synthesis functions are implemented in ``src/GPU_synthesis.cpp``
using SYCL via AdaptiveCpp. These are the **sorted** variants of the standard
synthesis algorithms.

Overview
========

The sorted approach:

1. Sorts states by transition probability, enabling efficient parallel reduction
2. Removes the dependency on the GLPK linear programming library
3. Enables execution on GPUs (NVIDIA, Intel, etc.) via SYCL

All sorted synthesis functions are methods of the :cpp:class:`IMDP` class.
See :doc:`imdp` for the full function signatures.

Usage
=====

To use GPU synthesis:

1. **Compute abstraction on CPU** and save to HDF5 files
2. **Update the Makefile**: replace ``IMDP.cpp`` with ``GPU_synthesis.cpp``
   and change ``--acpp-targets`` to your GPU target
3. **Load abstraction** from files and call sorted synthesis functions

.. code-block:: cpp

   IMDP imdp(dim_x, dim_u, dim_w);

   // Load pre-computed data
   imdp.loadStateSpace("ss.h5");
   imdp.loadInputSpace("is.h5");
   imdp.loadMinTransitionMatrix("minTM.h5");
   imdp.loadMaxTransitionMatrix("maxTM.h5");
   // ... load other vectors ...

   // GPU-accelerated synthesis
   imdp.infiniteHorizonReachControllerSorted(true);
   imdp.saveController();

Makefile Configuration
======================

.. code-block:: makefile

   CC = acpp
   CFLAGS = --acpp-targets="cuda:sm_70" -O3 \
       -lnlopt -lm \
       -I/usr/include/hdf5/serial \
       -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
       -lhdf5 -lgsl -lgslcblas \
       -DH5_USE_110_API -larmadillo

   SRCS = example.cpp ../../src/GPU_synthesis.cpp ../../src/MDP.cpp

Note: ``-lglpk`` is not needed for sorted synthesis.

See :doc:`/getting-started/gpu` for full GPU setup instructions.
