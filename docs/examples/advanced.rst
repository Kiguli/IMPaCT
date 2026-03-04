========
Advanced
========

These examples demonstrate specialized features of IMPaCT.

GPU Synthesis (``ex_GPU``)
==========================

Demonstrates GPU-accelerated sorted synthesis. The abstraction is pre-computed
on CPU and loaded from HDF5 files, then synthesis runs on the GPU.

See :doc:`/getting-started/gpu` for setup instructions.

**Run:**

.. code-block:: bash

   cd examples/ex_GPU
   make && ./gpu_example

Custom Distributions (``ex_customPDF``)
========================================

Uses a user-defined probability distribution instead of the built-in Gaussian.
The custom PDF is defined in ``src/custom.cpp`` and integrated via Monte Carlo.

**Key configuration:**

.. code-block:: cpp

   imdp.setNoise(NoiseType::CUSTOM, false, 1000);

See :doc:`/user-guide/configuration` for details on custom distributions.

**Run:**

.. code-block:: bash

   cd examples/ex_customPDF
   make && ./customPDF

Multivariate Normal (``ex_multivariateNormalPDF``)
===================================================

Demonstrates a Gaussian distribution with a **full covariance matrix**
(non-diagonal, with off-diagonal correlations). Uses Monte Carlo integration.

**Key configuration:**

.. code-block:: cpp

   imdp.setNoise(NoiseType::NORMAL, false, 1000);
   imdp.setInvCovDet(inv_cov, det);

**Run:**

.. code-block:: bash

   cd examples/ex_multivariateNormalPDF
   make && ./multivariateNormal

Loading Pre-Computed Data (``ex_load_reach``)
==============================================

Shows how to load transition matrices computed externally (e.g., from data-driven
methods or other tools) and use IMPaCT for synthesis only.

This makes IMPaCT useful beyond model-based approaches --- any method that produces
transition probability bounds can feed into IMPaCT's synthesis algorithms.

**Run:**

.. code-block:: bash

   cd examples/ex_load_reach
   make && ./load_reach

Absorbing States (``ex_load_safe``)
====================================

Demonstrates handling absorbing states when loading pre-computed matrices for
safety specifications.

**Run:**

.. code-block:: bash

   cd examples/ex_load_safe
   make && ./load_safe

See :doc:`/user-guide/io` for details on the save/load API and debugging tips.
