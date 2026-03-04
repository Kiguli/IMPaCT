:html_theme.sidebar_secondary.remove:

.. meta::
   :description: IMPaCT — Parallel IMDP construction and controller synthesis for large-scale stochastic systems
   :keywords: IMDP, interval MDP, controller synthesis, stochastic systems, GPU, SYCL

======
IMPaCT
======

**Interval MDP Parallel Construction for Controller Synthesis of Large-Scale Stochastic Systems**

IMPaCT is an open-source C++ tool for parallelized verification and controller synthesis
of large-scale stochastic systems using interval Markov decision processes (IMDPs).
Built on `AdaptiveCpp <https://github.com/AdaptiveCpp/AdaptiveCpp>`_ (SYCL) for
adaptive parallelism across CPUs and GPUs from all hardware vendors.

.. grid:: 2 2 4 4
   :gutter: 3

   .. grid-item-card:: :octicon:`cpu;1.5em` GPU Accelerated
      :class-card: sd-border-0 sd-shadow-sm

      SYCL-based parallelism via AdaptiveCpp.
      Runs on CPUs and GPUs from Intel, NVIDIA, and others.

   .. grid-item-card:: :octicon:`verified;1.5em` Formal Guarantees
      :class-card: sd-border-0 sd-shadow-sm

      Interval iteration algorithms with convergence guarantees
      for infinite-horizon specifications.

   .. grid-item-card:: :octicon:`shield-check;1.5em` Flexible Specifications
      :class-card: sd-border-0 sd-shadow-sm

      Safety, reachability, and reach-while-avoid properties
      over finite and infinite time horizons.

   .. grid-item-card:: :octicon:`graph;1.5em` Scalable
      :class-card: sd-border-0 sd-shadow-sm

      Tested from 2D to 14D systems.
      Parallel construction tackles the state-explosion problem.

----

Getting Started
===============

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: getting-started/installation
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Install IMPaCT from source on Linux or macOS, or use Docker.

   .. grid-item-card:: Quick Start
      :link: getting-started/quickstart
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Walk through a complete example: define a system, abstract, and synthesize a controller.

   .. grid-item-card:: Docker
      :link: getting-started/docker
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      The easiest way to get started --- run IMPaCT in a container.

   .. grid-item-card:: GPU Setup
      :link: getting-started/gpu
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Configure GPU acceleration for NVIDIA, Intel, and other hardware.

----

Example Case Studies
====================

.. list-table::
   :header-rows: 1
   :widths: 35 15 20 30

   * - Case Study
     - Dimensions
     - Specification
     - Description
   * - 2D Robot
     - 2
     - Reachability
     - Mobile robot reaching a target region
   * - 2D Robot (Disturbed)
     - 2
     - Reachability
     - Robot with additive disturbance
   * - 2D Robot Reach-Avoid
     - 2
     - Reach-Avoid
     - Reach target while avoiding obstacles
   * - 3D Autonomous Vehicle
     - 3
     - Reach-Avoid
     - Vehicle navigation with obstacle avoidance
   * - 3D Room Temperature
     - 3
     - Safety
     - Temperature regulation over finite horizon
   * - 4D Building Automation
     - 4
     - Safety
     - Multi-room temperature control
   * - 7D Building Automation
     - 7
     - Safety (Verification)
     - Large-scale IMC verification
   * - 14D Stochastic System
     - 14
     - Safety
     - High-dimensional scalability demonstration

:doc:`See all examples -> <examples/index>`

----

API Modules
===========

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: MDP
      :link: api/mdp
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Base class for state/input/disturbance spaces, dynamics, and noise distributions.

   .. grid-item-card:: IMDP
      :link: api/imdp
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Interval MDP class for abstraction, synthesis, and controller save/load.

   .. grid-item-card:: GPU Synthesis
      :link: api/gpu-synthesis
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Sorted synthesis algorithms optimized for GPU execution via SYCL.

   .. grid-item-card:: IO Utilities
      :link: api/io-utils
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      HDF5 save/load functions for matrices, vectors, and controllers.

.. toctree::
   :hidden:
   :caption: Getting Started

   getting-started/installation
   getting-started/quickstart
   getting-started/docker
   getting-started/gpu

.. toctree::
   :hidden:
   :caption: User Guide

   user-guide/configuration
   user-guide/synthesis
   user-guide/io
   user-guide/makefiles

.. toctree::
   :hidden:
   :caption: API Reference

   api/mdp
   api/imdp
   api/gpu-synthesis
   api/io-utils

.. toctree::
   :hidden:
   :caption: Examples

   examples/index

.. toctree::
   :hidden:

   citing
