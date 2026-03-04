========
Examples
========

IMPaCT includes a comprehensive set of examples demonstrating different
specifications, system dimensions, and features. Each example contains a
configuration file (``.cpp``), a Makefile, and MATLAB plotting scripts.

.. tip::

   **Smaller examples** (2D robot, 4D BAS) run on personal computers.
   **Larger examples** (5D+) may require more powerful machines.

.. list-table::
   :header-rows: 1
   :widths: 30 10 15 15 30

   * - Example
     - Dim
     - Specification
     - Horizon
     - Directory
   * - 2D Robot (Undisturbed)
     - 2
     - Reachability
     - Infinite
     - ``ex_2Drobot-R-U``
   * - 2D Robot (Disturbed)
     - 2
     - Reachability
     - Infinite
     - ``ex_2Drobot-R-D``
   * - 2D Robot Reach-Avoid (Undisturbed)
     - 2
     - Reach-Avoid
     - Infinite
     - ``ex_2Drobot-RA-U``
   * - 2D Robot Reach-Avoid (Disturbed)
     - 2
     - Reach-Avoid
     - Infinite
     - ``ex_2Drobot-RA-D``
   * - 3D Autonomous Vehicle
     - 3
     - Reach-Avoid
     - Infinite
     - ``ex_3Dvehicle-RA``
   * - 3D Room Temperature
     - 3
     - Safety
     - Finite (10)
     - ``ex_3Droom-S``
   * - 4D Building Automation
     - 4
     - Safety
     - Finite (10)
     - ``ex_4DBAS-S``
   * - 5D Room Temperature
     - 5
     - Safety
     - Finite
     - ``ex_5Droom-S``
   * - 7D Building Automation
     - 7
     - Safety (Verification)
     - Infinite
     - ``ex_7DBAS-S``
   * - 14D Stochastic System
     - 14
     - Safety
     - Finite
     - ``ex_14Dstochy-S``
   * - GPU Synthesis
     - --
     - Various
     - Various
     - ``ex_GPU``
   * - Custom PDF
     - --
     - Various
     - Various
     - ``ex_customPDF``
   * - Multivariate Normal
     - --
     - Various
     - Various
     - ``ex_multivariateNormalPDF``
   * - Load & Synthesize
     - --
     - Reachability
     - Various
     - ``ex_load_reach``
   * - Absorbing States
     - --
     - Safety
     - Various
     - ``ex_load_safe``

.. toctree::
   :hidden:

   reachability
   reach-avoid
   safety
   advanced

By Specification
================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Reachability
      :link: reachability
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Synthesize controllers to reach a target region, with and without disturbance.

   .. grid-item-card:: Reach-Avoid
      :link: reach-avoid
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Reach a target while avoiding unsafe regions.

   .. grid-item-card:: Safety
      :link: safety
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Keep the system within a safe region over finite or infinite horizons.

   .. grid-item-card:: Advanced
      :link: advanced
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      GPU synthesis, custom distributions, loading pre-computed data.
