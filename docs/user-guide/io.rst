===================
Loading and Saving
===================

IMPaCT uses `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ for all data
persistence. This enables interoperability with MATLAB, Python (h5py), R (rhdf5),
and other tools. The HDF5 field name is ``dataset`` by default.

Saving Data
===========

Save any component after it has been computed:

.. code-block:: cpp

   // Spaces
   imdp.saveStateSpace();        // -> ss.h5
   imdp.saveInputSpace();        // -> is.h5
   imdp.saveDisturbSpace();      // -> ws.h5
   imdp.saveTargetSpace();       // -> ts.h5
   imdp.saveAvoidSpace();        // -> as.h5

   // Transition matrices and vectors
   imdp.saveMinTransitionMatrix();         // -> minTM.h5
   imdp.saveMaxTransitionMatrix();         // -> maxTM.h5
   imdp.saveMinTargetTransitionVector();   // -> minTV.h5
   imdp.saveMaxTargetTransitionVector();   // -> maxTV.h5
   imdp.saveMinAvoidTransitionVector();    // -> minAV.h5
   imdp.saveMaxAvoidTransitionVector();    // -> maxAV.h5

   // Controller
   imdp.saveController();        // -> controller.h5

Loading Data
============

Load pre-computed data from HDF5 files:

.. code-block:: cpp

   // Spaces
   imdp.loadStateSpace("ss.h5");
   imdp.loadInputSpace("is.h5");
   imdp.loadDisturbSpace("ws.h5");
   imdp.loadTargetSpace("ts.h5");
   imdp.loadAvoidSpace("as.h5");

   // Transition matrices and vectors
   imdp.loadMinTransitionMatrix("minTM.h5");
   imdp.loadMaxTransitionMatrix("maxTM.h5");
   imdp.loadMinTargetTransitionVector("minTV.h5");
   imdp.loadMaxTargetTransitionVector("maxTV.h5");
   imdp.loadMinAvoidTransitionVector("minAV.h5");
   imdp.loadMaxAvoidTransitionVector("maxAV.h5");

   // Controller
   imdp.loadController("controller.h5");

This is useful for:

- **GPU synthesis**: compute abstraction on CPU, load on GPU for synthesis
- **Data-driven methods**: load transition matrices computed externally
- **Resuming computation**: save intermediate results and continue later

.. tip::

   The ``misc/`` folder contains MATLAB, Python, and R utilities for reading
   HDF5 files and plotting results.

Debugging Tips
==============

.. warning::

   If synthesis diverges to values greater than 1, your loaded matrices may need
   to be **transposed**. Use tools like MATLAB's ``h5disp`` and ``h5read`` to
   verify dimensions before loading.

.. warning::

   If the controller shows both upper and lower bounds as zero, this indicates an
   **infeasible solution** --- typically because a minimal value exceeds a maximal
   value somewhere. Verify that max >= min for all entries.
