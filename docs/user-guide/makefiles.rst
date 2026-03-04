=========
Makefiles
=========

Each IMPaCT example uses a Makefile to compile the configuration file with all
dependencies. Understanding the Makefile structure helps when creating new examples
or troubleshooting build issues.

Structure
=========

A typical Makefile compiles a command equivalent to:

.. code-block:: bash

   acpp robot2D.cpp ../../src/IMDP.cpp ../../src/MDP.cpp \
       --acpp-targets="omp" -O3 \
       -lnlopt -lm \
       -I/usr/include/hdf5/serial \
       -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
       -lhdf5 -lglpk -lgsl -lgslcblas \
       -DH5_USE_110_API -larmadillo \
       -o robot2D

Key Components
==============

**Compiler**: ``acpp`` (AdaptiveCpp). Also works with ``syclcc`` in some cases.

**Targets**: ``--acpp-targets="omp"`` selects OpenMP as the parallelism backend.
For GPU, replace with your target (e.g., ``cuda:sm_70``).

**Source files**:

- Your configuration file (e.g., ``robot2D.cpp``)
- ``../../src/IMDP.cpp`` and ``../../src/MDP.cpp`` --- core library files
- For GPU: replace ``IMDP.cpp`` with ``GPU_synthesis.cpp``

**Linker flags** (``-l``):

- ``-lnlopt`` --- NLopt optimization
- ``-lm`` --- math library
- ``-lhdf5`` --- HDF5 data storage
- ``-lglpk`` --- GLPK linear programming (not needed for sorted synthesis)
- ``-lgsl -lgslcblas`` --- GNU Scientific Library
- ``-larmadillo`` --- Armadillo linear algebra

**Include/Library paths** (``-I``, ``-L``):

Explicit paths to HDF5 headers and libraries. Adjust these if HDF5 is installed
in a different location on your system.

**Preprocessor**: ``-DH5_USE_110_API`` selects the HDF5 1.10 API.

**Optimization**: ``-O3`` enables maximum compiler optimization.

Creating a New Example
======================

1. Create a new directory under ``examples/``.
2. Copy a Makefile from an existing example.
3. Update the source file name in the Makefile.
4. Ensure the relative paths to ``../../src/`` are correct.
5. Run ``make`` and then ``./your_executable``.

Troubleshooting
===============

- **Library not found**: add ``-I`` and ``-L`` flags pointing to the correct paths.
- **Compiler issues**: try ``syclcc`` instead of ``acpp``, or vice versa.
- **Debug output**: set ``export ACPP_DEBUG_LEVEL=0`` to suppress AdaptiveCpp warnings.
- See the `Makefile guide <https://opensource.com/article/18/8/what-how-makefile>`_
  for general Makefile help.

Videos
======

- `Configuration file walkthrough <https://www.youtube.com/watch?v=rsU6fZU_O4c&list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL&index=3>`_
- `Makefile setup walkthrough <https://www.youtube.com/watch?v=6kzuQC_X9WQ&list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL&index=2>`_
