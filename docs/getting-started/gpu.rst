=========
GPU Setup
=========

IMPaCT supports GPU-accelerated synthesis via AdaptiveCpp's SYCL implementation.
GPU acceleration applies to the **sorted synthesis algorithms** --- abstraction is
computed on the CPU and loaded from file.

Prerequisites
=============

Install the base IMPaCT dependencies first (see :doc:`installation`), then add
GPU-specific libraries.

Identify your GPU:

.. code-block:: bash

   lspci | grep -i vga

LLVM
----

.. code-block:: bash

   wget https://apt.llvm.org/llvm.sh
   chmod +x llvm.sh
   sudo ./llvm.sh 16
   sudo apt install -y libclang-16-dev clang-tools-16 libomp-16-dev llvm-16-dev

CUDA (NVIDIA)
-------------

.. code-block:: bash

   sudo apt install nvidia-cuda-toolkit

Verify with ``nvcc --version``.

Rebuild AdaptiveCpp with GPU Support
=====================================

After installing GPU libraries, rebuild AdaptiveCpp with the appropriate flags:

.. code-block:: bash

   cd AdaptiveCpp-develop
   sudo cmake . -DCUDAToolkit_LIBRARY_ROOT=/usr/lib/cuda
   sudo make install

See the `AdaptiveCpp documentation <https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md>`_
for other GPU backends (Intel, AMD).

Makefile Changes
================

To use GPU synthesis, update your example's Makefile:

1. Change the target from ``--acpp-targets="omp"`` to your GPU target
   (e.g., ``cuda:sm_70`` for NVIDIA V100).

2. Replace ``IMDP.cpp`` with ``GPU_synthesis.cpp`` in the source files.

.. code-block:: makefile

   CC = acpp
   CFLAGS = --acpp-targets="cuda:sm_70" -O3 ...
   SRCS = example.cpp ../../src/GPU_synthesis.cpp ../../src/MDP.cpp

.. note::

   GPU synthesis requires pre-computed abstraction matrices loaded from file.
   Compute the abstraction on CPU first, save to HDF5, then load for GPU synthesis.
   See :doc:`/user-guide/io` for details.
