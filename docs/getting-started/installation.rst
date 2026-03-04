============
Installation
============

IMPaCT requires the following dependencies:

- Python 3, CMake, Boost
- `AdaptiveCpp <https://github.com/AdaptiveCpp/AdaptiveCpp>`_ (with clang++ and OpenMP)
- HDF5, `Armadillo <https://arma.sourceforge.net>`_ (with HDF5 enabled)
- GSL, NLopt, GLPK

.. tip::

   The easiest way to use IMPaCT is via :doc:`Docker <docker>`. For Ubuntu 22.04, use the
   automated install script.

.. tab-set::

   .. tab-item:: Ubuntu 22 (Recommended)

      Run the automated installer:

      .. code-block:: bash

         chmod +x install_ubuntu22.sh
         sudo ./install_ubuntu22.sh

      This installs all dependencies and builds IMPaCT. See the
      `installation video <https://www.youtube.com/watch?v=wwfP2ErgLcM&list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL>`_
      for a walkthrough.

   .. tab-item:: Linux (Manual)

      Install essential build tools:

      .. code-block:: bash

         sudo apt-get install build-essential

      **Python 3 and CMake:**

      .. code-block:: bash

         sudo apt-get install python3 cmake

      **Boost:**

      .. code-block:: bash

         sudo apt-get install libboost-all-dev

      **AdaptiveCpp:**

      See the `official instructions <https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md>`_
      for the most up-to-date guide. A copy is also bundled in the repository:

      .. code-block:: bash

         unzip AdaptiveCpp-develop.zip
         cd AdaptiveCpp-develop
         sudo apt install -y libclang-16-dev clang-tools-16 libomp-16-dev
         sudo cmake .
         sudo make install

      If issues occur, add flags such as ``-DCMAKE_CXX_COMPILER=/path/to/clang++-16``.

      **HDF5:**

      .. code-block:: bash

         sudo apt-get install libhdf5-serial-dev

      **Armadillo:**

      .. code-block:: bash

         sudo apt install libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
         tar -xvf armadillo-12.6.4.tar.xz
         cd armadillo-12.6.4
         sudo cmake .
         sudo make install

      .. important::

         Enable HDF5 in Armadillo: find the ``armadillo_bits`` directory
         (``sudo find / -type d -name 'armadillo_bits'``), edit ``config.hpp``,
         and uncomment the line ``ARMA_USE_HDF5``.

      **GSL:**

      .. code-block:: bash

         tar -xzf gsl-2.7.1.tar.gz
         cd gsl-2.7.1
         ./configure && make && sudo make install

      **NLopt:**

      .. code-block:: bash

         tar -xzf nlopt-2.7.1.tar.gz
         cd nlopt-2.7.1 && mkdir build && cd build
         cmake .. && make && sudo make install

      **GLPK:**

      .. code-block:: bash

         sudo apt-get install glpk-utils libglpk-dev

   .. tab-item:: macOS

      .. note::

         IMPaCT has been tested on macOS with Intel chips. For Apple Silicon (M1/M2+),
         check `AdaptiveCpp support <https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md>`_.

      Install dependencies via Homebrew:

      .. code-block:: bash

         brew install python cmake boost hdf5 gsl nlopt glpk

      **Armadillo:**

      .. code-block:: bash

         brew install armadillo

      Enable HDF5 in Armadillo: find ``armadillo_bits``
      (``sudo find / -type d -name 'armadillo_bits'``), edit ``config.hpp``,
      and uncomment ``ARMA_USE_HDF5``.

      **AdaptiveCpp:**

      .. code-block:: bash

         unzip AdaptiveCpp-develop.zip
         cd AdaptiveCpp-develop
         sudo cmake . -DCMAKE_INSTALL_PREFIX=/usr/local \
                       -DCMAKE_CXX_COMPILER=/path/to/clang++ \
                       -DOpenMP_ROOT=/path/to/libomp/include
         sudo make install

   .. tab-item:: Docker

      See :doc:`docker` for container-based installation --- no local dependencies needed.

Verification
------------

After installation, verify by building and running an example:

.. code-block:: bash

   cd examples/ex_2Drobot-R-U
   make
   ./robot2D

You should see output showing the abstraction and synthesis progress.
