======
Docker
======

The easiest way to use IMPaCT is via Docker --- no local dependency installation required.

Building the Image
==================

.. code-block:: bash

   docker build -t impact .

The first build may take a while as it compiles all dependencies.

Running the Container
=====================

.. code-block:: bash

   docker run --rm -it --name impact impact

This opens a shell inside the container. Run examples with:

.. code-block:: bash

   cd /app/examples/ex_2Drobot-R-U
   make
   ./robot2D

Mounting a Volume
=================

To use your own examples or persist results, mount a local directory:

.. code-block:: bash

   docker run --rm -it --name impact -v ~/my-examples:/app/examples impact

Files are shared between the container and host, so results appear in ``~/my-examples``.

.. warning::

   Files created inside the container (not in a mounted volume) are lost when the
   container stops.

Copying Results
===============

Alternatively, copy files out of a running container:

.. code-block:: bash

   docker cp impact:/app/examples/ex_2Drobot-R-U/controller.h5 .
