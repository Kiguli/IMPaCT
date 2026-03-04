================================
IO Utilities (``IMPaCT_IO``)
================================

Utility functions for saving and loading Armadillo matrices and vectors to/from
HDF5 format. Defined in ``src/IO_utils.h``.

.. cpp:namespace:: IMPaCT_IO

.. cpp:function:: template<typename T> void saveData(const T& data, const string& default_filename, const string& data_name)

   Save an Armadillo matrix or vector to an HDF5 file.

   :param data: The matrix or vector to save.
   :param default_filename: Output filename (e.g., ``"ss.h5"``).
   :param data_name: Human-readable name for error messages.

   If ``data`` is empty, prints a warning and does not create a file.

.. cpp:function:: template<typename T> void loadData(T& data, const string& filename, const string& data_name)

   Load an Armadillo matrix or vector from an HDF5 file.

   :param data: The matrix or vector to load into.
   :param filename: Path to the HDF5 file.
   :param data_name: Human-readable name for error messages.

.. cpp:function:: std::vector<int> getSortedIndices(const vec& values, bool ascending = true)

   Return indices that would sort the input values. Used internally by GPU
   sorted synthesis functions.

   :param values: Armadillo vector of values to sort.
   :param ascending: Sort order (default: ascending).
   :returns: Vector of indices in sorted order.
