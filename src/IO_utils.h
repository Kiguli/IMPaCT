#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <armadillo>
#include <string>
#include <iostream>

using namespace arma;
using namespace std;

/**
 * IO_utils.h
 *
 * Utility functions for saving and loading Armadillo matrices and vectors
 * to/from HDF5 format. This consolidates the repetitive save/load functions
 * that were duplicated across IMDP.cpp and GPU_synthesis.cpp.
 *
 * Part of Phase 1 refactoring to reduce code redundancy in IMPaCT v1.0
 */

namespace IMPaCT_IO {

    /**
     * Generic save function for Armadillo matrices/vectors
     *
     * @param data The Armadillo matrix or vector to save
     * @param default_filename Default filename to use if saving
     * @param data_name Human-readable name for error messages
     */
    template<typename T>
    void saveData(const T& data, const string& default_filename, const string& data_name) {
        if (data.empty()) {
            cout << data_name << " is empty, can't save file." << endl;
        } else {
            data.save(default_filename, hdf5_binary);
        }
    }

    /**
     * Generic load function for Armadillo matrices/vectors
     *
     * @param data The Armadillo matrix or vector to load into
     * @param filename The HDF5 file to load from
     * @param data_name Human-readable name for error messages
     */
    template<typename T>
    void loadData(T& data, const string& filename, const string& data_name) {
        bool ok = data.load(filename);
        if (!ok) {
            cout << "Issue loading " << data_name << "!" << endl;
        }
    }

} // namespace IMPaCT_IO

#endif // IO_UTILS_H
