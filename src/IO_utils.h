#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <armadillo>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>

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

    /**
     * Sort values and return sorted indices
     *
     * Creates a vector of indices that would sort the input values.
     * Used by GPU_synthesis.cpp sorted synthesis functions.
     *
     * @param values The Armadillo vector of values to sort
     * @param ascending True for ascending order, false for descending
     * @return Vector of indices in sorted order
     */
    inline std::vector<int> getSortedIndices(const vec& values, bool ascending = true) {
        std::vector<double> vals = conv_to<std::vector<double>>::from(values);
        std::vector<std::pair<int, double>> indexed;
        indexed.reserve(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            indexed.emplace_back(static_cast<int>(i), vals[i]);
        }
        if (ascending) {
            std::sort(indexed.begin(), indexed.end(),
                [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                });
        } else {
            std::sort(indexed.begin(), indexed.end(),
                [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second > b.second;
                });
        }
        std::vector<int> indices;
        indices.reserve(indexed.size());
        for (const auto& p : indexed) {
            indices.push_back(p.first);
        }
        return indices;
    }

} // namespace IMPaCT_IO

#endif // IO_UTILS_H
