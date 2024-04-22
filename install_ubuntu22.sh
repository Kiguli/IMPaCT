#!/bin/bash

apt-get install unzip
apt-get install tar

# Unzip the other libraries and remove the zip files
unzip AdaptiveCpp-develop.zip
tar -xf armadillo-12.6.4.tar.xz
tar -xf gsl-2.7.1.tar.gz
tar -xf nlopt-2.7.1.tar.gz

# Install system dependencies
apt-get update
apt-get install -y \
    build-essential \
    cmake \
    xz-utils \
    zlib1g-dev \
    clang-tools \
    glpk-utils \
    libglpk-dev \
    glpk-doc \
    libblas-dev \
    liblapack-dev \
    libnlopt-dev \
    libopenblas-dev \
    libopenmpi-dev \
    libhdf5-dev \
    libopenblas-dev \
    libomp-dev \
    liblapack-dev \
    libarpack2-dev \
    libboost-all-dev \
    libclang-dev \
    libsuperlu-dev \
    python3 \
    python3-pip

# Install AdaptiveCpp
cd AdaptiveCpp-develop
mkdir build
cd build
cmake ..
make install
cd ../..

# Install Armadillo
cd armadillo-12.6.4
mkdir build
cd build
cmake ..
make install
cd ../..

# Install GSL
cd gsl-2.7.1
./configure
make
make install
cd ..

# Install NLopt
cd nlopt-2.7.1
mkdir build
cd build
cmake ..
make
make install
cd ../..

# Add support for HDF5 in Armadillo
find /usr -type f -wholename '*/armadillo_bits/config.hpp' -exec sed -i 's/\s*\/\/\s*#define ARMA_USE_HDF5/#define ARMA_USE_HDF5/g' {} \;

