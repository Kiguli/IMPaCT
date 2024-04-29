FROM python:3

# Set the working directory
WORKDIR /app

# Copy the other libraries into the container
COPY ./AdaptiveCpp-develop.zip ./armadillo-12.6.4.tar.xz ./gsl-2.7.1.tar.gz ./nlopt-2.7.1.tar.gz ./

# Unzip the other libraries and remove the zip files
RUN unzip AdaptiveCpp-develop.zip && \
    tar -xf armadillo-12.6.4.tar.xz && \
    tar -xf gsl-2.7.1.tar.gz && \
    tar -xf nlopt-2.7.1.tar.gz && \
    rm AdaptiveCpp-develop.zip armadillo-12.6.4.tar.xz gsl-2.7.1.tar.gz nlopt-2.7.1.tar.gz

# Install the system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    xz-utils \
    zlib1g-dev \
    clang-tools-16 \
    glpk-utils \
    libglpk-dev \
    glpk-doc \
    libblas-dev \
    liblapack-dev \
    libnlopt-dev \
    libopenblas-dev \
    libopenmpi-dev \
    libhdf5-serial-dev \
    libopenblas-dev \
    libomp-16-dev \
    liblapack-dev \
    libarpack2-dev \
    libboost-all-dev \
    libclang-16-dev \
    libsuperlu-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a .git directory to avoid errors in cmake builds
RUN git init && \
    git config --global user.email "you@example.com"  && \
    git config --global user.name "Your Name" && \
    git add -A && \
    git commit -m "initial commit"

# Install AdaptiveCpp
RUN cd AdaptiveCpp-develop && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install && \
    cd ../.. && \
    rm -rf AdaptiveCpp-develop

# Install Armadillo
RUN cd armadillo-12.6.4 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install && \
    cd ../.. && \
    rm -rf armadillo-12.6.4

# Install GSL
RUN cd gsl-2.7.1 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf gsl-2.7.1

# Install NLopt
RUN cd nlopt-2.7.1 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf nlopt-2.7.1

# Add support for HDF5 in Armadillo
RUN find /usr -type f -wholename '*/armadillo_bits/config.hpp' -exec sed -i 's/\s*\/\/\s*#define ARMA_USE_HDF5/#define ARMA_USE_HDF5/g' {} \;

COPY src src
COPY examples examples
COPY sorted-examples sorted-examples

ENTRYPOINT [ "bash" ]
