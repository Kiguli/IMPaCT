# Installation

The following pre-requisite libraries and software are required in order to run **IMPaCT** properly:
- Python3
- CMake
- Boost
- AdaptiveCPP
- HDF5
- Armadillo (*including enabling HDF5 usage*)
- GSL
- NLopt
- GLPK

Most of these can be installed simply using either `sudo apt-get install` or `sudo brew install` for Linux or MacOS respectively. Some have been provided in zip files in this repository for ease of installation. Due to updates to specific different libraries and tools, please check online if any of the below installation points no longer work and we will try and update them quickly.

- [MacOS](#macOS)
- [Linux](#linux)
- [Windows](#windows)

## MacOS

Open your machines terminal, not the the terminal of an application e.g. PyCharm.

### Python3

Check if already installed:
`python3 --version`

Install:
`brew install python`

### CMake

Check if already installed:
`cmake --version`

Install:
`brew install cmake`

### Boost

Check if already installed:
`which boost`

Install:
`brew install boost`

### AdaptiveCPP

See the most up to date installation instructions [here](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md).

either clone from [AdaptiveCPP GitHub](https://github.com/AdaptiveCpp/AdaptiveCpp) or unzip the folder in this repository: 
`unzip AdaptiveCpp-develop.zip`

go into the directory:
`cd AdaptiveCpp-develop`

build the files
`sudo cmake .`

install the files
`sudo make install`

### HDF5

Check if already installed:
`h5dump --version`

Install:
`brew install hdf5`

### Armadillo

should be able to install everything with: `brew install armadillo`

**Important!** Enable HDF5 usage in Armadillo by going to the folder `armadillo_bits` where Armadillo is installed:
`cd usr/local/armadillo_bits`

open config file:
`sudo vi config.hpp`

uncomment the line `ARMA_USE_HDF5` by removing the `#` symbol, press `i` to edit the file!

save and exit the file: `Esc` then `:x` and enter

### GSL

Install:
`brew install gsl`

### NLopt

Install: 
`brew install nlopt`

### GLPK

Install:
`brew install glpk`

## Linux

Open your machines terminal, not the the terminal of an application e.g. PyCharm.

### Python3

Check if already installed:
`python3 --version`

Install (can replace star with version number, e.g. 8):
`sudo apt-get install python 3.*`

### CMake

Check if already installed:
`cmake --version`

Install:
`sudo apt-get install cmake`

### Boost

Check if already installed:
`dpkg -s libboost-dev | grep 'version'`

Install:
`sudo apt-get install libboost-all-dev`

### AdaptiveCPP

See the most up to date installation instructions [here](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md).

either clone from [AdaptiveCPP GitHub](https://github.com/AdaptiveCpp/AdaptiveCpp) or unzip the folder in this repository: 
`unzip AdaptiveCpp-develop.zip`

go into the directory:
`cd AdaptiveCpp-develop`

build the files
`sudo cmake .`

install the files
`sudo make install`

### HDF5

Check if already installed:
`dpkg -s libhdf5-dev`

Install:
`sudo apt-get install libhdf5-serial-dev`

### Armadillo

install pre-requisites:
`sudo apt install libopenblas-dev, liblapack-dev, libarpack2-dev, libsuperlu-dev`

either download armadillo [here](https://arma.sourceforge.net/download.html) or unzip the folder:
`tar -xvf armadillo-12.6.4.tar.xz`

go into the directory:
`cd armadillo-12.6.4`

build the files: 
`sudo cmake .`

install:
`sudo make install`

**Important!** Enable HDF5 usage in Armadillo by going to the folder `armadillo_bits` where Armadillo is installed:
`cd usr/local/armadillo_bits`

open config file:
`sudo vi config.hpp`

uncomment the line `ARMA_USE_HDF5` by removing the `#` symbol, press `i` to edit the file!

save and exit the file: `Esc` then `:x` and enter

### GSL

either download GSL [here](https://www.gnu.org/software/gsl/) or unzip the file:
`tar -xzf gsl-latest.tar.gz`

configure the files: `./configure`

build the files: `make`

install the files: `sudo make install`

### NLopt

either download NLopt [here](https://nlopt.readthedocs.io/en/latest/) or unzip the file:
`tar -xzf nlopt-2.7.1.tar.gz`

create a folder for the build: `mkdir build`

go into the folder: `cd build`

build the files: `cmake ..`

make the files: `make`

install the files: `sudo make install`

### GLPK

Check if already installed:
`dpkg -l | grep glpk`

Install:
`sudo apt-get install glpk-utils libglpk-dev glpk-doc`

## Windows

The AdaptiveCPP community have managed to get their software working for Windows, but we have not attempted it ourselves. Feel free to try **IMPaCT** on Windows and share with us your experience!
