# Installation

The following pre-requisite libraries and software are required in order to run **IMPaCT** properly:
- Python3
- CMake
- Boost
- AdaptiveCPP (*with clang++ and OpenMP for compiler and parallelization*)
- HDF5
- Armadillo (*including enabling HDF5 usage*)
- GSL
- NLopt
- GLPK
- *optional* GPU specific extras, e.g. llvm, cuda, etc.

Most of these can be installed simply using either `sudo apt-get install` or `sudo brew install` for Linux or MacOS respectively. Some have been provided in zip files in this repository for ease of installation. Due to updates to specific different libraries and tools, please check online if any of the below installation points no longer work and we will try and update them quickly.

**Inside this repository you can find a Dockerfile with IMPaCT pre-installed which is the easiest way to use the tool. Additionally we have included an install script which automatically installs IMPaCT and has been tested to work on Ubuntu 22.**

- [MacOS](#macOS)
- [Linux](#linux)
- [Windows](#windows)
- [GPU](#gpu)
  
## MacOS

I have personally got **IMPaCT** running on MacOS with an Intel chip, for the newer M1 and M2 chips you may need to see the AdaptiveCPP page [here](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md) to see the support.

Open your machines terminal, not the the terminal of an application e.g. PyCharm.

If you have never run or installed code on your machine before then its a good idea to install XCode and this will download many packages automatically.

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

either clone from [AdaptiveCPP GitHub](https://github.com/AdaptiveCpp/AdaptiveCpp) or unzip the folder directly from this repository [here](./AdaptiveCpp-develop.zip): 
`unzip AdaptiveCpp-develop.zip`

go into the directory:
`cd AdaptiveCpp-develop`

build the files
`sudo cmake .` (note this dot is important to signify the current folder)

If errors occur then adding flags will help with the compilation such as:

`sudo cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_CXX_COMPILER=/path/to/clang++ -DOpenMP_ROOT=/path/to/libomp/include`

You may need to find these files yourself to link appropriately.

install the files:
`sudo make install`

### HDF5

Check if already installed:
`h5dump --version`

Install:
`brew install hdf5`

### Armadillo

should be able to install everything with: `brew install armadillo`

**Important!** Enable HDF5 usage in Armadillo by going to the folder `armadillo_bits` where Armadillo is installed:

e.g. `cd usr/local/../armadillo_bits`

you can find the location of this file using (if multiple locations exist you can try all of them to be safe!):

`sudo find / -type d -name 'armadillo_bits'`

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

### Ubuntu (tested on Ubuntu 22)

For your local machine if it runs ubuntu (tested on ubuntu 22, but may work for some other linux systems too) an installation script [install_ubuntu22.sh](./install_ubuntu22.sh) can be easily used by running in the terminal the commands:

`chmod +x install_ubuntu22.sh` to make the file executable followed by `sudo ./install_ubuntu22.sh` to install all the packages and dependencies.

### Other Linux Machines

Open your machines terminal, not the the terminal of an application e.g. PyCharm.

If you have never run code on your PC before, it's a good idea to run `sudo apt-get install build-essential` which will download many essential packages for your PC.

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

See the most up to date installation instructions [here](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md), this installation can be tricky and so there much more info on the official page. The most common problem that is faced is with correctly linking AdaptiveCpp with the correct installed files and compilers.

either clone from [AdaptiveCPP GitHub](https://github.com/AdaptiveCpp/AdaptiveCpp) or unzip the folder directly from this repository [here](./AdaptiveCpp-develop.zip): 
`unzip AdaptiveCpp-develop.zip`

go into the directory:
`cd AdaptiveCpp-develop`

you may need to install clang and openMP:
`sudo apt install -y libclang-16-dev clang-tools-16 libomp-16-dev`

build the files:
`sudo cmake .` (note this dot is important to signify the current folder)

if issues occur add some flags such as `-DCMAKE_CXX_COMPILER=/path/to/clang++-16` can help with the correct compilation, check the official page for more details.

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

e.g. `cd usr/local/../armadillo_bits`

you can find the location of this file using (if multiple locations exist you can change all of them to be safe!):

`sudo find / -type d -name 'armadillo_bits'`

open config file:
`sudo vi config.hpp`

uncomment the line `ARMA_USE_HDF5` by removing the `#` symbol, press `i` to edit the file!

save and exit the file: `Esc` then `:x` and enter

### GSL

either download GSL [here](https://www.gnu.org/software/gsl/) or unzip the file:
`tar -xzf gsl-2.7.1.tar.gz`

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

## GPU

Check AdaptiveCPP instructions for different GPU cards and operating systems [here](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md). 

Below are some basic commands for GPU installation for Linux (on top of the previous installation instructions).

Find your GPU card using 

`lspci`

### To install LLVM

```
llvm-config --version

sudo apt-install clang

wget https://apt.llvm.org/llvm.sh

chmod +x llvm.sh

sudo ./llvm.sh 16

sudo apt install -y libclang-16-dev clang-tools-16 libomp-16-dev llvm-16-dev
```
### To install CUDA

```
nvcc --version

sudo apt install nvidia-cuda-toolkit
```
### Add Flags to AdaptiveCPP installation

go into directory

```
sudo cmake . -DCUDAToolkit_LIBRARY_ROOT=/usr/lib/cuda

sudo make install
```
