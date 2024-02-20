CC = acpp # check AdaptiveCPP GitHub for correct GPU selection and put inside --acpp-targets="" (see: https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/using-hipsycl.md)
CFLAGS = --acpp-targets="cuda:sm_70;omp" -O3 -lnlopt -lm -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lglpk -lgsl -lgslcblas -DH5_USE_110_API -larmadillo

# Find all .cpp files in the current directory
CPP_FILES := $(wildcard *.cpp)

# Generate corresponding executable names
EXECUTABLES := $(patsubst %.cpp,%,$(CPP_FILES))

all: $(EXECUTABLES)

# see: below IMDP.cpp replaced with GPU_synthesis.cpp
%: %.cpp ../../src/GPU_synthesis.cpp ../../src/MDP.cpp
	$(CC) $^ $(CFLAGS) -o $@

clean:
	rm -f $(EXECUTABLES)

