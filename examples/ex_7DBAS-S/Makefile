CC = acpp # or syclcc (then remove --acpp-targets="omp")
ARCH := $(shell uname -m)
CFLAGS = --acpp-targets="omp" -O3 -lnlopt -lm -I/usr/include/hdf5/serial -L/usr/lib/$(ARCH)-linux-gnu/hdf5/serial -lhdf5 -lglpk -lgsl -lgslcblas -DH5_USE_110_API -larmadillo

# Find all .cpp files in the current directory
CPP_FILES := $(wildcard *.cpp)

# Generate corresponding executable names
EXECUTABLES := $(patsubst %.cpp,%,$(CPP_FILES))

all: $(EXECUTABLES)

%: %.cpp ../../src/IMDP.cpp ../../src/MDP.cpp
	$(CC) $^ $(CFLAGS) -o $@

clean:
	rm -f $(EXECUTABLES)

