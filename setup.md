# IMPaCT Setup

In this file, I hope to provide all the necessary details required for running the case studies and also creating your own case studies to be used by **IMPaCT**. **IMPaCT** is designed to be as user friendly as possible, particularly for those who are maybe less familiar with programming. The trickiest part therefore is likely to be with the configuration of the Makefiles - I leave the discussions around this until the end.

In addition the functions are named primarily for the synthesis of interval Markov decision processes (IMDPs) but the functions are also useful for verification of interval Markov chains (IMCs). The tool will automatically detect when there is no input present within the system formulation and return to the user a lookup table with the verification satisfaction probabilities instead of the usual lookup table controller with both optimal actions and satisfaction probabilities. This can be seen in some of the examples such as the [7D BAS Verification Example](./examples/ex_7DBAS-S).

**IMPaCT** is also very flexible as it can enable the loading in of transition matrices that have been computed elsewhere, in order to be used for either infinite-time horizon or finite-time horizon synthesis within **IMPaCT**. This is powerful as it enables **IMPaCT** to be useful for data-driven methods.

## Table of Contents
- [Running an Example](#running-an-example)
- [Constructing IMDP/IMC](#constructing-imdpimc)
- [Noise Distributions](#noise-distributions)
- [State Space and Specification](#state-space-and-specification)
- [Dynamics](#dynamics)
- [Compute Abstraction](#compute-abstraction)
- [Synthesis](#synthesis)
- [Loading and Saving Files](#loading-and-saving-files)
- [Makefiles](#makefiles)

## Running an Example

In general one should be able to go into a folder of any of the examples where there will be some `.cpp` file e.g. `example.cpp` which contains the configuration file for the case study and also a make file which will compile and then run the code. To run the case study, open up a terminal or command line interface and run:

`make` which will compile the code followed by `./example` which will execute the new file. The name of the executable file will match the file `*.cpp` but without the `.cpp` part at the end.

When looking at the output, its possible there appear lots of debug warnings, these can be removed by running `export ACPP_DEBUG_LEVEL=0`.

## Constructing IMDP/IMC

## Noise Distributions

## State Space and Specification

## Dynamics

## Compute Abstraction

## Synthesis

## Loading and Saving Files

## Makefiles
