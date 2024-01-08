# IMPaCT Setup

In this file, I hope to provide all the necessary details required for running the case studies and also creating your own case studies to be used by **IMPaCT**. **IMPaCT** is designed to be as user friendly as possible, particularly for those who are maybe less familiar with programming. The trickiest part therefore is likely to be with the configuration of the Makefiles - I leave the discussions around this until the end.

## Table of Contents
- [Running an Example](#running-an-example)
- []
- [Makefiles](#makefiles)

## Running an Example

In general one should be able to go into a folder of any of the examples where there will be some `.cpp` file e.g. `example.cpp` which contains the configuration file for the case study and also a make file which will compile and then run the code. To run the case study, open up a terminal or command line interface and run:

`make` which will compile the code followed by `./example` which will execute the new file. The name of the executable file will match the file `*.cpp` but without the `.cpp` part at the end.

When looking at the output, its possible there appear lots of debug warnings, these can be removed by running `export ACPP_DEBUG_LEVEL=0`.

## 


## Makefiles
