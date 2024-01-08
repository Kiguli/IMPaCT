# IMPaCT Software Tool

**IMPaCT**: **I**nterval **M**DP **Pa**rallel Construction for **C**ontroller Synthesis of Large-Scale S**T**ochastic Systems.

**IMPaCT** is a software tool for the parallelized verification and controller synthesis of large-scale stochastic systems using *interval Markov chains* (IMCs) and *interval Markov decision processes* (IMDPs), respectively. The tool serves to (i) construct IMCs/IMDPs as finite abstractions of underlying original systems, and (ii) leverage *interval iteration algorithms* for formal verification and controller synthesis over infinite-horizon properties, including *safety*, *reachability*, and *reach-while-avoid*, while offering convergence guarantees.

**IMPaCT** is developed in C++ and designed using [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), an independent open-source implementation of SYCL, for adaptive parallelism over CPUs and GPUs of all hardware vendors, including Intel and NVIDIA. **IMPaCT** stands as the first software tool for the parallel construction of IMCs/IMDPs, empowered with the capability to leverage high-performance computing platforms and cloud computing services. Specifically, parallelism offered by **IMPaCT** effectively addresses the challenges arising from the state-explosion problem inherent in discretization-based techniques applied to large-scale stochastic systems. 

We benchmark **IMPaCT** on several physical case studies, adopted from the *ARCH tool competition* for stochastic models, including a *2-dimensional robot*, a *3-dimensional autonomous vehicle*, a *5-dimensional room temperature system*, and a *7-dimensional building automation system*. To show the scalability of our tool, we also employ **IMPaCT** for the formal analysis of a *14-dimensional* case study.

## Table of Contents
- [Related Paper](#related-paper)
- [Installation](#installation)
- [Examples](#examples)
- [Contributing and Reporting Bugs](#contributing-and-reporting-bugs)
- [License](#license)

## Related Paper
For more information about the underlying concepts or research, you can find the related arXiv paper [here](./IMPaCT-Paper_arXiv.pdf).

## Installation
For detailed installation instructions on Mac and Linux, please refer to the [Installation Guide](./installation.md) file.

The AdaptiveCPP community have managed to get their software working for Windows, but we have not attempted that ourselves. We would love for you to try and demonstrate **IMPaCT** also works on Windows and provide us with the installation instructions we can share with others to widen the reach and usability of **IMPaCT**!

## Examples

For detailed instructions on how to run and adjust the examples, or how to create your own examples, please refer to the [Setup Guide](./setup.md) file.

Here are some examples demonstrating the usage of the code:

### Example 1 - 2D Robot Reaching a Target
<p align="center">
<img src="./examples/ex_2Drobot-R-U/fig.png" alt="Example 1 - 2D Robot Reaching a Target" width="500"/>
</p>
  
A 2D robot controller is synthesized over an infinite-time horizon with the goal of reaching the target region in green, see [ex_2Drobot-R-U](./examples/ex_2Drobot-R-U/).

### Example 2 - 2D Robot Reaching a Target while Avoiding a Region

<p align="center">
<img src="./examples/ex_2Drobot-RA-U/fig.png" alt="Example 2 - 2D Robot Reaching a Target while Avoiding a Region" width="500"/>
</p>

A 2D robot controller is synthesized over an infinite-time horizon with the goal of reaching the target region in green, while avoiding the unsafe region marked in red, see [ex_2Drobot-RA-U](./examples/ex_2Drobot-RA-U/).

### Example 3 - 3D Autonomous Vehicle Reaching a Target while Avoiding a Region
<p align="center">
<img src="./examples/ex_3Dvehicle-RA/fig.png" alt="Example 3 - 3D Autonomous Vehicle Reaching a Target while Avoiding a Region" width="500"/>
</p>

A 3D autonomous vehicle controller is synthesized over an infinite-time horizon with the goal of reaching the target region in green, while avoiding the unsafe region marked in red, see [ex_3Dvehicle-RA](./examples/ex_3Dvehicle-RA/).

### Example 4 - 3D Room Temperature Model Remaining in a Safe Region
<p align="center">
<img src="./examples/ex_3Droom-S/fig.png" alt="Example 4 - 3D Room Temperature Model Remaining in a Safe Region" width="500"/>
</p>

A 3D room temperature model is synthesized over a finite-time horizon of 10 steps with the goal of remaining inside of the safe region, bounded by the red dashes, see [ex_3Droom-S](./examples/ex_3Droom-S/).

### Example 5 - 4D Building Automation System Remaining in a Safe Region
<p align="center">
<img src="./examples/ex_4DBAS-S/fig.png" alt="Example 5 - 4D Building Automation System Remaining in a Safe Region" width="500"/>
</p>

A 4D building automation system controller is synthesized over a finite-time horizon of 10 steps with the goal of remaining inside of the safe region, bounded by the red dashes, see [ex_4DBAS-S](./examples/ex_4DBAS-S/).

### Testing the Examples
Running case studies for IMDP/IMC synthesis/verification can be computationally very heavy. We have deliberately provided some smaller examples that can run on small personal computers as well as providing larger models that may require much more power machines.

Smaller Examples: [ex_2Drobot-R-U](./examples/ex_2Drobot-R-U/), [ex_2Drobot-R-D](./examples/ex_2Drobot-R-D/), [ex_4DBAS-S](./examples/ex_4DBAS-S/),

Larger Examples: [ex_3Dvehicle-RA](./examples/ex_3Dvehicle-RA/), [ex_3Droom-S](./examples/ex_3Droom-S/), [ex_5Droom-S](./examples/ex_5Droom-S/), [ex_7DBAS-S](./examples/ex_7DBAS-S/), [ex_14Dstochy-S](./examples/ex_14Dstochy-S/)

### Setup For New Examples
The most helpful and descriptive example is [ex_2Drobot-R-D](./examples/ex_2Drobot-R-D/) which has plenty of additional comments in the configuration file to aid the user to design their own configuration file. The other examples are also helpful, but omit some of the less common functions. Other specific examples for various types of case study include:

1. Multivariate noise distribution - [ex_multivariateNormalPDF](./examples/ex_multivariateNormalPDF)
2. Custom noise distribution - [ex_customPDF](./examples/ex_customPDF)
3. Load files and run synthesis - [ex_load_reach](./examples/ex_load_reach)
4. Managing absorbing states in the state space - [ex_load_safe](./examples/ex_load_safe)

## Contributing and Reporting Bugs
Contributions and Collaborations are welcome! Please contact me for more details.

Similarly, please get in contact if you wish to report any bugs, I will do my best to resolve these in a timely manner. When contacting me, please provide as much details as you can regarding the bug: e.g. what occurred? why you think it happened? and what you think the fix would be?

## License
This project is licensed under the [MIT License](./LICENSE) see the file for details.
