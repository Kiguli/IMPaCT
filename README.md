Note: some new features are being added at the moment, there may be some bugs caused by this, do see the release package on the right hand menu to use **IMPaCT** v1.0 which passed the repeatability evaluation of QEST+FORMATS. The tool according to **IMPaCT** v1.0 aligns with the 2024 QEST+FORMATS research paper.

# IMPaCT Software Tool

[![DOI](https://zenodo.org/badge/739478882.svg)](https://zenodo.org/doi/10.5281/zenodo.11085097) [![Create and publish a Docker image](https://github.com/Kiguli/IMPaCT/actions/workflows/docker.yml/badge.svg)](https://github.com/Kiguli/IMPaCT/actions/workflows/docker.yml)[![CC BY 4.0][cc-by-shield]][cc-by]

**IMPaCT**: <ins>**I**</ins>nterval <ins>**M**</ins>DP <ins>**Pa**</ins>rallel Construction for <ins>**C**</ins>ontroller Synthesis of Large-Scale S<ins>**T**</ins>ochastic Systems.

**IMPaCT** is an open-source software tool for the parallelized verification and controller synthesis of large-scale stochastic systems using *interval Markov chains* (IMCs) and *interval Markov decision processes* (IMDPs), respectively. The tool serves to (i) construct IMCs/IMDPs as finite abstractions of underlying original systems, and (ii) leverage *interval iteration algorithms* for formal verification and controller synthesis over infinite-horizon properties, including *safety*, *reachability*, and *reach-while-avoid*, while offering convergence guarantees.

**IMPaCT** is developed in C++ and designed using [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), an independent open-source implementation of SYCL, for adaptive parallelism over CPUs and GPUs of all hardware vendors, including Intel and NVIDIA. **IMPaCT** stands as the first software tool for the parallel construction of IMCs/IMDPs, empowered with the capability to leverage high-performance computing platforms and cloud computing services. Specifically, parallelism offered by **IMPaCT** effectively addresses the challenges arising from the state-explosion problem inherent in discretization-based techniques applied to large-scale stochastic systems.

Youtube tutorial videos including how to install **IMPaCT** on an Ubuntu Linux Virtual Machine can be found [here](https://www.youtube.com/playlist?list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL).

## Artifact Evaluation Repeatability Committee
 
If you are on the artifact evaluation repeatability committee, we have provided some instructions of how to reproduce the results of our paper in the document [Artifact-Evaluation-Instructions.pdf](./Artifact-Evaluation-Instructions.pdf).

## Table of Contents
- [Related Paper](#related-paper)
- [Installation](#installation)
- [Examples](#examples)
- [Reporting Bugs](#reporting-bugs)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Youtube Videos about IMPaCT](https://www.youtube.com/watch?v=wwfP2ErgLcM&list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL&index=1&ab_channel=Kiguli)

## Related Paper
The arXiv version of the paper is located [here](https://arxiv.org/abs/2401.03555).

### Authors
- [Ben Wooding](https://woodingben.com)
- [Abolfazl Lavaei](https://lavaei-cps.de/)

### Citing IMPaCT
```
@inproceedings{wooding2024impact,
      title={IMPaCT: Interval MDP Parallel Construction for Controller Synthesis of Large-Scale Stochastic Systems}, 
      author={Ben Wooding and Abolfazl Lavaei},
      booktitle={International Conference on Quantitative Evaluation of Systems and Formal Modeling and Analysis of Timed Systems},
      pages={249--267},
      year={2024},
      organization={Springer}
}

```

## Installation

The easiest way to install and run the tool would be using the image that can be downloaded from this repository by going to packages in the right-hand side menu. You can also build the docker image [Dockerfile](./Dockerfile) yourself and you can use the docker [notes](./Docker_instructions.md) to help (kindly put together for us by Ernesto Casablanca).

For your local machine (or a virtual machine) if it runs ubuntu 22.04, or equivalent, an installation script [install_ubuntu22.sh](./install_ubuntu22.sh) can be easily used by running these terminal commands in the respective folder:

`chmod +x install_ubuntu22.sh` to make the file executable followed by `sudo ./install_ubuntu22.sh` to install all the packages and dependencies.

You can see a video of how to install **IMPaCT** on an Ubuntu Linux Virtual Machine [here](https://www.youtube.com/watch?v=wwfP2ErgLcM&list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL&index=1&ab_channel=Kiguli).

You will need to install the parts for your specific GPU seperately, please see the AdaptiveCpp project for these installation details. 

For more detailed step by step manual installation instructions for Mac and Linux, please refer to the [Installation Guide](./installation.md). These can be a little bit finicky, particularly trying to install AdaptiveCpp on your machine.

We believe it should be possible to install the tool also for Windows, we have not tried this ourselves, do let us know if you have success!

## Examples

For detailed instructions on how to run and adjust the examples, or how to create your own examples, please refer to the [setup guide](./setup.md) file.

Here are some examples demonstrating the usage of the code:

### Example 1 - 2D Robot Reaching a Target
<p align="center">
<img src="./examples/ex_2Drobot-R-U/fig.png" alt="Example 1 - 2D Robot Reaching a Target" width="400"/>
</p>
  
A 2D robot controller is synthesized over an infinite-time horizon with the goal of reaching the target region in green, see [ex_2Drobot-R-U](./examples/ex_2Drobot-R-U/).

### Example 2 - 2D Robot Reaching a Target while Avoiding a Region

<p align="center">
<img src="./examples/ex_2Drobot-RA-U/fig.png" alt="Example 2 - 2D Robot Reaching a Target while Avoiding a Region" width="400"/>
</p>

A 2D robot controller is synthesized over an infinite-time horizon with the goal of reaching the target region in green, while avoiding the unsafe region marked in red, see [ex_2Drobot-RA-U](./examples/ex_2Drobot-RA-U/).

### Example 3 - 3D Autonomous Vehicle Reaching a Target while Avoiding a Region
<p align="center">
<img src="./examples/ex_3Dvehicle-RA/fig.png" alt="Example 3 - 3D Autonomous Vehicle Reaching a Target while Avoiding a Region" width="400"/>
</p>

A 3D autonomous vehicle controller is synthesized over an infinite-time horizon with the goal of reaching the target region in green, while avoiding the unsafe region marked in red, see [ex_3Dvehicle-RA](./examples/ex_3Dvehicle-RA/).

### Example 4 - 3D Room Temperature Remaining in a Safe Region
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
For IMC/IMDP construction for verification/synthesis, we have deliberately provided some smaller examples that can run on small personal computers as well as providing larger models that may require more power machines.

Smaller Examples: [ex_2Drobot-R-U](./examples/ex_2Drobot-R-U/), [ex_2Drobot-R-D](./examples/ex_2Drobot-R-D/), [ex_4DBAS-S](./examples/ex_4DBAS-S/),

Larger Examples: [ex_3Dvehicle-RA](./examples/ex_3Dvehicle-RA/), [ex_3Droom-S](./examples/ex_3Droom-S/), [ex_5Droom-S](./examples/ex_5Droom-S/), [ex_7DBAS-S](./examples/ex_7DBAS-S/), [ex_14Dstochy-S](./examples/ex_14Dstochy-S/)

### Configuration for New Examples
The most helpful and descriptive example is [ex_2Drobot-R-D](./examples/ex_2Drobot-R-D/) which has plenty of additional comments in the configuration file to aid the user to design their own configuration file, please all see the details in the [setup guide](./setup.md). The other examples are also helpful, but omit some of the unused functions. The majority of examples provide the code for both infinite-time horizon and finite-time horizon with one choice commented out. Below is a list of the type of case studies that the user may wish to use and an example demonstrating that type of case study:
- Verification: [ex_7DBAS-S](./examples/ex_7DBAS-S)
- Reachability *without* Disturbance: [ex_2Drobot-R-U](./examples/ex_2Drobot-R-U)
- Reachability *with* Disturbance: [ex_2Drobot-R-D](./examples/ex_2Drobot-R-D)
- Reach-while-Avoid *without* Disturbance: [ex_2Drobot-RA-U](./examples/ex_2Drobot-RA-U)
- Reach-while-Avoid *with* Disturbance: [ex_2Drobot-RA-U](./examples/ex_2Drobot-RA-D)
- Safety: [ex_4DBAS-S](./examples/ex_4DBAS-S)
- Multivariate normal distributions: [ex_multivariateNormalPDF](./examples/ex_multivariateNormalPDF)
- custom distributions: [ex_customPDF](./examples/ex_customPDF)
- Loading source files into **IMPaCT** for synthesis: [ex_load_reach](./examples/ex_load_reach)
- Dealing with Absorbing states: [ex_load_safe](./examples/ex_load_safe)
- Using the GPU [ex_GPU](./examples/ex_GPU)

A Youtube video explaining the configuration files can be found [here](https://www.youtube.com/watch?v=rsU6fZU_O4c&list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL&index=3&ab_channel=Kiguli).

A Youtube video explaining the Makefile setup can be found [here](https://www.youtube.com/watch?v=6kzuQC_X9WQ&list=PL50OJg3FHS4fBxhua92ZS3e6bxEnFaetL&index=2&ab_channel=Kiguli).

## Reporting Bugs
Please get in contact if you wish to report any bugs, we will do our best to resolve these in a timely manner. When contacting us, please provide as many details as you can regarding the bug: e.g. what occurred, why you think it happened, and what you think the fix would be?

## Acknowledgements

We want to take the time to thank some people who were generous in giving their time to the support of this tool.

- Sadegh Soudjani who assisted with some of the discussions in the early stages of this work.
- Ernesto Casablanca who kindly provided the Dockerfile that we provide for the tool.
- Omid Akbarzadeh, Ali Aminzadeh, Jamie Gardner, Milad Kazemi, Marco Lewis, Oliver Sch&ouml;n, and Mahdieh Zaker, who assisted with debugging the installation instructions.
- Max Planck Institute, and Tobias Kaufmann, for providing access and support with using the computing infrastructure used for simulations.

## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
