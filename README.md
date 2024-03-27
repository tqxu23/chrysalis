# A Tale of Two Domains: Exploring Efficient Architecture Design for Truly Autonomous Things
Artifact Evaluation Repository for ISCA 24

## Installation

**Repository Clone.** Firstly, clone our artifact from the repository.

`$ git clone ......`

**Basic environment setup.** The following steps are required to install the software dependencies of the artifact. The linux operating system environment and python should be ready before the setup.

`$ cd chrysalis`

`$ pip install -r requirements.txt`

**Environment for accelerator search.** For experiment B in our paper, an external compilation for accelerator simulator is required. The simulation is based on MAESTRO while extend for intermittent scenarios.

`$ apt install libboost-all-dev scons g++`

`$ cd chrysalis/models/components/insitu/GammaCostCore/cost_model/maestro_source`

`$ scons`

After the compilation, copy the executable maestro file into the cost_model directory.

`$ cp ./maestro ../maestro`

## Experiment workflow

Multiple python scripts are ready for reproducing the data shown in the paper. Run the following scripts to achieve the search results.

### Optimizing Existing AuTs with CHRYSALIS

Run `$ python engineA.py engineA.yaml` in terminal to start a search process.

A fast search for existing AuTs can be executed by using the script `search/engineA.py` and parameters in `search/engineA.yaml`. The parameters can be customized including the capacitance search range, solar panel size search range, light environment, simulation step size, and architecture search number. 

### AI Accelerator-based AuT design with CHRYSALIS

Run `$ python engineB.py engineB.yaml` in terminal to start a search process.

A fast search for AI acceelerator-based AuTs can be executed by using the script `search/engineB.py` and paramerters in `search/engineB.yaml`. The parameters can be customized including the capacitance search range, solar panel size search range, accelerator PE number search range, accelerator memory search range, light environment, simulation step size, and architecture search number.