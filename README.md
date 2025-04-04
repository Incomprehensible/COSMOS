# COSMOS: A Simulation Framework for Swarm-Based Orchestration in the Edge-Fog-Cloud Continuum

## Introduction

Here we present **COSMOS (Continuum Optimization for Swarm-based Multitier Orchestration System)**, a novel simulation framework for evaluating self-organizing scheduling algorithms in edge-fog-cloud ecosystems. Unlike conventional schedulers that rely on centralized optimization, COSMOS implements swarm intelligence principles inspired by decentralized biological systems, enabling emergent coordination across distributed nodes. 

The framework extends [Mesa](https://mesa.readthedocs.io/latest/)’s agent-based modeling toolkit to simulate:
* Multi-tier resource dynamics: Agent populations representing edge devices, fog nodes, and cloud servers with configurable behavioral policies
* Constraint-aware scheduling: Integration of answer set programming (ASP) for hard constraints and metaheuristic optimization for soft constraints
* Network-aware orchestration: Evaluation of communication costs and dependencies across continuum layers

Developed in [Lakeside Labs](https://www.lakeside-labs.com/) in scope of [Myrtus](https://myrtus-project.eu/) (Multi-layer 360 dynamic orchestration and interoperable design environment for compute-continuum systems) project.

<img src="https://myrtus-project.eu/wp-content/uploads/2024/03/orizzontali-01.png" alt="Myrtus Project Logo" width="50%">


## Installation

### Make sure the right Python version is installed
Make sure that `python` binary is of the correct version. For Windows users, Python 3.10 can be installed from Microsoft Store.
Linux users already know how to install Python on their system.
The simulation framework was tested with `Python 3.10` and `Mesa 3.0.3` (with corresponding dependencies). To check the version of your binary, execute the following command:
```bash
python --version
```

### Dependencies
The following Python modules are required for this project. Ensure that you have the specified versions installed:

- **mesa** (3.0.3)
- **matplotlib** (3.5.1)
- **numpy** (1.21.5)
- **networkx** (3.3)
- **protobuf** (3.12.4)
- **pyyaml** (6.0)

### Additional Dependencies (installed via mesa)
- **pandas** (2.0.3)
- **tqdm** (4.66.4)
- **python-dateutil** (2.8.2)

To install the required dependencies, run:

```bash
python3.10 -m pip install -r requirements.txt
```

Or:
```bash
python3.10 -m pip install mesa==3.0.3 matplotlib==3.5.1 numpy==1.21.5 networkx==3.3 protobuf==3.12.4 pyyaml==6.0
```

### On Windows
In case of issues with installing `pyyaml`, the following steps can be taken:
```bash
python3.10 -m pip install "cython<3.0.0"
# then run:
pip install --no-build-isolation pyyaml==6.0
```
Proceed further with the installation of the remaining dependencies from `requirements.txt`.

## Running the simulation
To run the simulation:
* adjust the configuration file named [`config.yml`](./modules/config/config.yml) in the [`config`](./modules/config) directory
* navigate to the [`modules`](./modules) directory:
```bash
cd modules
```
* execute the following command:
```bash
python3.10 EFC_main.py
```

## Directory Structure
The directory structure of the simulation framework is as follows:
* `modules`: contains the source code structured in modules.
    * `config`: contains the configuration file.
    * `generated/data` contains the generated dataset of the EFC network graphs/models.
    * `trained_mlp` contains the configuration files of the trained MLP models.
    * `interfaces` contains compiled protocol buffer files.
    * `evaluation` contains the script `eval.py` for evaluation of the trained MLP models.
* `proto`: contains the protocol buffer files describing the inter-process communication with the FREVO training framework. Is not needed for running the simulation.

## Configuration
The configuration file `config.yml` contains the following parameters:
* `num_pods`: number of pods in the network. Must be bigger or equal to sim_steps.
* `sim_steps`: number of simulation steps. The number of pods entering the network at each step is calculated as: `num_pods / sim_steps`
* `num_edge_devices`: number of edge devices in the network. The `num_` parameters are not considered when the graph is imported with `import_graph: True`.
* `num_fog_nodes`: number of fog nodes in the network.
* `num_cloud_servers`: number of cloud servers in the network.
* `behaviour_type`: type of decision-making behaviour ('random_walk', 'aco' or 'mlp'). The 'mlp' behaviour requires a trained MLP model to be imported if `mode: normal` is set.
* `import_graph`: whether to import a graph from the `generated/data` directory.
* `graph_sample`: name of the graph to import. The graph must be in the `generated/data` directory. For example, if the file is named `EFC1.graphml`, the value of `graph_sample` should be `EFC1`. The right format is `.graphml`.
* `nn_sample`: name of the MLP configuration file to import. The file must be in the `trained_mlp` directory. For example, if the file is named `mlp_config.yml`, the value of `nn_sample` should be `mlp_config.yml`.
* `plot_all`: whether to plot the simulation results. Is disabled during the training phase.
* `gen_samples_num`: number of samples to generate. Is only used when `mode: gen` is set.
* `mode`: mode of the simulation ('normal', 'gen' or 'train'). The 'train' mode trains an MLP model using the FREVO training framework. The 'gen' mode generates a dataset of EFC network graphs/models.
* `path_to_dataset`: path to the dataset. Is only used when `mode: gen` is set or when the graph is imported.
The following parameters are only used when `behaviour_type: mlp` and `mode: normal` are set:
* `frevo_port`: port number of the FREVO training framework to connect to.
* `candidate_evals`: number of inference requests to evaluate a candidate.
* `training_results_path`: path to save the best trained model.
* `evaluate_best_candidate`: whether to evaluate the best trained model at the end of training+exporting.
* `randomized_pod_interval`: whether to randomize the interval between the pods entering the network. If set to `False`, the interval is equal to `1` and number of issued pods is always predictable.
* `mu`: parameter `µ ∈ (0, 1]` defining the frequency of randomly arriving pods. The value is only used when `randomized_pod_interval: True` is set.

Typical configuration:
```yaml
num_pods: 500
sim_steps: 10
num_edge_devices: 35
num_fog_nodes: 10
num_cloud_servers: 5
behaviour_type: 'aco'
import_graph: True
graph_sample: EFC1
nn_sample: 'mlp_config.yml'
plot_all: True
gen_samples_num: 50
mode: 'normal'
path_to_dataset: 'generated/dataset'
frevo_port: 9990
candidate_evals: 10
training_results_path: "/home/user/Documents/Myrtus/Frevo/Frevo/Results/EFC_test"
evaluate_best_candidate: True
randomized_pod_interval: False
mu: 0.5
```

## Automatic evaluation
Saved trained MLP candidates can be evaluated automatically using the combination of `EFC_Evaluator.py` source file located in `modules` directory and `eval.py` script located in the `modules/evaluation` directory.
To generate the evaluation results, the following steps should be taken:
* adjust the configuration file named [`config.yml`](./modules/config/config.yml) in the [`config`](./modules/config) directory
* navigate to the [`modules`](./modules) directory:
```bash
cd modules
```
* run [`EFC_Evaluator.py`](./modules/EFC_Evaluator.py):
```bash
python3.10 EFC_Evaluator.py
```
The preliminary evaluation results for plotting will be saved in the `evaluation` directory as `eval.csv`.
* navigate to the [`modules/evaluation`](./modules/evaluation) directory:
```bash
cd evaluation
```
* run [`eval.py`](./modules/evaluation/eval.py) to visualize and plot the evaluation results:
```bash
python3.10 eval.py
```
