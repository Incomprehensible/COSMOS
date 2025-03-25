# COSMOS: A Simulation Framework for Swarm-Based Orchestration in the Edge-Fog-Cloud Continuum

Here we present **COSMOS (Continuum Optimization for Swarm-based Multitier Orchestration System)**, a novel simulation framework for evaluating self-organizing scheduling algorithms in edge-fog-cloud ecosystems. Unlike conventional schedulers that rely on centralized optimization, COSMOS implements swarm intelligence principles inspired by decentralized biological systems, enabling emergent coordination across distributed nodes. 

The framework extends [Mesa](https://mesa.readthedocs.io/latest/)’s agent-based modeling toolkit to simulate:
* Multi-tier resource dynamics: Agent populations representing edge devices, fog nodes, and cloud servers with configurable behavioral policies
* Constraint-aware scheduling: Integration of answer set programming (ASP) for hard constraints and metaheuristic optimization for soft constraints
* Network-aware orchestration: Evaluation of communication costs and dependencies across continuum layers

Developed in [Lakeside Labs](https://www.lakeside-labs.com/) in scope of [Myrtus](https://myrtus-project.eu/) (Multi-layer 360 dynamic orchestration and interoperable design environment for compute-continuum systems) project.
