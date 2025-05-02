In the **virtual network embedding problem**, the goal is to map (embed) a set of virtual network instances to a given physical network substrate at minimal cost, while respecting the capacity constraints of the physical network. This NP-hard problem is fundamental to network virtualization, embodying essential properties of resource allocation problems faced by service providers in the edge-to-cloud spectrum. Due to its centrality, this problem and its variants have been extensively studied and remain in the focus of the research community. 


This repository contains the source code for these two publications:

### The Power of Alternatives in Network Embedding

In this work, we present a new variant, the **virtual network embedding with alternatives problem** (VNEAP). This new problem captures the power of a common network virtualization practice, in which virtual network topologies are malleable - embedding of a given virtual network instance can be performed using any of the alternatives from a given set of topology alternatives. We provide two efficient heuristics for VNEAP and show that having multiple virtual network alternatives for the same application is superior to the best results known for the classic formulation. 


```
Kolosov, Oleg, and Yadgar, Gala and Behravesh, Rasoul and Breitgand, David and Lorenz, Dean H. "The Power of Alternatives in Network Embedding." In IEEE INFOCOM 2025-IEEE International Conference on Computer Communications, IEEE, 2025.
```
### Plan-Based Scalable Online Virtual Network Embedding

We focus on the online variant of VNE, in which deployment requests are not known in advance. This reflects the highly skewed and unpredictable demand intrinsic to the edge. Unfortunately, existing solutions to online VNE do not scale well with the number of requests per second and the physical topology size.

We propose a novel approach in which our new online algorithm, OLIVE, leverages a nearly optimal embedding for an aggregated expected demand. This embedding is computed offline. It serves as a plan that OLIVE uses as a guide for handling actual individual requests while dynamically compensating for deviations from the plan. In the paper, we demonstrate that our solution can handle a number of requests per second greater by two orders of magnitude than the best results reported in the literature. Thus, it is particularly suitable for realistic edge environments.

```
Kolosov, Oleg and Breitgand, David and Lorenz, Dean H. and Yadgar, Gala. "Plan-Based Scalable Online Virtual Network Embedding." In 2025 IEEE 45th International Conference on Distributed Computing Systems (ICDCS), IEEE, 2025.
```

## Project Usage and Information

### Project Structure

- `Topologies/` - physical topologies
- `Results` - stores results under `Offline`/`Online` directories
- `src/`
  - `src/include/`
    - `src/include/applications` - definition of the applications (edgelist format)
    - `config.py` - configuration file
    - `enums.puy` - enums for the project
  - `src/requests` - generated traces
  - `allocation_heuristic.py` - TANTO/OLIVE implementation
  - `allocation_logger.py` - classes used for logging and debug
  - `application.py` - application handler
  - `comparison_algorithms.py` - greedy algorithms
  - `element_translator.py` - converter for embedding result into processable format
  - `enhanced_graph.py` - override networkx DiGraph
  - `experiments.py` - executor of experiments
  - `fluid_model.py` - linear program
  - `multiplier.py` - multiplier of sizes from virtual to physical layers
  - `online_handler.py` - helper functions for the online mode
  - `oracle_runner.py` - SlotOFF implementation
  - `path_finder.py` - parses physical topology for virtual embedding
  - `physical_graph.py` - process physical topology
  - `req_timing_gen.py` - generates online traces
  - `result_exporter.py` - export results
  - `user_processor.py` - process requests
  - `users.py` - request class

### Reading the Results
#### Offline Mode
The results are stored in the `Results/Offline/<topology>` directory.
Each iteration-specific `Run_<run_number>` directory contains detailed information about each executed algorithm.
`Run_summary/online/<algorithm>.csv` contains a per-timeslot information log of each algorithm. These are the main files used to generate the results for the paper.

#### Online Mode
The results are stored in the `Results/Online/<topology>` directory.
Each iteration-specific `Run_<run_number>` directory contains detailed information about each executed algorithm.



### Running the Code
The code is executed from the `main.py` file. 
Ensure CPLEX is installed and the path to the CPLEX library (e.g., `\IBM\ILOG\CPLEX_Studio2211\cplex\python\3.10\x64_win64`) is set in the env.
The `config.py` file contains the configuration for the experiments, including online/offline mode, physical and virtual topologies, requests, and algorithms.


