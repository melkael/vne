# vne.jl

This is a collection of algorithms for the Virtual Network Embedding problem.
Please cite the our following paper if you use this repo:
>Maxime Elkael and Massinissa Ait Aba and Andrea Araldo and Hind Castel and Badii Jouaber, "Monkey Business: Reinforcement learning meets neighborhood search for Virtual Network Embedding", 2022,  https://arxiv.org/abs/2202.13706, in review.

Model:
* A virtual node must be placed on one physical node
* Two virtual nodes from the same VNR (virtual network request) cannot use the same physical node
* A virtual link is placed on a single path (possible to use path-splitting instead, see below)
* Virtual nodes & link should respect capacity constraints (CPU/BW)

## Dependencies
You need to install Julia to run the implemented algorithms
Then install the following dependencies (using Pkg.add(...) in the Julia prompt):
* Graphs
* MetaGraphs
* DataStructures
* JSON
* StatsBase
* JLD2


## Implemented algorithms
* NEPA  
>Maxime Elkael and Massinissa Ait Aba and Andrea Araldo and Hind Castel and Badii Jouaber, "Monkey Business: Reinforcement learning meets neighborhood search for Virtual Network Embedding", 2022,  https://arxiv.org/abs/2202.13706, in review.
```
julia nepa.jl <scenario folder> <log file> <level> <N> <use distance heuristic(true/false)> <level refine> <number of refinements> <random seed>
```
* NRPA
>M. Elkael, H. Castel-Taleb, B. Jouaber, A. Araldo and M. A. Aba, "Improved Monte Carlo Tree Search for Virtual Network Embedding," 2021 IEEE 46th Conference on Local Computer Networks (LCN), 2021, pp. 605-612.

usage:
```
julia nrpa.jl <scenario folder> <log file> <level> <N> <use distance heuristic(true/false)> <random seed>
```
* MCTS from 
>S. Haeri and L. Trajković, "Virtual Network Embedding via Monte Carlo Tree Search," in IEEE Transactions on Cybernetics, vol. 48, no. 2, pp. 510-521, Feb. 2018, doi: 10.1109/TCYB.2016.2645123.

usage:
```
julia mcts.jl <scenario folder> <log file> <total budget> <random seed>
```
Where total budget will then be divided by the number of nodes of each VNR during placement, so the algorithm spends  `total budget>/num_nodes` per node

* UEPSO 
>Zhang, Z., Cheng, X., Su, S., Wang, Y., Shuang, K., & Luo, Y. (2013). A unified enhanced particle swarm optimization‐based virtual network embedding algorithm. International Journal of Communication Systems, 26(8), 1054-1073.

usage:
```
julia uepso.jl <scenario folder> <log file> <number of particles> <number of iterations> <random seed>
```
