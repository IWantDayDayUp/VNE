# VNE

## System Model

### Physical Network

The Physical Network is modeled as a **undirected graph** $G_p=(N^p, L^p)$, where

- $N^p$: the set of physical nodes. Each physical node $n^p \in N^p$ has a set of computing resources, such as CPU, GPU memory, and bandwidth , which are represented as a vector $C(n^p)$.
- $L^p$: the set of physical links. Each physical link $l^p \in L^p$ has a bandwidth capacipy $B(l^p)$.

### Virtual Network

The Virtual Network is modeled as a **undirected graph** $G_v=(N^v, L^v, d^v)$, where

- $N^v$: the set of virtual nodes. Each virtual node $n^v \in N^v$ has a set of computing requirements, such as CPU, GPU memory, and bandwidth , which are represented as a vector $R(n^v)$.
- $L^v$: the set of virtual links. Each virtual link $l^v \in L^v$ has a bandwidth requirement $B(l^v)$.
- $d^v$: the lifetime of the request. Once the VNR is accepted, it will be maintained for $d^v$ time slots.

## Mapping Process

The mapping process aims to map the virtual nodes and links onto the substrate network with minimal resource cost while satisfying various QoS constraints.

$$ f:G^v \rightarrow G^p $$

### Node Mapping $f_n:N^v \rightarrow N^p$

Node mapping involves assigning each virtual node $n^v \in N^v$ to a physical node $n^p \in N^p$. In this process, the following constraints should be satisfied:

- **One-to-one mapping constraint**: Each virtual node should be mapped to exactly one substrate node.
- **Computing Resource Availability**: The computing resources required by the virtual node should be available on the physical node.

### Link Mapping $f_l:L^v \rightarrow L^p$

Link mapping involves finding a physical path $l^p \in L^p$ for each virtual link $l^v \in L^v$. In this process, the following constraints should be satisfied:

- **Link-to-path mapping constraint**: Each virtual link can only be mapped to a path consisting of physical links.
- **Connectivity constraint**: The mapping should preserve the connectivity of the virtual network, i.e., if there is a virtual link between two virtual nodes, the corresponding physical nodes should be connected by a physical link.
- **Link resource constraint**: The sum of the bandwidth requirements of the virtual links mapped to a physical link cannot exceed its capacity.

## Evaluatiion Metric

### Resource Efficiency

For the network provider, they aim to maximize the revenue generated by the accepted virtual network requests while minimizing the cost incurred by the mapped virtual networks.

- **Total Revenue**

> The total revenue measures the obtained revenue of the network provider. It is defined as the sum of the revenue generated by the accepted virtual network requests.

$$Total Revenue = \sum_{n^v \in N^v}{Revenue(n^v)} + \sum_{l^v \in L^v}{Revenue(l^v)}$$

- **Total Cost**

> The total cost measures the cost incurred by the network provider. It is defined as the sum of the cost incurred by the mapped virtual networks.

$$Total Cost = \sum_{n^v \in N^v}{Cost(n^v)} + \sum_{l^v \in L^v}{Cost(l^v)}$$

- **Revenue-to-Cost Ratio**

> The revenue-to-cost ratio measures the efficiency of the network provider in generating revenue while minimizing the cost. It is defined as the ratio of the total revenue to the total cost incurred by the mapped virtual networks.

$$Revenue-to-Cost Ratio = \frac{Total Revenue}{Total Cost}$$

- **Resource Utilization**

> The resource utilization measures the efficiency of resource usage in the physical network. It is defined as the ratio of the total resources used by the mapped virtual networks to the total resources available in the physical network.

$$Resource Utilization = \frac{\sum_{n^v \in N^v}{R(n^v)} + \sum_{l^v \in L^v}{B(l^v)}}{\sum_{n^p \in N^p}{C(n^p)} + \sum_{l^p \in L^p}{B(l^p)}}$$

### QoS Satisfaction

For the network users, they aim to ensure that their service requests are satisfied with the desired QoS requirements.

- **Acceptance Ratio**

> The acceptance ratio measures the percentage of accepted virtual network requests among all incoming requests. It is defined as the ratio of the number of accepted virtual network requests to the total number of incoming requests.

$$Acceptance Ratio = \frac{Number of Accepted Requests}{Total Number of Requests}$$

- **QoS Violation Ratio**

> The QoS violation ratio measures the percentage of virtual network requests that violate the QoS constraints. It is defined as the ratio of the number of violated requests to the total number of requests.

$$QoS Violation Ratio = \frac{Number of Violated Requests}{Total Number of Requests} $$

## Simulation Setting

### General Setting

#### Physical Network Setting - "p_net_setting.yaml"

- num_nodes: the number of nodes in the physical network.
- save_dir: the directory in which to save the output file.
- topology: the topology to be used for the physical network. This can either be a file path to a .gml file or one of several built-in network models. In this case, a Waxman model is used with parameters wm_alpha: 0.5 and wm_beta: 0.2.
- link_attrs_setting: the attributes to be assigned to links in the physical network. In this case, a single attribute is specified: bw, which represents the link bandwidth. It is assigned a uniform distribution with minimum value 50 and maximum value 100.
- node_attrs_setting: the attributes to be assigned to nodes in the physical network. In this case, a single attribute is specified: cpu, which represents the node’s processing power. It is assigned a uniform distribution with minimum value 50 and maximum value 100.
- file_name: the name of the output file to be generated, which will be saved in the save_dir directory. In this case, the file name is p_net.gml.

#### Virtual Network Request Setting - "v_net_setting.yaml"

- num_nodes: the number of nodes in the physical network.
- save_dir: the directory in which to save the output file.
- topology: the topology to be used for the physical network. This can either be a file path to a .gml file or one of several built-in network models. In this case, a Waxman model is used with parameters wm_alpha: 0.5 and wm_beta: 0.2.
- link_attrs_setting: the attributes to be assigned to links in the physical network. In this case, a single attribute is specified: bw, which represents the link bandwidth. It is assigned a uniform distribution with minimum value 50 and maximum value 100.
- node_attrs_setting: the attributes to be assigned to nodes in the physical network. In this case, a single attribute is specified: cpu, which represents the node’s processing power. It is assigned a uniform distribution with minimum value 50 and maximum value 100.
- file_name: the name of the output file to be generated, which will be saved in the save_dir directory. In this case, the file name is p_net.gml.

### Topology Setting

## Implemented Algorithms

- []

## Project structure

```shell

.
├── args.py
├── main.py
├── settings
│   ├── p_net.yaml  # Simulation setting of physical network 
│   ├── v_sim.yaml  # Simulation setting of virtual network request simulator 
└── vne_simulator
    ├── base                # Core components: environment, controller, recorder, scenario, solution
    ├── config.py           # Configuration class
    ├── data                # Data class: attribute, generator, network, physical_network, virtual_network, virtual_network_request_simulator
    ├── solver
    │   ├── heuristic                                    # 
    │   │   ├── node_rank.py                             # Baseline-1 & 2: NRM-VNE and NEA-VNE
    │   ├── learning                                     # 
    │   │   ├── flag_vne                                 # Our Algorithm and its variations: FlagVNE, FlagVNE-MetaFree-SinglePolicy, FlagVNE-MetaFree-MultiPolicies, FlagVNE-NoCurriculum, FlagVNE-NEARank
    │   │   ├── a3c_gcn                                  # Baseline-3 and its variations: A3C-GCN, A3C-GCN-NRM, A3C-GCN-NEA, A3C-GCN-MultiPolicies
    │   │   ├── ddpg_attention                           # Baseline-4: DDPG-Attention
    │   │   ├── mcts                                     # Baseline-5: MCTS
    │   │   └── pg_cnn                                   # Baseline-6: PG-CNN
    │   ├── meta_heuristic                               #
    │   │   └── particle_swarm_optimization_solver.py    # Baseline-7: PSO-VNE
    │   ├── registry.py
    │   └── solver.py
    └── utils

```
