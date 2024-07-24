import os
import copy
import random
import numpy as np
import networkx as nx

from .network import Network
from .attribute import NodeInfoAttribute, LinkInfoAttribute


class PhysicalNetwork(Network):
    """
    PhysicalNetwork class is a subclass of Network class. It represents a physical network.

    Attributes:
        degree_benchmark (dict): The degree benchmark for the network.
        node_attr_benchmarks (dict): The node attribute benchmarks for the network.
        link_attr_benchmarks (dict): The link attribute benchmarks for the network.

    Methods:
        from_topology_zoo_setting(topology_zoo_setting: Dict[str, Any], seed: Optional[int] = None) -> PhysicalNetwork:
            Returns a PhysicalNetwork object generated from the Topology Zoo data, with optional seed for reproducibility.

        save_dataset(self, dataset_dir: str) -> None:
            Saves the dataset as a .gml file in the specified directory.

        load_dataset(dataset_dir: str) -> PhysicalNetwork:
            Loads the dataset from the specified directory as a PhysicalNetwork object,
            and calculates the benchmarks for normalization.
    """

    def __init__(
        self,
        graph_data: nx.Graph = None,
        node_attrs_setting: list = [],
        link_attrs_setting: list = [],
        **kwargs,
    ) -> None:
        """
        Initialize a PhysicalNetwork object.

        Args:
            incoming_graph_data (nx.Graph): An existing graph object (optional).
            node_attrs_setting (list): Node attribute settings (default []).
            link_attrs_setting (list): Link attribute settings (default []).
            **kwargs: Additional keyword arguments.
        """
        super(PhysicalNetwork, self).__init__(
            graph_data,
            node_attrs_setting,
            link_attrs_setting,
            **kwargs,
        )

    def generate_topology(self, num_nodes: int, type: str = "waxman", **kwargs):
        """Generate a topology in the specified format

        Args:
            num_nodes (int): the number of nodes of the topology
            typs (str, optional): The type of network to generate. Defaults to 'waxman'
            **kwargs: Keyword arguments to pass to the network generator
        """
        super().generate_topology(num_nodes=num_nodes, type=type, **kwargs)

        # self.degree_benchmark = self.get_degree_benchmark()

    def generate_attrs_data(self, n_bool: bool = True, l_bool: bool = True):
        """
        Generate attribute data for the network.

        Args:
            n_bool (bool, optional): Whether or not to generate node attribute data. Defaults to True.
            l_bool (bool, optional): Whether or not to generate link attribute data. Defaults to True.
        """
        super().generate_attrs_data(n_bool=n_bool, l_bool=l_bool)

    @staticmethod
    def load_from_setting(setting: dict, seed: int = None) -> "PhysicalNetwork":
        """
        Load a PhysicalNetwork object from the given setting.

        1. Create a new PhysicalNetwork instance `net`.
        2. Load a networkx instance `G` from `.gml` file
        3. copy attrs of `G` to `net`
        4. Generate net attr data

        Args:
            setting (dict): The network settings.
            seed (int): The random seed for network generation.

        Returns:
            PhysicalNetwork: A PhysicalNetwork object.
        """
        # 1. Create a new PhysicalNetwork instance `net`.
        fpath = setting["topology"].get("file_path")
        net = PhysicalNetwork(
            node_attrs_setting=node_attrs_setting,
            link_attrs_setting=link_attrs_setting,
            **setting,
        )

        # 2. Load a networkx instance `G` from `.gml` file
        G = nx.read_gml(fpath, label="id")

        # 3. copy attrs of `G` to `net`
        # clean the data, because we already have our own `node_attrs_setting/link_attrs_setting``
        if "node_attrs_setting" in G.__dict__["graph"]:
            G.__dict__["graph"].pop("node_attrs_setting")
        if "link_attrs_setting" in G.__dict__["graph"]:
            G.__dict__["graph"].pop("link_attrs_setting")
        net.__dict__["graph"].update(G.__dict__["graph"])
        net.__dict__["_node"] = G.__dict__["_node"]
        net.__dict__["_adj"] = G.__dict__["_adj"]

        n_attr_names = net.nodes[list(net.nodes)[0]].keys()
        for n_attr_name in n_attr_names:
            if n_attr_name not in net.node_attrs.keys():
                net.node_attrs[n_attr_name] = NodeInfoAttribute(n_attr_name)

        l_attr_names = net.links[list(net.links)[0]].keys()
        for l_attr_name in l_attr_names:
            if l_attr_name not in net.link_attrs.keys():
                net.link_attrs[l_attr_name] = LinkInfoAttribute(l_attr_name)

        # 4. Generate net attr data
        if seed is None:
            seed = setting.get("seed", 0)
        random.seed(seed)
        np.random.seed(seed=seed)

        net.generate_attrs_data()

        print(f"Loaded the topology from {fpath}")

        return net

    @staticmethod
    def create_from_setting(setting: dict, seed: int = None) -> "PhysicalNetwork":
        """
        Create a PhysicalNetwork object from the given setting.

        - create a net instance
        - create node / link attrs
        - generate node / link attrs data

        Args:
            setting (dict): The network settings.
            seed (int): The random seed for network generation.

        Returns:
            PhysicalNetwork: A PhysicalNetwork object.
        """

        setting = copy.deepcopy(setting)
        node_attrs_setting = setting.pop("node_attrs_setting")
        link_attrs_setting = setting.pop("link_attrs_setting")

        num_nodes = setting.get("num_nodes")
        net = PhysicalNetwork(
            node_attrs_setting=node_attrs_setting,
            link_attrs_setting=link_attrs_setting,
            **setting,
        )
        topology_setting = setting.pop("topology")
        net.generate_topology(num_nodes, **topology_setting)

        if seed is None:
            seed = setting.get("seed", 0)
        random.seed(seed)
        np.random.seed(seed=seed)

        net.generate_attrs_data()

        return net

    def save_net(self, fpath: str) -> None:
        """
        Save the physical network dataset to a directory.

        Args:
            fpath (str): The path to the directory where the physical network dataset is to be saved.
        """
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        file_path = os.path.join(fpath, "p_net.gml")
        self.save_to_gml(file_path)

    def load_from_gml(self, fpath: str) -> "PhysicalNetwork":
        """
        Load the physical network dataset from a directory.

        Args:
            fpath (str): The path to the directory where the physical network dataset is stored.
        """
        if not os.path.exists(fpath):
            raise ValueError(
                f"Find no dataset in {fpath}.\n Please firstly generating it."
            )
        file_path = os.path.join(fpath, "p_net.gml")
        # p_net = PhysicalNetwork.load_from_gml(file_path)
        p_net = super().load_from_gml(file_path)

        return p_net


if __name__ == "__main__":

    node_attrs_setting = [
        {
            "name": "cpu",
            "distribution": "uniform",
            "dtype": "int",
            "generative": True,
            "high": 100,
            "low": 50,
            "owner": "node",
            "type": "resource",
        },
        {"name": "max_cpu", "originator": "cpu", "owner": "node", "type": "extrema"},
        {
            "name": "gpu",
            "distribution": "uniform",
            "dtype": "int",
            "generative": True,
            "high": 100,
            "low": 50,
            "owner": "node",
            "type": "resource",
        },
        {"name": "max_gpu", "originator": "gpu", "owner": "node", "type": "extrema"},
        {
            "name": "ram",
            "distribution": "uniform",
            "dtype": "int",
            "generative": True,
            "high": 100,
            "low": 50,
            "owner": "node",
            "type": "resource",
        },
        {"name": "max_ram", "originator": "ram", "owner": "node", "type": "extrema"},
    ]
    link_attrs_setting = [
        {
            "name": "bw",
            "distribution": "uniform",
            "dtype": "int",
            "generative": True,
            "high": 100,
            "low": 50,
            "owner": "link",
            "type": "resource",
        },
        {"name": "max_bw", "originator": "bw", "owner": "link", "type": "extrema"},
    ]
    kwargs = {
        "num_nodes": 100,
        "save_dir": "dataset/p_net",
        "topology": {
            "type": "waxman",
            "wm_alpha": 0.5,
            "wm_beta": 0.2,
            "file_path": "dataset/topology/Waxman100.gml",
        },
        "file_name": "p_net.gml",
    }

    """
    Get a Physical Network instance
    """
    G = PhysicalNetwork(
        None,
        node_attrs_setting=node_attrs_setting,
        link_attrs_setting=link_attrs_setting,
        **kwargs,
    )
    """
    Create node/link attrs
    """
    print("G.attrs: ", G.graph.keys())
    for k, v in G.graph.items():
        print(k, v)

    print(G.graph["node_attrs_setting"])
    print(G.graph["link_attrs_setting"])

    print("G.node_attrs: ", G.node_attrs.keys())
    print("G.link_attrs: ", G.link_attrs.keys())

    print("G.get_node_attrs()[0]: ", G.get_node_attrs()[0])
    print("G.get_link_attrs()[0]: ", G.get_link_attrs()[0])

    """
    Create G topology
    """
    G.generate_topology(
        num_nodes=kwargs["num_nodes"],
        type=kwargs["topology"]["type"],
        kwargs=kwargs["topology"],
    )
    print("G.number_of_nodes(): ", G.number_of_nodes())  # 100
    print("G.nodes.items()[0]: ", list(G.nodes.items())[0])  # 自带 `pos` 属性

    """
    Generate node/link attrs data
    """
    G.generate_attrs_data(n_bool=True, l_bool=False)
    print("G.nodes.items()[0]: ", list(G.nodes.items())[0])

    """
    Get G node/link attrs data
    """
    print("cpu attr_data: ", G.get_node_attrs_data(["cpu"]))
