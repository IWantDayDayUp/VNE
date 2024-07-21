import os
import copy
import random
import numpy as np
import networkx as nx

from network import Network
from attribute import NodeInfoAttribute, LinkInfoAttribute


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
        super().generate_attrs_data(n_bool=n_bool, l_bool=n_bool)

    @staticmethod
    def create_from_setting(setting: dict, seed: int = None) -> "PhysicalNetwork":
        """
        Create a PhysicalNetwork object from the given setting.

        Args:
            setting (dict): The network settings.
            seed (int): The random seed for network generation.

        Returns:
            PhysicalNetwork: A PhysicalNetwork object.
        """
        # TODO: why deepcopy? why not dict.get(key, default)?
        setting = copy.deepcopy(setting)
        node_attrs_setting = setting.pop("node_attrs_setting")
        link_attrs_setting = setting.pop("link_attrs_setting")

        try:
            if (
                "file_path" not in setting["topology"]
                and setting["topology"]["file_path"] not in ["", None, "None", "null"]
                and os.path.exists(setting["topology"]["file_path"])
            ):
                raise Exception("Invalid file path for topology.")

            fpath = setting["topology"].get("file_path")
            net = PhysicalNetwork(
                node_attrs_setting=node_attrs_setting,
                link_attrs_setting=link_attrs_setting,
                **setting,
            )
            G = nx.read_gml(fpath, label="id")

            # clean the data
            if "node_attrs_setting" in G.__dict__["graph"]:
                G.__dict__["graph"].pop("node_attrs_setting")
            if "link_attrs_setting" in G.__dict__["graph"]:
                G.__dict__["graph"].pop("link_attrs_setting")
            net.__dict__["graph"].update(G.__dict__["graph"])
            net.__dict__["_node"] = G.__dict__["_node"]
            net.__dict__["_adj"] = G.__dict__["_adj"]

            # node attr
            n_attr_names = net.nodes[list(net.nodes)[0]].keys()
            for n_attr_name in n_attr_names:
                if n_attr_name not in net.node_attrs.keys():
                    net.node_attrs[n_attr_name] = NodeInfoAttribute(n_attr_name)

            # link attr
            l_attr_names = net.links[list(net.links)[0]].keys()
            for l_attr_name in l_attr_names:
                if l_attr_name not in net.link_attrs.keys():
                    net.link_attrs[l_attr_name] = LinkInfoAttribute(l_attr_name)

            print(f"Loaded the topology from {fpath}")

        except Exception as e:
            num_nodes = setting.get("num_nodes")
            net = PhysicalNetwork(
                node_attrs_setting=node_attrs_setting,
                link_attrs_setting=link_attrs_setting,
                **setting,
            )
            topology_setting = setting.pop("topology")
            # topology_type = topology_setting.pop('type')
            net.generate_topology(num_nodes, **topology_setting)

        if seed is None:
            seed = setting.get("seed")
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
        p_net = PhysicalNetwork.load_from_gml(file_path)

        return p_net
