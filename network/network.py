import networkx as nx
import copy

from typing import Optional
from functools import cached_property
from .attribute import Attribute
from networkx.classes.reportviews import EdgeView


class Network(nx.Graph):
    """
    Network class inherited from networkx.Graph.

    Attributes:
        node_attrs: Node attributes.
        link_attrs: Link attributes.
        graph_attrs: Graph attributes.

    Methods:
        create_attrs_from_setting: Create node and link attribute dictionaries from their respective settings.
        init_graph_attrs: Initialize graph attributes.
        set_net_attrs: Set graph attributes data.
        get_graph_attrs_data: Get graph attributes data.
        get_node_attrs: Get node attributes.
        get_link_attrs: Get link attributes.
        get_node_attrs_data: Get node attributes data.
        get_link_attrs_data: Get link attributes data.
        get_node_attr_data: Get node attribute data.
        get_link_attr_data: Get link attribute data.
        set_node_attrs_data: Set node attributes data.
        set_link_attrs_data: Set link attributes data.
        set_node_attr_data: Set node attribute data.
        set_link_attr_data: Set link attribute data.
        generate_topology: Generate the network topology.
        check_attrs_existence: Check if all defined attributes exist in the graph.
        write_setting: Write network setting to file.
    """

    def __init__(
        self,
        incoming_graph_data: Optional[nx.Graph] = None,
        node_attrs_setting: list = [],
        link_attrs_setting: list = [],
        **kwargs,
    ):
        """
        Initializes a new Network instance.

        Args:
            incoming_graph_data (optional, Graph): Graph instance to convert to Network or data to initialize graph. Default: None.
            node_attrs_setting (list): List of dictionaries containing node attribute settings. Default: [].
            link_attrs_setting (list): List of dictionaries containing link attribute settings. Default: [].
            **kwargs: Additional keyword arguments to set graph attributes.
        """
        super(Network, self).__init__(incoming_graph_data)

        # TODO
        # init graph attributes
        self.init_graph_attrs()

        # set graph attributes
        # self.graph: dictionary for graph attributes
        self.graph["node_attrs_setting"] += node_attrs_setting
        self.graph["link_attrs_setting"] += link_attrs_setting
        self.create_attrs_from_setting()

        # Read extra kwargs
        self.set_net_attrs(kwargs)

    def generate_topology(self, num_nodes: int, type: str = "path", **kwargs):
        """
        Generate the network topology.

        Args:
            num_nodes: the number of nodes in the generated graph
            type: the type of graph to generate, currently supported types are: 'path', 'star', 'waxman', and 'random'
            **kwargs: additional keyword arguments required for certain graph types
        """
        assert num_nodes >= 1
        assert type in ["path", "star", "waxman", "random"], ValueError(
            "Unsupported graph type!"
        )

        self.set_net_attrs({"num_nodes": num_nodes, "type": type})
        if type == "path":
            G = nx.path_graph(num_nodes)
        elif type == "star":
            G = nx.star_graph(num_nodes)
        elif type == "grid_2d":
            wm_alpha = kwargs.get("m")
            wm_beta = kwargs.get("n")
            G = nx.grid_2d_graph(num_nodes, periodic=False)
        elif type == "waxman":
            wm_alpha = kwargs.get("wm_alpha", 0.5)
            wm_beta = kwargs.get("wm_beta", 0.2)
            not_connected = True
            # TODO: 封装
            while not_connected:
                G = nx.waxman_graph(num_nodes, wm_alpha, wm_beta)
                not_connected = not nx.is_connected(G)
            self.set_net_attrs({"wm_alpha": wm_alpha, "wm_beta": wm_beta})
        elif type == "random":
            random_prob = kwargs.get("random_prob", 0.5)
            self.set_net_attrs({"random_prob": random_prob})
            G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
            # TODO: 封装
            not_connected = True
            while not_connected:
                G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
                not_connected = not nx.is_connected(G)
        else:
            raise NotImplementedError
        self.__dict__["_node"] = G.__dict__["_node"]
        self.__dict__["_adj"] = G.__dict__["_adj"]

    def set_net_attrs(self, attrs_data: dict):
        """Set some attributes of the network"""
        for key, value in attrs_data.items():
            self.set_net_attr(key, value)

    def set_net_attr(self, attr_name, value):
        """Set a attribute of the network: net.attr_name = value."""
        if attr_name in ["num_nodes"]:
            self.graph[attr_name] = value
            return
        self.graph[attr_name] = value
        self[attr_name] = value

    def init_graph_attrs(self):
        """Initialize the graph attributes."""
        # TODO: 这里的逻辑可能有问题: 为什么不是所有属性全归零, 而是有选择的赋值?
        self.init_node_attr()
        self.init_link_attr()
        self.init_graph_items()

    def init_graph_items(self):
        """Initialize all the attributes of self.graph"""
        for key, value in self.graph.items():
            if key not in ["num_nodes"]:
                self[key] = value

    def init_node_attr(self):
        """Initialize the node attributes of the net"""
        self.graph["node_attrs_setting"] = (
            [] if "node_attrs_setting" not in self.graph else None
        )

    def init_link_attr(self):
        """Initialize the link attributes of the net"""
        self.graph["link_attrs_setting"] = (
            [] if "link_attrs_setting" not in self.graph else None
        )

    def create_attrs_from_setting(self):
        """Create node and link attribute dictionaries from their respective settings."""
        # TODO: 封装
        self.node_attrs = {
            n_attr_dict["name"]: Attribute.from_dict(n_attr_dict)
            for n_attr_dict in self.graph["node_attrs_setting"]
        }
        self.link_attrs = {
            e_attr_dict["name"]: Attribute.from_dict(e_attr_dict)
            for e_attr_dict in self.graph["link_attrs_setting"]
        }

    def check_net_attrs(self):
        """Check if all defined attributes exist in the net"""
        self.check_node_attr()
        self.check_link_attr()

    def check_node_attr(self):
        """check node attrs"""
        for n_attr_name in self.node_attrs.keys():
            assert (
                n_attr_name in self.nodes[list(self.nodes)[0]].keys()
            ), f"{n_attr_name}"

    def check_link_attr(self):
        """check link attrs"""
        for l_attr_name in self.link_attrs.keys():
            assert (
                l_attr_name in self.links[list(self.links)[0]].keys()
            ), f"{l_attr_name}"

    def generate_attrs_data(self, n_bool: bool = False, l_bool: bool = False):
        """Generate the data of network attributes based on attributes."""
        if n_bool:
            self.generate_node_attr()
        if l_bool:
            self.generate_link_attr()

    def generate_node_attr(self):
        """Generate node attr"""
        for n_attr in self.node_attrs.values():
            if n_attr.generative or n_attr.type == "extrema":
                attribute_data = n_attr.generate_data(self)
                n_attr.set_data(self, attribute_data)

    def generate_link_attr(self):
        """Generate link attr"""
        for l_attr in self.link_attrs.values():
            if l_attr.generative or l_attr.type == "extrema":
                attribute_data = l_attr.generate_data(self)
                l_attr.set_data(self, attribute_data)

    @property
    def num_node_features(self) -> int:
        """
        Get the number of node features.

        Returns:
            int: The number of node features.
        """
        return len(self.node_attrs)

    @property
    def num_link_features(self) -> int:
        """
        Get the number of link features.

        Returns:
            int: The number of link features.
        """
        return len(self.link_attrs)

    @property
    def num_node_resource_features(self) -> int:
        """
        Get the number of node resource features.

        Returns:
            int: The number of node resource features.
        """
        return len(
            [
                node_attr
                for node_attr in self.node_attrs.values()
                if node_attr.type == "resource"
            ]
        )

    @property
    def num_link_resource_features(self) -> int:
        """
        Get the number of link resource features.

        Returns:
            int: The number of link resource features.
        """
        return len(
            [
                link_attr
                for link_attr in self.link_attrs.values()
                if link_attr.type == "resource"
            ]
        )

    @cached_property
    def num_nodes(self) -> int:
        """Get the number of nodes."""
        return self.number_of_nodes()

    @cached_property
    def num_links(self) -> int:
        """Get the number of links."""
        return self.number_of_edges()

    @cached_property
    def num_edges(self) -> int:
        """Get the number of edges(links)."""
        return self.number_of_edges()

    @cached_property
    def links(self):
        """Get the number of links."""
        return EdgeView(self)

    @property
    def adj_matrix(self):
        """Get the adjacency matrix of Network."""
        return nx.to_scipy_sparse_matrix(self, format="csr")

    def get_net_attrs(self, names: list = None):
        """
        Get the attributes of the network.

        Args:
            names (list): The names of the attributes to retrieve. If None, return all attributes.

        Returns:
            dict: A dictionary of network attributes.
        """
        if names is None:
            return self.graph
        return {attr: self.graph[attr] for attr in names}

    def get_node_attrs(self, types: list = None, names: list = None):
        """
        Get the node attributes of the network.

        Args:
            types (list): The types of the node attributes to retrieve. If None, return all attributes.
            names (list): The names of the node attributes to retrieve. If None, return all attributes.

        Returns:
            list: A list of node attributes.
        """
        if types is None and names is None:
            return list(self.node_attrs.values())
        elif types is not None:
            selected_node_attrs = []
            for n_attr in self.node_attrs.values():
                selected_node_attrs.append(n_attr) if n_attr.type in types else None
        elif names is not None:
            selected_node_attrs = []
            for n_attr in self.node_attrs.values():
                selected_node_attrs.append(n_attr) if n_attr.name in names else None
        return selected_node_attrs

    def get_link_attrs(self, types: list = None, names: list = None):
        """Get the link attributes of the network.

        Args:
            types (list): The types of the link attributes to retrieve. If None, return all attributes.
            names (list): The names of the link attributes to retrieve. If None, return all attributes.

        Returns:
            list: A list of link attributes.
        """
        if types is None and names is None:
            return list(self.link_attrs.values())
        elif types is not None:
            selected_link_attrs = []
            for l_attr in self.link_attrs.values():
                selected_link_attrs.append(l_attr) if l_attr.type in types else None
        elif names is not None:
            selected_link_attrs = []
            for l_attr in self.link_attrs.values():
                selected_link_attrs.append(l_attr) if l_attr.name in names else None
        return selected_link_attrs

    def get_node_attrs_data(self, attrs_names):
        """Get the data of node attributes."""
        if isinstance(attrs_names[0], str):
            node_attrs_data = [
                list(nx.get_node_attributes(self, n_attr_name).values())
                for n_attr_name in attrs_names
            ]
        else:
            node_attrs_data = [n_attr.get_data(self) for n_attr in attrs_names]
        return node_attrs_data

    def get_link_attrs_data(self, link_attrs):
        """Get the data of link attributes."""
        if isinstance(link_attrs[0], str):
            link_attrs_data = [
                list(nx.get_edge_attributes(self, l_attr_name).values())
                for l_attr_name in link_attrs
            ]
        else:
            link_attrs_data = [l_attr.get_data(self) for l_attr in link_attrs]
        return link_attrs_data

    def get_adjacency_attrs_data(self, link_attrs, normalized=False):
        """Get the data of adjacency attributes."""
        adjacency_data = [
            l_attr.get_adjacency_data(self, normalized) for l_attr in link_attrs
        ]
        return adjacency_data

    def get_aggregation_attrs_data(self, link_attrs, aggr="sum", normalized=False):
        """Get the data of aggregation attributes."""
        aggregation_data = [
            l_attr.get_aggregation_data(self, aggr, normalized) for l_attr in link_attrs
        ]
        return aggregation_data

    def update_node_resources(self, node_id, v_net_node, method="+"):
        """Update (increase) the value of node atributes."""
        for n_attr in self.node_attrs.keys():
            if n_attr.type != "resource":
                continue
            n_attr.update(self.nodes[node_id], v_net_node, method)

    def update_link_resources(self, link_pair, v_net_link, method="+"):
        """Update (increase) the value of link atributes."""
        for l_attr in self.link_attrs:
            if l_attr.type != "resource":
                continue
            l_attr.update(self.links[link_pair], v_net_link, method)

    def update_path_resources(self, path, v_net_link, method="+"):
        """Update (increase) the value of links atributes of path with the same increments."""
        assert len(path) >= 1
        for l_attr in self.link_attrs:
            l_attr.update_path(self, path, v_net_link, method)

    def save_to_gml(self, fpath):
        nx.write_gml(self, fpath)

    @classmethod
    def load_from_gml(cls, fpath):
        gml_net = nx.read_gml(fpath, destringizer=int)
        net = cls(incoming_graph_data=gml_net)
        net.check_net_attrs()
        return net

    def save_attrs_dict(self, fpath):
        attrs_dict = {
            "graph_attrs_dict": self.get_graph_attrs(),
            "node_attrs": [n_attr.to_dict() for n_attr in self.node_attrs.values()],
            "link_attrs": [l_attr.to_dict() for l_attr in self.link_attrs.values()],
        }
        # TODO: 考虑到写入特定文件的多处使用, 是不是应该封装?
        # write_setting(attrs_dict, fpath)

    def clone(self):
        return self.__class__.from_dict(
            {k: copy.deepcopy(v) for k, v in self.__dict__.items()}
        )


if __name__ == "__main___":
    pass
