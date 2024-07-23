import copy
import numpy as np
import networkx as nx

import os, sys

sys.path.append(
    ".."
)  # 等价于 sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.path import path_to_links
from utils.data import generate_data_with_distribution


class Attribute(object):
    """
    Attribute class inherited from object.

    Attributes:
        name: the name of the attr
        owner: the attr belongs to node or link
        type: the type of the attr
        originator: Initial value
        generative: whether the attribute value needs to be generated
        dtype: the value of attr is int, float or bool
        distribution:
            "unifrom": low, high
            "normal": loc, scale
            "exponential": scale
            "possion": scale
            "customized": min, max

    Methods:
        create_attr_from_dict: Create attribute from `dict`
        get_attr_by_id: Get the `attr_value` of the node/edge with the node/edge id
        get_attr_size: The size of attributes usually refers to the number of nodes/links in the network
        generate_attr_data: Generate data with a specified size, data type, and distribution based on the attribute settings
        to_dict: Get a `dict` representation of all the properties of the object
        __repr__: Get a `string` representation of all the properties of the object
    """

    def __init__(self, name: str, owner: str, type: str, *args, **kwargs) -> None:
        """
        Initializes a new Attribute instance.

        Args:
            name (str): the name of the attribute, such as: `cpu`, `max_cpu`, `gpu`, `max_gpu`, `ram`, `max_ram`, ...
            owner (str): whether the attribute belongs to a node or a link, such as: "node", "link"
            type (str): the type of the attribute, such as: "resource", "extrema"
        """
        self.name = name
        self.owner = owner
        self.type = type

        self.originator = kwargs.get("originator") if type == "extrema" else None

        self.generative = kwargs.get("generative", False)
        assert self.generative in [True, False]
        if self.generative:
            self.distribution = kwargs.get("distribution", "normal")
            self.dtype = kwargs.get("dtype", "float")

            # TODO: create a new fun to check all of the args
            assert self.distribution in [
                "normal",
                "uniform",
                "exponential",
                "possion",
                "customized",
            ]
            assert self.dtype in ["int", "float", "bool"]

            if self.distribution in ["uniform"]:
                self.low = kwargs.get("low", 0.0)
                self.high = kwargs.get("high", 1.0)
            elif self.distribution in ["normal"]:
                self.loc = kwargs.get("loc", 0.0)
                self.scale = kwargs.get("scale", 1.0)
            elif self.distribution in ["exponential"]:
                self.scale = kwargs.get("scale", 1.0)
            elif self.distribution in ["possion"]:
                self.scale = kwargs.get("lam", 1.0)
            elif self.distribution in ["customized"]:
                self.min = kwargs.get("min", 0.0)
                self.max = kwargs.get("max", 1.0)

    @classmethod
    def create_attr_from_dict(cls, dict):
        """Create attribute from dict

        Args:
            dict (dict): contains attribute information, such as: "name", "owner", "type"
        """
        dict_copy = copy.deepcopy(dict)
        name = dict_copy.pop("name")
        owner = dict_copy.pop("owner")
        type = dict_copy.pop("type")

        # TODO
        assert (owner, type) in ATTRIBUTES_DICT.keys(), ValueError(
            "Unsupproted attribute!"
        )

        AttributeClass = ATTRIBUTES_DICT.get((owner, type))

        return AttributeClass(name, **dict_copy)

    def get_attr_by_id(self, net, id):
        """Get the attr_value of the node/edge with the node/edge id

        Args:
            net: nx.Grapth instance
            id: the id of node/edge, for example: `1` for node and `(1, 2)` for edge


        Return:
            for node: id = 1, so return `G.nodes[1][attr_name]`
            for edge: id = (1, 2), so return `G.edges[(1, 2)][attr_name]`

        """
        if self.owner == "node":
            return net.nodes[id][self.name]
        elif self.owner == "link":
            return net.edges[id][self.name]

    def check_attr(self):
        # TODO
        return True

    def get_attr_size(self, net):
        """Get the size of the attribute of the `net`, usually refers to the number of nodes/links in the network"""
        return net.num_nodes if self.owner == "node" else net.num_edges

    def generate_attr_data(self, net):
        """Generate data with a specified size, data type, and distribution based on the attribute settings

        Args:
            size: the number of nodes/links
            type: attr.type
            distribution: attr.distribution

        Return:
           np.ndarray: [34, 45, 65, 45, ...] for example
        """
        assert self.generative
        # size = net.num_nodes if self.owner == "node" else net.num_links
        size = net.number_of_nodes() if self.owner == "node" else net.number_of_links()

        if self.distribution == "uniform":
            kwargs = {"low": self.low, "high": self.high}
        elif self.distribution == "normal":
            kwargs = {"loc": self.loc, "scale": self.scale}
        elif self.distribution == "exponential":
            kwargs = {"scale": self.scale}
        elif self.distribution == "possion":
            kwargs = {"lam": self.lam}
        elif self.distribution == "customized":
            data = np.random.uniform(0.0, 1.0, size)
            return data * (self.max - self.min) + self.min
        else:
            raise NotImplementedError

        # TODO
        return generate_data_with_distribution(
            size, distribution=self.distribution, dtype=self.dtype, **kwargs
        )

    def to_dict(self):
        """Get a string representation of all the properties of the object

        such as: {'name': 'cpu', 'owner': 'node', 'type': 'resource', 'originator': None, 'generative': False}
        """
        return self.__dict__

    def __repr__(self) -> str:
        """Get a string representation of all the properties of the object

        such as: Attribute(name=cpu, owner=node, type=resource, originator=None, generative=False)
        """
        # info = [f"{key}={self._size_repr(item)}" for key, item in self.__dict__]
        info = [f"{key}={item}" for key, item in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(info)})"


class InfoAttribute(Attribute):

    def __init__(self, name, owner, *args, **kwargs):
        super().__init__(name, owner, "info", *args, **kwargs)


class NodeInfoAttribute(InfoAttribute):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, "node", *args, **kwargs)


class LinkInfoAttribute(InfoAttribute):

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, "link", *args, **kwargs)


class ResourceMethod:
    """
    Methods:
        update: Update(+ or -) resource
        _check_one_element: Determine the v_node >= p_node? v_node <= p_node?, v_node == p_node?
        generate_data: Generate attr data if `attr.generative == True`
    """

    def update(self, v, p, method="+", safe=True):
        """Update(+ or -) resource

        Args:
            self: attr instance
            v: v_node/v_link instance
            p: p_node.p_link instance
            method: The way of updating, '+' or '-'
            safe: Check whether or not there are negative values
        """
        assert self.type in ["resource"]
        assert method.lower() in ["+", "-", "add", "sub"]

        if method.lower() in ["+", "add"]:
            p[self.name] += v[self.name]
        elif method.lower() in ["-", "sub"]:
            if safe:
                assert (
                    v[self.name] <= p[self.name]
                ), f"{self.name}: (v = {v[self.name]}) > (p = {p[self.name]})"
            p[self.name] -= v[self.name]
        else:
            raise NotImplementedError

        return True

    def _check_one_element(self, v_net, p_net, method="le"):
        """Determine whether or not the mapping requirements are met

        Args:
            v: v_node/v_link instance
            p: p_node/p_link instance
            methods: default to "le"

        Returns:
            True/False: Whether or not the mapping requirements are met
            value: The remaining resource after the mapping
        """
        assert method in [">=", "<=", "ge", "le", "eq"]
        if method in [">=", "ge"]:
            return (
                v_net[self.name] >= p_net[self.name],
                v_net[self.name] - p_net[self.name],
            )
        elif method in ["<=", "le"]:
            return (
                v_net[self.name] <= p_net[self.name],
                p_net[self.name] - v_net[self.name],
            )
        elif method in ["==", "eq"]:
            return v_net[self.name] == p_net[self.name], abs(
                v_net[self.name] - p_net[self.name]
            )
        else:
            raise NotImplementedError(f"Used method {method}")

    def generate_data(self, net):
        """Generate attr data if `attr.generative == True`"""
        if self.generative:
            return self.generate_attr_data(net)
        else:
            raise NotImplementedError


class ExtremaMethod:

    def update(self, v_node, p_node, method="+", safe=True):
        return True

    def check(self, v_net, p_net, v_node_id, p_node_id, method="le"):
        return True

    def generate_data(self, network):
        if self.owner == "node":
            originator_attribute = network.node_attrs[self.originator]
        else:
            originator_attribute = network.link_attrs[self.originator]
        attribute_data = originator_attribute.get_data(network)

        return attribute_data


class NodeMethod:

    def set_data(self, net, attr_data):
        """Set the value(list) of a specific node_attr_name to a given network

        Args:
            self: attr instance
            net: G
            attr_data: dict-like
        """
        if not isinstance(attr_data, dict):
            # attr_data = {n: attr_data[i] for i, n in enumerate(net.nodes)}
            attr_data = {n: attr_data[i] for i, n in enumerate(list(net.nodes))}
        nx.set_node_attributes(net, attr_data, self.name)

    def get_data(self, net):
        """Get the value(list) of a specific attr_name in a given network

        Args:
            self: attr instance
            net: G
        """
        attr_data = list(nx.get_node_attributes(net, self.name).values())
        return attr_data


class LinkMethod:

    def set_data(self, net, attr_data):
        """Set the value(list) of a specific link_attr_name to a given network

        Args:
            self: attr instance
            net: G
            attr_data: dict-like
        """
        if not isinstance(attr_data, dict):
            # attr_data = {e: attr_data[i] for i, e in enumerate(net.links)}
            attr_data = {e: attr_data[i] for i, e in enumerate(list(net.links))}
        nx.set_edge_attributes(net, attr_data, self.name)

    def get_data(self, net):
        """Get the value(list) of a specific link_attr_name in a given network

        Args:
            self: attr instance
            net: G
        """
        attr_data = list(nx.get_edge_attributes(net, self.name).values())
        return attr_data

    def get_adj_data(self, net, normalized=False):
        """Get the adj_data of a specific attr_name in a given network

        Args:
            self: attr instance
            net: G
            normalized: default False
        """
        adj_data = nx.attr_sparse_matrix(
            net, edge_attr=self.name, normalized=normalized, rc_order=net.nodes
        ).toarray()
        return adj_data

    def get_aggregation_data(self, ney, aggr="sum", normalized=False):
        """Get the aggregation_data of a specific attr_name in a given network

        Args:
            self: attr instance
            net: G
            aggr: default "sum"
            normalized: default False
        """
        assert aggr in ["sum", "mean", "max"], NotImplementedError
        attr_sparse_matrix = nx.attr_sparse_matrix(
            ney, edge_attr=self.name, normalized=normalized, rc_order=ney.nodes
        ).toarray()
        if aggr == "sum":
            aggregation_data = attr_sparse_matrix.sum(axis=0)
            aggregation_data = np.asarray(aggregation_data)
        elif aggr == "mean":
            aggregation_data = attr_sparse_matrix.mean(axis=0)
            aggregation_data = np.asarray(aggregation_data)
        elif aggr == "max":
            aggregation_data = attr_sparse_matrix.max(axis=0)
        return aggregation_data


class NodeResourceAttribute(Attribute, NodeMethod, ResourceMethod):

    def __init__(self, name, *args, **kwargs) -> None:
        super(NodeResourceAttribute, self).__init__(
            name, "node", "resource", *args, **kwargs
        )

    def check(self, v_node, p_node, method="le"):
        """Determine whether or not the `v_node -> p_node mapping` requirements are met"""
        return super()._check_one_element(v_node, p_node, method)


class NodeExtremaAttribute(Attribute, NodeMethod, ExtremaMethod):
    def __init__(self, name, *args, **kwargs):
        super(NodeExtremaAttribute, self).__init__(
            name, "node", "extrema", *args, **kwargs
        )


class NodePositionAttribute(Attribute, NodeMethod):

    def __init__(self, name="pos", *args, **kwargs) -> None:
        super(NodePositionAttribute, self).__init__(
            name, "node", "position", *args, **kwargs
        )

    def generate_data(self, net):
        """Generate node_position data"""
        if self.generative:
            pos_x = self.generate_attr_data(net)
            pos_y = self.generate_attr_data(net)

            pos_r = self.generate_attr_data(net)
            pos_r = np.clip(pos_r, self.min_r, self.max_r, out=None)

            pos_data = [(x, y, pos_r) for x, y in zip(pos_x, pos_y)]
        elif "pos" in net.nodes[0].keys():
            pos_data = list(nx.get_node_attributes(net, "pos").values())
        else:
            return AttributeError("Please specify how to generate data")

        return pos_data


class LinkResourceAttribute(Attribute, LinkMethod, ResourceMethod):

    def __init__(self, name, *args, **kwargs) -> None:
        super(LinkResourceAttribute, self).__init__(
            name, "link", "resource", *args, **kwargs
        )

    def check(self, v_link, p_link, method="le"):
        """Determine whether or not the `v_link -> p_link mapping` requirements are met"""
        return super()._check_one_element(v_link, p_link, method)

    def update_path(self, v_link, p_net, path, method="+", safe=True):
        """Update link resources while mapping v_link -> p_link"""
        assert self.type in ["resource"]
        assert method.lower() in ["+", "-", "add", "sub"], NotImplementedError
        assert len(path) > 1

        links_list = path_to_links(path)

        for link in links_list:
            self.update(v_link, p_net.links[link], method, safe)

        return True


class LinkExtremaAttribute(Attribute, LinkMethod, ExtremaMethod):

    def __init__(self, name, *args, **kwargs) -> None:
        super(LinkExtremaAttribute, self).__init__(
            name, "link", "extrema", *args, **kwargs
        )

    def update_path(self, v_link, p_net, method="+", safe=True):
        """Extrema can not update path, so rewrite"""
        return True


ATTRIBUTES_DICT = {
    # Resource
    ("node", "resource"): NodeResourceAttribute,
    ("node", "extrema"): NodeExtremaAttribute,
    ("link", "resource"): LinkResourceAttribute,
    ("link", "extrema"): LinkExtremaAttribute,
    # Fixed
    ("node", "position"): NodePositionAttribute,
}

if __name__ == "__main__":
    pass
