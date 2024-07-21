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

    def __init__(self, name, owner, type, *args, **kwargs) -> None:
        self.name = name
        self.owner = owner
        self.type = type

        # for extrema
        if type == "extrema":
            self.originator = kwargs.get("originator")

        # for generative
        self.generative = kwargs.get("generative", False)
        assert self.generative in [True, False]

        if self.generative:
            self.distribution = kwargs.get("distribution", "normal")
            self.dtype = kwargs.get("dtype", "float")
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
    def load_from_dict(cls, dict):
        dict_copy = copy.deepcopy(dict)
        name = dict_copy.pop("name")
        owner = dict_copy.pop("owner")
        type = dict_copy.pop("type")
        assert (owner, type) in ATTRIBUTES_DICT.keys(), ValueError(
            "Unsupproted attribute!"
        )

        AttributeClass = ATTRIBUTES_DICT.get((owner, type))

        return AttributeClass(name, **dict_copy)

    def get_attr_by_name(self, net, id):
        """Get the specified attribute of the node or link with the specified ID of the network"""
        if self.owner == "node":
            return net.nodes[id][self.name]
        elif self.owner == "link":
            return net.links[id][self.name]

    def check(self):
        # TODO
        return True

    def get_size(self, net):
        return net.num_nodes if self.owner == "node" else net.num_links

    def _generate_data_with_dist(self, net):
        assert self.generative
        size = net.num_nodes if self.owner == "node" else net.num_links

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
        return self.__dict__

    def __repr__(self) -> str:
        info = [f"{key}={self._size_repr(item)}" for key, item in self.__dict__]
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

    def update(self, v, p, method="+", safe=True):
        """Update(+ or -) resource"""
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

    def _check_one_element(self, v_node, p_node, method="le"):
        """Determine the v_node >= p_node? v_node <= p_node?, v_node == p_node?"""
        assert method in [">=", "<=", "ge", "le", "eq"]
        if method in [">=", "ge"]:
            return (
                v_node[self.name] >= p_node[self.name],
                v_node[self.name] - p_node[self.name],
            )
        elif method in ["<=", "le"]:
            return (
                v_node[self.name] <= p_node[self.name],
                p_node[self.name] - v_node[self.name],
            )
        elif method in ["==", "eq"]:
            return v_node[self.name] == p_node[self.name], abs(
                v_node[self.name] - p_node[self.name]
            )
        else:
            raise NotImplementedError(f"Used method {method}")

    def generate_data(self, net):
        if self.generative:
            return self._generate_data_with_dist(net)
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
        if not isinstance(attr_data, dict):
            attr_data = {n: attr_data[i] for i, n in enumerate(net.nodes)}
        nx.set_node_attributes(net, attr_data, self.name)

    def get_data(self, net):
        attr_data = list(nx.get_node_attributes(net, self.name).values())
        return attr_data


class LinkMethod:

    def set_data(self, net, attr_data):
        if not isinstance(attr_data, dict):
            attr_data = {e: attr_data[i] for i, e in enumerate(net.links)}
        nx.set_edge_attributes(net, attr_data, self.name)

    def get_data(self, net):
        attr_data = list(nx.get_edge_attributes(net, self.name).values())
        return attr_data

    def get_adj_data(self, net, normalized=False):
        adj_data = nx.attr_sparse_matrix(
            net, edge_attr=self.name, normalized=normalized, rc_order=net.nodes
        ).toarray()
        return adj_data

    def get_aggregation_data(self, ney, aggr="sum", normalized=False):
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

    def __init__(self, name, owner, type, *args, **kwargs) -> None:
        super(NodeResourceAttribute, self).__init__(
            name, "node", "resource", *args, **kwargs
        )

    def check(self, v_node, p_node, method="le"):
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
        if self.generative:
            pos_x = self._generate_data_with_dist(net)
            pos_y = self._generate_data_with_dist(net)

            pos_r = self._generate_data_with_dist(net)
            pos_r = np.clip(pos_r, self.min_r, self.max_r, out=None)

            pos_data = [(x, y, pos_r) for x, y in zip(pos_x, pos_y)]
        elif "pos" in net.nodes[0].keys():
            pos_data = list(nx.get_node_attributes(net, "pos").values())
        else:
            return AttributeError("Please specify how to generate data")

        return pos_data


class LinkResourceAttribute(Attribute, LinkMethod, ResourceMethod):

    def __init__(self, name, owner, type, *args, **kwargs) -> None:
        super(LinkResourceAttribute, self).__init__(
            name, "link", "resource", *args, **kwargs
        )

    def check(self, v_link, p_link, method="le"):
        return super()._check_one_element(v_link, p_link, method)

    def update_path(self, v_link, p_net, path, method="+", safe=True):
        assert self.type in ["resource"]
        assert method.lower() in ["+", "-", "add", "sub"], NotImplementedError
        assert len(path) > 1

        # TODO
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
