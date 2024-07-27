import abc
import numpy as np
import networkx as nx

from typing import Union

from ..network.network import Network


class NodeRank(object):
    """Abstract class for node ranking."""

    def __init__(self, **kwargs):
        __metaclass__ = abc.ABCMeta
        super(NodeRank, self).__init__()

    @staticmethod
    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """
        Rank nodes in the network.

        Args:
            network (Network): Network object.
            sort (bool, optional): Sort the ranking result. Defaults to True.

        Returns:
            Union[list, dict]: A list or dict of node ranking result.
        """
        raise NotImplementedError

    def __call__(self, network, sort=True):
        return self.rank(network, sort=sort)

    @staticmethod
    def to_dict(network: Network, node_rank: list, sort: bool = True) -> dict:
        """
        Convert node ranking result to dict.

        Args:
            network (Network): Network object.
            node_rank (list): Node ranking result.
            sort (bool, optional): Sort the ranking result. Defaults to True.

        Returns:
            dict: A dict of node ranking result.
        """
        assert network.num_nodes == len(node_rank)
        node_rank = {node_id: node_rank[i] for i, node_id in enumerate(network.nodes)}
        if sort:
            node_rank = sorted(node_rank.items(), reverse=True, key=lambda x: x[1])
            node_rank = {i: v for i, v in node_rank}

        return node_rank


class OrderNodeRank(NodeRank):
    """
    Node Ranking Strategy with the default order occurring in the network.
    """

    def __init__(self, **kwargs):
        super(OrderNodeRank, self).__init__(**kwargs)

    def rank(self, network: Network, sort: bool = True) -> Union[list, dict]:
        """Rank nodes with the default order occurring in the network."""
        rank_values = 1 / len(network.nodes)
        node_ranking = {node_id: rank_values for node_id in range(len(network.nodes))}
        return node_ranking
