import numpy as np

from ..network.network import Network


class LinkRank(object):
    """LinkRank class is an abstract class for link ranking algorithms."""

    def __init__(self, **kwargs):
        super(LinkRank, self).__init__()

    def __call__(self, network, sort=True):
        return self.rank(network, sort=sort)

    @staticmethod
    def rank(self, network: Network, sort: bool = True) -> dict:
        """
        Rank links in the network.

        Args:
            network (Network): Network object.
            sort (bool, optional): Sort the ranking result. Defaults to True.

        Returns:
            dict: A dict of link ranking result.
        """
        pass

    @staticmethod
    def to_dict(link_rank_vector: list, network, sort: bool = True) -> dict:
        link_rank_vector_dict = {
            e: link_rank_vector[i] for i, e in enumerate(network.links)
        }
        if sort:
            link_rank_vector_dict = sorted(
                link_rank_vector_dict.items(), reverse=True, key=lambda x: x[1]
            )
            link_rank_vector_dict = {i: v for i, v in link_rank_vector_dict}

        return link_rank_vector_dict
