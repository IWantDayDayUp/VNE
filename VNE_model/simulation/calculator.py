from typing import Union
import numpy as np
import pandas as pd

from ..network import VirtualNetwork, PhysicalNetwork
from ..network import Attribute
from .solution import Solution


def create_attrs_from_setting(attrs_setting):
    attrs = {
        attr_dict["name"]: Attribute.create_attr_from_dict(attr_dict)
        for attr_dict in attrs_setting
    }
    return attrs


class Calculator:

    def __init__(
        self, node_attrs_setting: list = [], link_attrs_setting: list = [], **kwargs
    ) -> None:
        self.node_resource_unit_price = kwargs.get("node_resource_unit_price", 1.0)
        self.link_resource_unit_price = kwargs.get("link_resource_unit_price", 1.0)
        self.revenue_service_time_weight = kwargs.get(
            "revenue_service_time_weight", 0.001
        )
        self.revenue_start_price_weight = kwargs.get("revenue_start_price_weight", 1.0)

        self.all_node_attrs = list(
            create_attrs_from_setting(node_attrs_setting).values()
        )
        self.all_link_attrs = list(
            create_attrs_from_setting(link_attrs_setting).values()
        )
        self.node_resource_attrs = [
            n_attr for n_attr in self.all_node_attrs if n_attr.type == "resource"
        ]
        self.link_resource_attrs = [
            l_attr for l_attr in self.all_link_attrs if l_attr.type == "resource"
        ]

        self.num_node_resource_attrs = len(self.node_resource_attrs)
        self.num_link_resource_attrs = len(self.link_resource_attrs)

    def count_partial_solution(self, v_net: VirtualNetwork, solution: Solution) -> dict:
        """
        Count the revenue and cost of a partial solution.

        Args:
            v_net (VirtualNetwork): Virtual network
            solution (Solution): Partial solution

        Returns:
            dict: The information of partial solution with revenue and cost
        """

        # partial node_revenue
        v_net_node_revenue = self.clac_partial_node_revenue(
            v_net=v_net, solution=solution
        )
        # partial link_revenue and link_cost
        v_net_link_revenue, v_net_link_cost = self.clac_partial_link_revenue(
            v_net=v_net, solution=solution
        )

        solution["v_net_node_revenue"] = (
            v_net_node_revenue / self.num_node_resource_attrs
        )
        solution["v_net_link_revenue"] = v_net_link_revenue

        solution["v_net_revenue"] = v_net_node_revenue + v_net_link_revenue
        solution["v_net_link_cost"] = v_net_link_cost
        solution["v_net_path_cost"] = v_net_link_cost - v_net_link_revenue
        solution["v_net_node_cost"] = v_net_node_revenue / self.num_node_resource_attrs
        solution["v_net_cost"] = (
            solution["v_net_node_cost"] + solution["v_net_link_cost"]
        )
        solution["v_net_r2c_ratio"] = (
            solution["v_net_revenue"] / solution["v_net_cost"]
            if solution["v_net_cost"] != 0
            else 0
        )

        return solution.to_dict()

    def clac_partial_node_revenue(self, v_net: VirtualNetwork, solution: Solution):
        """
        Calculate the node revenue of a partial solution.
        """
        partial_node_revenue = 0
        for nid in solution["node_slots"].keys():
            partial_node_revenue += sum(
                [v_net.nodes[nid][n_attr.name] for n_attr in self.node_resource_attrs]
            )

        return partial_node_revenue

    def clac_partial_link_revenue(self, v_net: VirtualNetwork, solution: Solution):
        """
        Calculate the link revenue and cost of a partial solution.
        """
        partial_link_revenue = 0
        partial_link_cost = 0
        for v_link, p_links in solution["link_paths"].items():
            one_revenue = sum(
                [
                    v_net.links[v_link][l_attr.name]
                    for l_attr in self.link_resource_attrs
                ]
            )
            partial_link_revenue += one_revenue
            for p_link in p_links:
                one_cost = sum(
                    [
                        solution["link_paths_info"][(v_link, p_link)][l_attr.name]
                        for l_attr in self.link_resource_attrs
                    ]
                )
            partial_link_cost += one_cost

        return partial_link_revenue, partial_link_cost

    def count_solution(self, v_net: VirtualNetwork, solution: Solution) -> dict:
        """
        Count the revenue and cost of a solution.

        Args:
            v_net (VirtualNetwork): Virtual network
            solution (Solution): Solution

        Returns:
            dict: The information of partial solution with revenue and cost
        """
        solution["num_placed_nodes"] = len(solution.node_slots)
        solution["num_routed_links"] = len(solution.link_paths)
        solution["v_net_node_demand"] = (
            self.calc_sum_vnet_node_demand(v_net) / self.num_node_resource_attrs
        )
        solution["v_net_link_demand"] = self.calc_sum_vnet_link_demand(v_net)
        solution["v_net_demand"] = (
            solution["v_net_node_demand"] + solution["v_net_demand"]
        )

        # success
        if solution["result"]:
            solution["place_result"] = True
            solution["route_result"] = True
            solution["early_rejection"] = False
            solution["v_net_node_revenue"] = solution["v_net_node_demand"]
            solution["v_net_link_revenue"] = solution["v_net_link_demand"]
            solution["v_net_node_cost"] = solution["v_net_node_revenue"]
            solution["v_net_link_cost"] = self.calc_v_net_link_cost(v_net, solution)
            solution["v_net_path_cost"] = (
                solution["v_net_link_cost"] - solution["v_net_link_revenue"]
            )
            solution["v_net_revenue"] = (
                solution["v_net_node_revenue"] + solution["v_net_link_revenue"]
            )
            solution["v_net_cost"] = (
                solution["v_net_revenue"] + solution["v_net_path_cost"]
            )
            solution["v_net_r2c_ratio"] = (
                solution["v_net_revenue"] / solution["v_net_cost"]
                if solution["v_net_cost"] != 0
                else 0
            )
        else:
            solution["v_net_node_revenue"] = 0
            solution["v_net_link_revenue"] = 0
            solution["v_net_revenue"] = 0
            solution["v_net_path_cost"] = 0
            solution["v_net_cost"] = 0
            solution["v_net_r2c_ratio"] = 0
            # solution['node_slots'] = {}
            # solution['link_paths'] = {}
        solution["v_net_time_revenue"] = solution["v_net_revenue"] * v_net.graph["lifetime"]
        solution["v_net_time_cost"] = solution["v_net_cost"] * v_net.graph["lifetime"]
        solution["v_net_time_rc_ratio"] = solution["v_net_r2c_ratio"] * v_net.graph["lifetime"]

        return solution.to_dict()

    def calc_sum_pnet_node_resource(self, p_net: PhysicalNetwork):
        """
        Calculate the sum of `node` resource of p_net.
        """
        n = np.array(p_net.get_node_attrs_data(self.node_resource_attrs)).sum()

        return n

    def calc_sum_pnet_link_resource(self, p_net: PhysicalNetwork):
        """
        Calculate the sum of `link` resource of p_net.
        """
        l = np.array(p_net.get_link_attrs_data(self.link_resource_attrs)).sum()

        return l

    def calc_sum_pnet_resource(
        self, p_net: PhysicalNetwork, node: bool = True, link: bool = True
    ):
        """
        Calculate the sum of network resource.

        Args:
            p_net (PhysicalNetwork): Physical Network
            node (bool, optional): Whether to calculate the sum of node resource. Defaults to True.
            link (bool, optional): Whether to calculate the sum of link resource. Defaults to True.

        Returns:
            float: The sum of network resource.
        """
        return (self.calc_sum_pnet_node_resource(p_net=p_net) if node else 0) + (
            self.calc_sum_pnet_link_resource(p_net=p_net) if link else 0
        )

    def calc_sum_vnet_node_demand(self, v_net: VirtualNetwork):
        """
        Calculate the sum of `node` resource demand of v_net.
        """
        n = np.array(v_net.get_node_attrs_data(self.node_resource_attrs)).sum()

        return n

    def calc_sum_vnet_link_demand(self, v_net: VirtualNetwork):
        """
        Calculate the sum of `link` resource demand of v_net.
        """
        l = np.array(v_net.get_link_attrs_data(self.link_resource_attrs)).sum()

        return l

    def calc_sum_vnet_resource_demand(
        self, v_net: VirtualNetwork, node: bool = True, link: bool = True
    ):
        """
        Calculate the sum of network resource demand.

        Args:
            v_net (VirtualNetwork): Virtual Network
            node (bool, optional): Whether to calculate the sum of node resource demand. Defaults to True.
            link (bool, optional): Whether to calculate the sum of link resource demand. Defaults to True.

        Returns:
            float: The sum of network resource demand.
        """
        return (self.calc_sum_vnet_node_demand(v_net=v_net) if node else 0) + (
            self.calc_sum_vnet_link_demand(v_net=v_net) if link else 0
        )

    def calc_v_net_node_cost(self, v_net: VirtualNetwork, solution: Solution):
        """
        Calculate the deployment cost of v_node in the v_net, which means the sum of v_node resource demand.
        """
        return self.calc_sum_vnet_node_demand(v_net=v_net)

    def calc_v_net_link_cost(self, v_net: VirtualNetwork, solution: Solution):
        """
        Calculate the deployment cost of v_link in the v_net according to `link paths`.
        """
        sum_link_cost = 0
        for v_link, p_links in solution["link_paths"].items():
            for p_link in p_links:
                for l_attr in self.link_resource_attrs:
                    sum_link_cost += solution["link_paths_info"][(v_link, p_link)][
                        l_attr.name
                    ]

        return sum_link_cost

    def calc_v_net_cost(self, v_net: VirtualNetwork, solution: Solution):
        """
        Calculate the deployment cost of v_net: cost(v_node) + cost(v_link).
        """
        return self.calc_v_net_node_cost(
            v_net=v_net, solution=solution
        ) + self.calc_v_net_link_cost(v_net=v_net, solution=solution)

    def calc_v_net_revenue(self, v_net: VirtualNetwork):
        """Calculate the deployment revenue of v_net, which means the sum of resource demand of v_node and v_link."""
        return self.calc_sum_vnet_resource_demand(v_net=v_net)

    def summary_records(self, records: Union[list, pd.DataFrame]):
        """
        Summarize the records.

        Args:
            records (Union[list, pd.DataFrame]): The records to be summarized.

        Returns:
            dict: The summary information.
        """
        if isinstance(records, list):
            records = pd.DataFrame(records)
        elif isinstance(records, pd.DataFrame):
            pass
        else:
            raise TypeError

        summary_info = {}

        summary_info["acceptance_rate"] = (
            records.iloc[-1]["success_count"] / records.iloc[-1]["v_net_count"]
        )
        summary_info["avg_r2c_ratio"] = records.loc[
            records["event_type"] == 1, "v_net_r2c_ratio"
        ].mean()
        summary_info["long_term_time_r2c_ratio"] = (
            records.iloc[-1]["total_time_revenue"] / records.iloc[-1]["total_time_cost"]
        )
        summary_info["long_term_avg_time_revenue"] = (
            records.iloc[-1]["total_time_revenue"]
            / records.iloc[-1]["v_net_arrival_time"]
        )

        # ac rate
        summary_info["success_count"] = records.iloc[-1]["success_count"]
        summary_info["early_rejection_count"] = (
            (records["event_type"] == 1) & (records["early_rejection"] == True)
        ).sum()
        summary_info["place_failure_count"] = (
            (records["event_type"] == 1) & (records["place_result"] == False)
        ).sum()
        summary_info["route_failure_count"] = (
            (records["event_type"] == 1) & (records["route_result"] == False)
        ).sum()

        # rc ratio
        summary_info["total_cost"] = records.iloc[-1]["total_cost"]
        summary_info["total_revenue"] = records.iloc[-1]["total_revenue"]
        summary_info["total_time_revenue"] = records.iloc[-1]["total_time_revenue"]
        summary_info["total_time_cost"] = records.iloc[-1]["total_time_cost"]
        summary_info["long_term_r2c_ratio"] = (
            summary_info["total_revenue"] / summary_info["total_cost"]
        )

        # revenue / cost
        summary_info["total_simulation_time"] = records.iloc[-1]["v_net_arrival_time"]
        summary_info["long_term_avg_revenue"] = (
            summary_info["total_revenue"] / summary_info["total_simulation_time"]
        )
        summary_info["long_term_avg_cost"] = (
            summary_info["total_cost"] / summary_info["total_simulation_time"]
        )
        summary_info["long_term_weighted_avg_time_revenue"] = (
            self.revenue_service_time_weight
            * summary_info["long_term_avg_time_revenue"]
            + self.revenue_start_price_weight * summary_info["long_term_avg_revenue"]
        )

        # state
        summary_info["min_p_net_available_resource"] = records.loc[
            :, "p_net_available_resource"
        ].min()
        summary_info["min_p_net_node_available_resource"] = records.loc[
            :, "p_net_node_available_resource"
        ].min()
        summary_info["min_p_net_link_available_resource"] = records.loc[
            :, "p_net_link_available_resource"
        ].min()
        summary_info["max_inservice_count"] = records.loc[:, "inservice_count"].max()

        # rl reward
        if "v_net_reward" in records.columns:
            summary_info["avg_reward"] = records.loc[
                records["event_type"] == 1, "v_net_reward"
            ].mean()
        else:
            summary_info["avg_reward"] = 0

        return summary_info

    @classmethod
    def summary_csv(cls, fpath: str):
        """
        Summary the records in csv file.

        Args:
            fpath (str): The path of csv file.

        Returns:
            dict: The summary information.
        """
        records = pd.read_csv(fpath, header=0)
        summary_info = cls.summary_records(records)

        return summary_info
