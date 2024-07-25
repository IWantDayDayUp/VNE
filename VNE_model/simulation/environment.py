import os
import copy
import time
import numpy as np

from pprint import pprint


class Environment:
    """A general environment for various solvers."""

    def __init__(
        self, p_net, v_net_sim, controller, recorder, counter, **kwargs
    ) -> None:
        """init"""

        self.p_net = p_net
        self.init_p_net = copy.deepcopy(p_net)
        self.v_net_sim = v_net_sim
        self.controller = controller
        self.recorder = recorder
        self.counter = counter

        self.num_v_nets = self.v_net_sim.num_v_nets
        self.num_events = self.num_v_nets * 2

        self.verbose = kwargs.get("verbose", 1)

        self.p_net_dataset_dir = kwargs.get(
            "p_net_dataset_dir", "unknown_p_net_dataset_dir"
        )
        self.v_nets_dataset_dir = kwargs.get(
            "v_nets_dataset_dir", "unknown_v_nets_dataset_dir"
        )

        self.solver_name = kwargs.get("solver_name", "unknown_solver")
        self.seed = kwargs.get("seed", None)
        self.if_save_records = kwargs.get("if_save_records", True)
        self.summary_file_name = kwargs.get("summary_file_name", "global_summary.csv")

        self.node_ranking_method = kwargs.get("node_ranking_method", "order")
        self.link_ranking_method = kwargs.get("link_ranking_method", "order")

        self.matching_method = kwargs.get("matching_method", "greedy")
        self.shortest_method = kwargs.get("shortest_method", "k_shortest")
        self.k_shortest = kwargs.get("k_shortest", 10)

        self.extra_summary_info = {}
        self.extra_record_info = {}

        self.r2c_ratio_threshold = kwargs.get("r2c_ratio_threshold", 0.0)
        self.vn_size_threshold = kwargs.get("vn_size_threshold", 10000)

        self.set_sim_info_to_object(kwargs, self)

    def set_sim_info_to_object(config_dict: dict, obj):
        if not isinstance(config_dict, dict):
            config_dict = vars(config_dict)
        for key in [
            "p_net_setting_num_nodes",
            "p_net_setting_num_node_attrs",
            "p_net_setting_num_link_attrs",
            "p_net_setting_num_node_resource_attrs",
            "p_net_setting_num_link_resource_attrs",
            "p_net_setting_num_node_extrema_attrs",
        ]:
            setattr(obj, key, config_dict[key]) if not hasattr(obj, key) else None
        for key in [
            "v_sim_setting_num_node_attrs",
            "v_sim_setting_num_link_attrs",
            "v_sim_setting_num_node_resource_attrs",
            "v_sim_setting_num_link_resource_attrs",
        ]:
            setattr(obj, key, config_dict[key]) if not hasattr(obj, key) else None

class SolutionStepEnvironment(Environment):
    pass