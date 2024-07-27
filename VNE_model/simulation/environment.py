import os
import copy
import time
import numpy as np

from pprint import pprint

from ..network import PhysicalNetwork, VirtualNetworkRequestSimulator
from .solution import Solution
from .recorder import Recorder
from .controller import Controller
from .calculator import Calculator
from ..network.utils import (
    get_p_net_dataset_dir_from_setting,
    get_v_nets_dataset_dir_from_setting,
)


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
        "v_net_setting_num_node_attrs",
        "v_net_setting_num_link_attrs",
        "v_net_setting_num_node_resource_attrs",
        "v_net_setting_num_link_resource_attrs",
    ]:
        setattr(obj, key, config_dict[key]) if not hasattr(obj, key) else None


class Environment:
    """A general environment for various solvers."""

    def __init__(
        self,
        p_net: PhysicalNetwork,
        v_net_sim: VirtualNetworkRequestSimulator,
        controller: Controller,
        recorder: Recorder,
        calculator: Calculator,
        **kwargs,
    ) -> None:
        """init"""

        self.p_net = p_net
        self.init_p_net = copy.deepcopy(p_net)
        self.v_net_sim = v_net_sim
        self.controller = controller
        self.recorder = recorder
        self.calculator = calculator

        self.num_v_nets = self.v_net_sim.num_v_nets
        self.num_events = self.num_v_nets * 2

        self.verbose = kwargs.get("verbose", 1)

        self.p_net_dataset_dir = kwargs.get(
            "p_net_dataset_dir", "unknown_p_net_dataset_dir"
        )
        self.v_nets_dataset_dir = kwargs.get(
            "v_nets_dataset_dir", "unknown_v_nets_dataset_dir"
        )

        self.renew_v_net_simulator = kwargs.get("renew_v_net_simulator", False)

        self.solver_name = kwargs.get("solver_name", "unknown_solver")
        self.run_id = kwargs.get("run_id", "unknown_device-unknown_run_time")
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

        set_sim_info_to_object(kwargs, self)

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

    def ready(self, event_id: int = 0):
        """
        Prepare for the given event.

        Args:
            event_id: the id of the event to be processed.
        """
        self.curr_event = self.v_net_sim.events[event_id]
        self.num_processed_v_nets += 1
        self.v_net = self.v_net_sim.v_nets[int(self.curr_event["v_net_id"])]
        self.solution = Solution(self.v_net)
        self.p_net_backup = (
            copy.deepcopy(self.p_net) if self.curr_event["type"] == 1 else None
        )
        self.recorder.update_state(
            {
                "event_id": self.curr_event["id"],
                "event_type": self.curr_event["type"],
                "event_time": self.curr_event["time"],
            }
        )
        # self.recorder.ready(self.curr_event)
        if self.verbose >= 2:
            print(f"\nEvent: id={event_id}, type={self.curr_event['type']}")
            print(f"{'-' * 30}")

    def reset(self, seed: int = None):
        """
        Reset the environment.

        Args:
            seed: the seed for the random number generator. If None, use the seed in the config.
        """
        seed = seed if seed is not None else self.seed

        self.p_net = copy.deepcopy(self.init_p_net)
        self.recorder.reset()
        self.recorder.count_init_p_net_info(self.p_net)
        if self.recorder.if_temp_save_records and self.verbose >= 1:
            print(f"temp save record in {self.recorder.temp_save_path}\n")

        self.v_nets_dataset_dir = get_v_nets_dataset_dir_from_setting(
            self.v_net_sim.v_sim_setting
        )
        if self.renew_v_net_simulator:
            self.v_net_sim.generate_sim(v_nets=True, events=True, seed=seed)
            (
                print(f"Generate virtual networks with seed {seed}")
                if self.verbose >= 1
                else None
            )
        elif os.path.exists(self.v_nets_dataset_dir):
            self.v_net_sim = self.v_net_sim.load_dataset(self.v_nets_dataset_dir)
            (
                print(f"Load virtual networks from {self.v_nets_dataset_dir}")
                if self.verbose >= 1
                else None
            )
        else:
            self.v_net_sim.generate_sim(v_nets=True, events=True, seed=seed)
            (
                print(f"\n*** Generate virtual networks with seed {seed}")
                if self.verbose >= 1
                else None
            )
        self.cumulative_reward = 0
        self.num_processed_v_nets = 0
        self.start_run_time = time.time()
        self.ready(event_id=0)

        return self.get_observation()

    def step(self, action):
        """
        Take an action and return the next observation, reward, done, and info.

        Args:
            action: the action to be taken.

        Returns:
            observation: the observation after taking the action.
            reward: the reward after taking the action.
            done: whether the episode is done.
            info: the extra information.
        """
        return NotImplementedError

    def compute_reward(self) -> float:
        """Compute the reward for the current Virtual Network."""
        return NotImplementedError

    def get_observation(self) -> dict:
        """Get the observation for the current Virtual Network."""
        return NotImplementedError

    def render(self, mode="human") -> None:
        """
        Render the environment.

        Args:
            mode: the mode to render the environment.
        """
        return NotImplementedError

    def release(self) -> dict:
        """
        Release the current Virtual Network when it leaves the system.
        """
        solution = self.recorder.get_record(v_net_id=self.v_net.graph["id"])
        self.controller.release(self.v_net, self.p_net, solution)
        self.solution["description"] = "Leave Event"
        record = self.count_and_add_record()

        return record

    def get_failure_reason(self, solution: Solution) -> str:
        """
        Get the reason of failure, which is used to rollback the state of the physical network.

        Args:
            solution (Solution): the solution of the current Virtual Network.

        Returns:
            reason (str): the reason of failure.
        """
        if solution["early_rejection"]:
            return "reject"
        if not solution["place_result"]:
            return "place"
        elif not solution["route_result"]:
            return "route"
        else:
            return "unknown"

    def rollback_for_failure(self, reason="place") -> None:
        """
        Rollback the state of the physical network for the failure of the current Virtual Network.

        Args:
            reason (str): the reason of failure.
        """
        # self.solution.reset()
        self.p_net = copy.deepcopy(self.p_net_backup)
        if reason in ["unknown", -1]:
            self.solution["description"] = "Unknown Reason"
        if reason in ["reject", 0]:
            self.solution["description"] = "Early Rejection"
            self.solution["early_rejection"] = True
        elif reason in ["place", 1]:
            self.solution["description"] = "Place Failure"
            self.solution["place_result"] = False
        elif reason in ["route", 2]:
            self.solution["description"] = "Route Failure"
            self.solution["route_result"] = False
        else:
            return NotImplementedError

    def transit_obs(self) -> bool:
        """
        Automatically Transit the observation to the next event until the next enter event comes or episode has done.

        Returns:
            done (bool): whether the episode is done.
        """

        # Leave events transition
        while True:
            next_event_id = int(self.curr_event["id"] + 1)
            # episode finished
            if next_event_id > self.num_events - 1:
                summary_info = self.summary_records()
                return True
            self.ready(next_event_id)
            if self.curr_event["type"] == 0:
                record = self.release()
            else:
                return False

    @property
    def selected_p_net_nodes(self) -> list:
        """Get the already selected physical nodes for the current Virtual Network."""
        return list(self.solution["node_slots"].values())

    @property
    def placed_v_net_nodes(self) -> list:
        """Get the already placed virtual nodes for the current Virtual Network."""
        return list(self.solution["node_slots"].keys())

    @property
    def num_placed_v_net_nodes(self) -> int:
        """Get the number of already placed virtual nodes for the current Virtual Network."""
        return len(self.solution["node_slots"].keys())

    ### recorder ###
    def add_record(self, record: dict, extra_info: dict = {}) -> dict:
        """
        Add extra information to the record and add the record to the recorder.

        Args:
            record (dict): the record to be added.
            extra_info (dict): the extra information to be added.

        Returns:
            record (dict): the record with extra information.
        """
        self.extra_record_info.update(extra_info)
        record = self.recorder.add_record(record, self.extra_record_info)
        if self.verbose >= 2:
            self.display_record(record, extra_items=list(extra_info.keys()))

        return record

    def count_and_add_record(self, extra_info: dict = {}) -> dict:
        """
        Count the record and add the record to the recorder.

        Args:
            extra_info (dict): the extra information to be added.
        """
        record = self.recorder.count(self.v_net, self.p_net, self.solution)
        record = self.add_record(record, extra_info)

        return record

    def display_record(
        self,
        record: dict,
        display_items: list = [
            "result",
            "v_net_id",
            "v_net_cost",
            "v_net_revenue",
            "p_net_available_resource",
            "total_revenue",
            "total_cost",
            "description",
        ],
        extra_items: list = [],
    ) -> None:
        """
        Display the record, including the default display items and extra display items.

        Args:
            record (dict): the record to be displayed.
            display_items (list): the default display items.
            extra_items (list): the extra display items.
        """
        display_items = display_items + extra_items
        print("".join([f"{k}: {v}\n" for k, v in record.items() if k in display_items]))

    def summary_records(
        self, extra_summary_info={}, summary_file_name=None, record_file_name=None
    ) -> None:
        """
        Summarize the records and save the summary information and records to the file.

        Args:
            extra_summary_info (dict): the extra summary information to be added.
            summary_file_name (str): the name of the summary file.
            record_file_name (str): the name of the record file.
        """
        start_run_time = time.strftime(
            "%Y%m%dT%H%M%S", time.localtime(self.start_run_time)
        )
        if summary_file_name is None:
            summary_file_name = self.summary_file_name
        if record_file_name is None:
            record_file_name = f"{self.solver_name}-{self.run_id}-{start_run_time}.csv"
        summary_info = self.recorder.summary_records(self.recorder.memory)
        end_run_time = time.time()
        clock_running_time = end_run_time - self.start_run_time
        run_info_dict = {
            "solver_name": self.solver_name,
            "seed": self.seed,
            "p_net_dataset_dir": self.p_net_dataset_dir,
            "v_nets_dataset_dir": self.v_nets_dataset_dir,
            "run_id": self.run_id,
            "start_run_time": start_run_time,
            "clock_running_time": clock_running_time,
        }
        for k, v in extra_summary_info.items():
            run_info_dict[k] = v
        info = {**summary_info, **run_info_dict}

        if self.if_save_records:
            record_path = self.recorder.save_records(record_file_name)
            summary_path = self.recorder.save_summary(info, summary_file_name)

        if self.verbose >= 1:
            pprint(info)
            if self.if_save_records:
                print(f"save records to {record_path}")
                print(f"save summary to {summary_path}")
        return info


class SolutionStepEnvironment(Environment):
    pass