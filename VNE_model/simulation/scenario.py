import os
import tqdm

from .controller import Controller
from .calculator import Calculator
from .recorder import Recorder
from .solution import Solution

from ..network.physical_network import PhysicalNetwork
from ..network.virtual_network_sim import VirtualNetworkRequestSimulator
from ..network.utils import get_p_net_dataset_dir_from_setting


class Scenario:

    def __init__(self, env, solver, config) -> None:
        self.env = env
        self.solver = solver
        self.config = config

        self.verbose = config.verbose

    @classmethod
    def from_config(cls, Env, Solver, config):
        """Create scenario from setting"""
        calculator = Calculator(
            config.v_net_setting["node_attrs_setting"],
            config.v_net_setting["link_attrs_setting"],
            **vars(config),
        )
        controller = Controller(
            config.v_net_setting["node_attrs_setting"],
            config.v_net_setting["link_attrs_setting"],
            **vars(config),
        )
        recorder = Recorder(calculator, **vars(config))

        # Create/Load p_net
        config.p_net_dataset_dir = get_p_net_dataset_dir_from_setting(
            config.p_net_setting
        )
        # print(config.p_net_dataset_dir)
        if os.path.exists(config.p_net_dataset_dir):
            p_net = PhysicalNetwork.load_from_setting(config.p_net_dataset_dir)
            (
                print(f"Load Physical Network from {config.p_net_dataset_dir}")
                if config.verbose >= 1
                else None
            )
        else:
            p_net = PhysicalNetwork.create_from_setting(config.p_net_setting)
            print(f"*** Generate Physical Network from setting")

        # Create v_net simulator
        v_net_simulator = VirtualNetworkRequestSimulator.from_setting(
            config.v_net_setting
        )
        print(f"Create VNR Simulator from setting") if config.verbose >= 1 else None

        # create env and solver
        env = Env(
            p_net, v_net_simulator, controller, recorder, calculator, **vars(config)
        )
        solver = Solver(controller, recorder, calculator, **vars(config))

        # Create scenario
        scenario = cls(env, solver, config)
        if config.verbose >= 2:
            print(config)
        if config.if_save_config:
            config.save_config()

        return scenario

    def ready(self):
        pass


class BasicScenario(Scenario):
    """
    `Basic` means `Time passes by unit(timewindow = 1)`
    """

    def __init__(self, env, solver, config):
        super(BasicScenario, self).__init__(env, solver, config)

    def run(self):
        self.ready()

        for epoch_id in range(self.config.num_epochs):
            print(f"\nEpoch {epoch_id}") if self.verbose >= 2 else None

            self.env.epoch_id = epoch_id
            self.solver.epoch_id = epoch_id

            state = self.env.reset()

            pbar = (
                tqdm.tqdm(
                    desc=f"Running with {self.config.solver_name} in epoch {epoch_id}",
                    total=self.env.num_v_nets,
                )
                if self.verbose <= 1
                else None
            )

            while True:
                solution = self.solver.solve(state)

                next_state, _, done, info = self.env.step(solution)

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "ac": f'{info["success_count"] / info["v_net_count"]:1.2f}',
                            "r2c": f'{info["long_term_r2c_ratio"]:1.2f}',
                            "inservice": f'{info["inservice_count"]:05d}',
                        }
                    )

                if done:
                    break
                state = next_state

            if pbar is not None:
                pbar.close()


class TimeWindowScenario(Scenario):
    """
    TODO: Batch Processing
    """

    def __init__(self, env, solver, config):
        super(TimeWindowScenario, self).__init__(env, solver, config)
        self.time_window_size = config.get("time_window_size", 100)

    def reset(self):
        self.current_time_window = 0
        self.next_event_id = 0
        return super().reset()

    def _receive(self):
        next_time_window = self.current_time_window + self.time_window_size
        enter_event_list = []
        leave_event_list = []
        while (
            self.next_event_id < len(self.v_net_simulator.events)
            and self.v_net_simulator.events[self.next_event_id]["time"]
            <= next_time_window
        ):
            if self.v_net_simulator.events[self.next_event_id]["type"] == 1:
                enter_event_list.append(self.v_net_simulator.events[self.next_event_id])
            else:
                leave_event_list.append(self.v_net_simulator.events[self.next_event_id])
            self.next_event_id += 1
        return enter_event_list, leave_event_list

    def _transit(self, solution_dict):
        return NotImplementedError

    def run(self):
        self.ready()

        for epoch_id in range(self.config.num_epochs):
            print(f"\nEpoch {epoch_id}") if self.verbose >= 2 else None
            pbar = (
                tqdm.tqdm(
                    desc=f"Running with {self.solver.name} in epoch {epoch_id}",
                    total=self.env.num_v_nets,
                )
                if self.verbose <= 1
                else None
            )
            instance = self.env.reset()

            current_event_id = 0
            events_list = self.env.v_net_simulator.events
            for current_time in range(
                0,
                int(events_list[-1]["time"] + self.time_window_size + 1),
                self.time_window_size,
            ):
                enter_event_list = []
                while events_list[current_event_id]["time"] < current_time:
                    # enter
                    if events_list[current_event_id]["type"] == 1:
                        enter_event_list.append(events_list[current_event_id])
                    # leave
                    else:
                        v_net_id = events_list[current_event_id]["v_net_id"]
                        solution = Solution(self.v_net_simulator.v_nets[v_net_id])
                        solution = self.recorder.get_record(v_net_id=v_net_id)
                        self.controller.release(
                            self.v_net_simulator.v_nets[v_net_id], self.p_net, solution
                        )
                        self.solution["description"] = "Leave Event"
                        record = self.count_and_add_record()
                    current_event_id += 1

                for enter_event in enter_event_list:
                    solution = self.solver.solve(instance)
                    next_instance, _, done, info = self.env.step(solution)

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "ac": f'{info["success_count"] / info["v_net_count"]:1.2f}',
                                "r2c": f'{info["long_term_r2c_ratio"]:1.2f}',
                                "inservice": f'{info["inservice_count"]:05d}',
                            }
                        )

                    if done:
                        break
                    instance = next_instance

            if pbar is not None:
                pbar.close()
