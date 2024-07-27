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

