import os
import tqdm

from .controller import Controller
from .counter import Counter
from .recorder import Recorder
from .solution import Solution

from VNE.network.physical_network import PhysicalNetwork
from VNE.network.virtual_network_sim import VirtualNetworkRequestSimulator
from VNE.network.utils import get_p_net_dataset_dir_from_setting


class Scenario:

    def __init__(self, env, solver, config) -> None:
        self.env = env
        self.solver = solver
        self.config = config

        self.verbose = config.verbose

    @classmethod
    def from_config(cls, Env, Solver, config):
        """Create scenario from setting"""
        counter = Counter(
            config.v_sim_setting["node_attrs_setting"],
            config.v_sim_setting["link_attrs_setting"],
            **vars(config),
        )
        controller = Controller(
            config.v_sim_setting["node_attrs_setting"],
            config.v_sim_setting["link_attrs_setting"],
            **vars(config),
        )
        recorder = Recorder(counter, **vars(config))

        # Create/Load p_net
        config.p_net_dataset_dir = get_p_net_dataset_dir_from_setting(
            config.p_net_setting
        )
        # print(config.p_net_dataset_dir)
        if os.path.exists(config.p_net_dataset_dir):
            p_net = PhysicalNetwork.load_dataset(config.p_net_dataset_dir)
            (
                print(f"Load Physical Network from {config.p_net_dataset_dir}")
                if config.verbose >= 1
                else None
            )
        else:
            p_net = PhysicalNetwork.from_setting(config.p_net_setting)
            print(f"*** Generate Physical Network from setting")

        # Create v_net simulator
        v_net_simulator = VirtualNetworkRequestSimulator.from_setting(
            config.v_sim_setting
        )
        print(f"Create VNR Simulator from setting") if config.verbose >= 1 else None

        # create env and solver
        env = Env(p_net, v_net_simulator, controller, recorder, counter, **vars(config))
        solver = Solver(controller, recorder, counter, **vars(config))

        # Create scenario
        scenario = cls(env, solver, config)
        if config.verbose >= 2:
            print(config)
        if config.if_save_config:
            config.save()

        return scenario
