import random
import numpy as np

import os, sys

sys.path.append(
    ".."
)  # 等价于 sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from network.physical_network import PhysicalNetwork
from network.virtual_network_sim import VirtualNetworkRequestSimulator

from utils.data import (
    get_p_net_dataset_dir_from_setting,
    get_v_net_dataset_dir_from_setting,
)


class Generator:

    @staticmethod
    def generate_dataset(config, p_bool=True, v_bool=True, save=False):
        """
        Generate a dataset consisting of a Physical Network and a Virtual Network request simulator.

        Args:
            config (dict or object): Configuration object containing the settings for the generator.
            p_bool (bool): Whether or not to generate a physical network dataset.
            v_bool (bool): Whether or not to generate a virtual network request simulator dataset.
            save (bool): Whether or not to save the generated datasets.

        Returns:
            Tuple: A tuple consisting of the generated Physical Network and a Virtual Network request simulator.
        """
        print(f"\n{'-' * 20}    Generate dataset Start    {'-' * 20}\n")

        # TODO: 可以只用一个函数`generate_net()`来实现
        p_net = Generator.generate_p_net(config=config, save=save) if p_bool else None
        v_net_sim = Generator.generate_v_net_sim(config, save=save) if v_bool else None

        print(f"\n{'-' * 20}    Generate dataset End    {'-' * 20}\n")

        return p_net, v_net_sim

    @staticmethod
    def generate_p_net(config, save=False):
        """
        Generate a physical network dataset based on the given configuration.

        Args:
            config (dict or object): Configuration object containing the settings for the generator.
            save (bool): Whether or not to save the generated dataset.

        Returns:
            PhysicalNetwork: A PhysicalNetwork object representing the generated dataset.
        """
        if not isinstance(config, dict):
            config = vars(config)

        p_net_setting = config["p_net_setting"]
        random.seed(config.get("seed", 0))
        np.random.seed(config.get("seed", 0))

        p_net = PhysicalNetwork.create_from_setting(p_net_setting)

        if save:
            Generator.save(p_net, p_net_setting, config, "p_net")

        # new_p_net = PhysicalNetwork.load_dataset(p_net_dataset_dir)
        return p_net

    @staticmethod
    def generate_v_net_sim(config, save=False):
        """
        Generate a virtual network request simulator dataset based on the given configuration.

        Args:
            config (dict or object): Configuration object containing the settings for the generator.
            save (bool): Whether or not to save the generated dataset.

        Returns:
            VirtualNetworkRequestSimulator: A VirtualNetworkRequestSimulator object representing the generated dataset.
        """
        if not isinstance(config, dict):
            config = vars(config)

        v_net_setting = config["v_net_setting"]
        random.seed(config.get("seed", 0))
        np.random.seed(config.get("seed", 0))

        v_net_sim = VirtualNetworkRequestSimulator.from_setting(v_net_setting)
        v_net_sim.generate_sim(v_nets=True, events=True, seed=None)

        if save:
            Generator.save(v_net_sim, v_net_setting, config, "v_net_sim")

        # new_v_net_sim = VirtualNetworkRequestSimulator.load_dataset(v_nets_dataset_dir)
        return v_net_sim

    def save(net, setting, config, label):
        """Save p_net/v_net to the specified location"""
        if label == "p_net":
            fpath = get_p_net_dataset_dir_from_setting(setting)
        else:
            fpath = get_v_net_dataset_dir_from_setting(setting)

        fpath = os.path.normpath(fpath)
        net.save_net(fpath)

        if config.get("verbose", 1):
            print(f"save p_net dataset in {fpath}")
