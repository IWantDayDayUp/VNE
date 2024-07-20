import random
import numpy as np

from ..network.physical_network import PhysicalNetwork
from ..utils.data import (
    get_p_net_dataset_dir_from_setting,
    get_v_nets_dataset_dir_from_setting,
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
        p_net = Generator.generate_p_net(config=config, save=save) if p_bool else None
        v_net_sim = Generator.generate_v_net_sim(config, save=save) if v_bool else None

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
        random.seed(config["seed"])
        np.random.seed(config["seed"])

        p_net = PhysicalNetwork.create_from_setting(p_net_setting)

        if save:
            p_net_dataset_dir = get_p_net_dataset_dir_from_setting(p_net_setting)
            p_net.save_net(p_net_dataset_dir)
            if config.get("verbose", 1):
                print(f"save p_net dataset in {p_net_dataset_dir}")

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

        v_sim_setting = config["v_sim_setting"]
        random.seed(config["seed"])
        np.random.seed(config["seed"])

        v_net_sim = VirtualNetworkRequestSimulator.from_setting(v_sim_setting)
        v_net_sim.renew()

        if save:
            v_nets_dataset_dir = get_v_nets_dataset_dir_from_setting(v_sim_setting)
            v_net_sim.save_net(v_nets_dataset_dir)
            if config.get("verbose", 1):
                print(f"save v_net dataset in {v_nets_dataset_dir}")

        # new_v_net_sim = VirtualNetworkRequestSimulator.load_dataset(v_nets_dataset_dir)
        return v_net_sim
