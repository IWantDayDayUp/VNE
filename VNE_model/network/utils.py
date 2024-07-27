import os
import json
import yaml
import numpy as np


def read_setting(fpath, mode="r"):
    """Read the setting from a file"""
    with open(fpath, mode, encoding="utf-8") as f:
        if fpath[-4:] == "json":
            setting_dict = json.load(f)
        elif fpath[-4:] == "yaml":
            setting_dict = yaml.load(f, Loader=yaml.Loader)
        else:
            return ValueError("Only supports settings files in yaml and json format!")
    return setting_dict


def write_setting(setting_dict, fpath, mode="w"):
    """Write the setting to a file"""
    with open(fpath, mode, encoding="utf-8") as f:
        if fpath[-4:] == "json":
            json.dump(setting_dict, f)
        elif fpath[-4:] == "yaml":
            yaml.dump(setting_dict, f)
        else:
            return ValueError("Only supports settings files in yaml and json format!")
    return setting_dict


def path_to_links(path: list) -> list:
    """
    Converts a given path to a list of tuples containing two elements each.

    Args:
        path (list): A list of elements representing a path.

    Returns:
        list: A list of tuples, each tuple containing two elements from the given path.
    """
    assert len(path) > 1
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def generate_data_with_distribution(size: int, distribution: str, dtype: str, **kwargs):
    """
    Generate data with the given distribution and data type.

    Args:
        size (int): The size of the data.
        distribution (str): The distribution of the data.
        dtype (str): The data type of the data.
        **kwargs: Keyword arguments to pass to the distribution generator.

    Returns:
        np.ndarray: The generated data.
    """
    assert distribution in ["uniform", "normal", "exponential", "possion"]
    assert dtype in ["int", "float", "bool"]
    if distribution == "normal":
        loc, scale = kwargs.get("loc"), kwargs.get("scale")
        data = np.random.normal(loc, scale, size)
    elif distribution == "uniform":
        low, high = kwargs.get("low"), kwargs.get("high")
        if dtype == "int":
            data = np.random.randint(low, high + 1, size)
        elif dtype == "float":
            data = np.random.uniform(low, high, size)
    elif distribution == "exponential":
        scale = kwargs.get("scale")
        data = np.random.exponential(scale, size)
    elif distribution == "possion":
        lam = kwargs.get("lam")
        if kwargs.get("reciprocal", False):
            lam = 1 / lam
        data = np.random.poisson(lam, size)
    else:
        raise NotImplementedError(
            f"Generating {dtype} data following the {distribution} distribution is unsupporrted!"
        )
    return data.astype(dtype).tolist()


def conver_format(fpath_1, fpath_2):
    """Convert the format of setting file"""
    setting_dict = read_setting(fpath_1)
    write_setting(setting_dict, fpath_2)


def get_p_net_dataset_dir_from_setting(p_net_setting):
    """Get the directory of the dataset of physical networks from the setting of the physical network simulation."""
    p_net_dataset_dir = p_net_setting.get("save_dir")
    n_attrs = [n_attr["name"] for n_attr in p_net_setting["node_attrs_setting"]]
    e_attrs = [l_attr["name"] for l_attr in p_net_setting["link_attrs_setting"]]

    if (
        "file_path" in p_net_setting["topology"]
        and p_net_setting["topology"]["file_path"] not in ["", None, "None"]
        and os.path.exists(p_net_setting["topology"]["file_path"])
    ):
        p_net_name = (
            f"{os.path.basename(p_net_setting['topology']['file_path']).split('.')[0]}"
        )
    else:
        p_net_name = f"{p_net_setting['num_nodes']}-{p_net_setting['topology']['type']}_[{p_net_setting['topology']['wm_alpha']}-{p_net_setting['topology']['wm_beta']}]"
    node_attrs_str = "-".join(
        [
            f'{n_attr_setting["name"]}_{get_parameters_string(get_distribution_parameters(n_attr_setting))}'
            for n_attr_setting in p_net_setting["node_attrs_setting"]
        ]
    )
    link_attrs_str = "-".join(
        [
            f'{e_attr_setting["name"]}_{get_parameters_string(get_distribution_parameters(e_attr_setting))}'
            for e_attr_setting in p_net_setting["link_attrs_setting"]
        ]
    )

    p_net_dataset_middir = p_net_name + "-" + node_attrs_str + "-" + link_attrs_str
    # f"{n_attrs}-[{p_net_setting['node_attrs_setting'][0]['low']}-{p_net_setting['node_attrs_setting'][0]['high']}]-" + \
    # f"{e_attrs}-[{p_net_setting['link_attrs_setting'][0]['low']}-{p_net_setting['link_attrs_setting'][0]['high']}]"
    p_net_dataset_dir = os.path.join(p_net_dataset_dir, p_net_dataset_middir)
    return p_net_dataset_dir


def get_v_nets_dataset_dir_from_setting(v_sim_setting):
    """Get the directory of the dataset of virtual networks from the setting of the virtual network simulation."""
    v_nets_dataset_dir = v_sim_setting.get("save_dir")
    # n_attrs = [n_attr['name'] for n_attr in v_sim_setting['node_attrs_setting']]
    # e_attrs = [l_attr['name'] for l_attr in v_sim_setting['link_attrs_setting']]
    node_attrs_str = "-".join(
        [
            f'{n_attr_setting["name"]}_{get_parameters_string(get_distribution_parameters(n_attr_setting))}'
            for n_attr_setting in v_sim_setting["node_attrs_setting"]
        ]
    )
    link_attrs_str = "-".join(
        [
            f'{e_attr_setting["name"]}_{get_parameters_string(get_distribution_parameters(e_attr_setting))}'
            for e_attr_setting in v_sim_setting["link_attrs_setting"]
        ]
    )

    v_nets_dataset_middir = (
        f"{v_sim_setting['num_v_nets']}-[{v_sim_setting['v_net_size']['low']}-{v_sim_setting['v_net_size']['high']}]-"
        + f"{v_sim_setting['topology']['type']}-{get_parameters_string(get_distribution_parameters(v_sim_setting['lifetime']))}-{v_sim_setting['arrival_rate']['lam']}-"
        + node_attrs_str
        + "-"
        + link_attrs_str
    )
    # f"{n_attrs}-[{v_sim_setting['node_attrs_setting'][0]['low']}-{v_sim_setting['node_attrs_setting'][0]['high']}]-" + \
    # f"{e_attrs}-[{v_sim_setting['link_attrs_setting'][0]['low']}-{v_sim_setting['link_attrs_setting'][0]['high']}]"
    v_net_dataset_dir = os.path.join(v_nets_dataset_dir, v_nets_dataset_middir)
    return v_net_dataset_dir


def get_distribution_parameters(distribution_dict):
    """Get the parameters of the distribution."""
    distribution = distribution_dict.get("distribution", None)
    if distribution is None:
        return []
    if distribution == "exponential":
        parameters = [distribution_dict["scale"]]
    elif distribution == "possion":
        parameters = [distribution_dict["lam"]]
    elif distribution == "uniform":
        parameters = [distribution_dict["low"], distribution_dict["high"]]
    elif distribution == "customized":
        parameters = [distribution_dict["min"], distribution_dict["max"]]
    return parameters


def get_parameters_string(parameters):
    """Get the string of the parameters."""
    if len(parameters) == 0:
        return "None"
    elif len(parameters) == 1:
        return str(parameters[0])
    else:
        str_parameters = [str(p) for p in parameters]
        return f'[{"-".join(str_parameters)}]'

