import argparse

parser = argparse.ArgumentParser(description="configuration file")


def str2bool(v):
    """str/int -> bool"""
    if isinstance(v, bool):
        return v
    elif isinstance(v, int):
        return False if v == 0 else True
    elif v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "n", "f", "0"):
        return True
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_dataset_args():
    """Get p_net and v_net args"""
    dataset_arg = parser.add_argument_group("dataset")

    dataset_arg.add_argument(
        "--p_net_setting_path",
        type=str,
        default="settings/p_net_setting_multi_resource.yaml",
        help="File path of physical network settings",
    )
    dataset_arg.add_argument(
        "--v_net_setting_path",
        type=str,
        default="settings/v_net_setting_multi_resource.yaml",
        help="File path of virtual network settings",
    )
    dataset_arg.add_argument(
        "--p_net_topology", type=str, default="wx100", help="Physical network topology"
    )
    dataset_arg.add_argument(
        "--v_net_setting_num_v_nets", type=int, default=1000, help=""
    )
    dataset_arg.add_argument(
        "--v_net_setting_v_net_size_low", type=int, default=2, help=""
    )
    dataset_arg.add_argument(
        "--v_net_setting_v_net_size_high", type=int, default=10, help=""
    )
    dataset_arg.add_argument(
        "--v_net_setting_node_resource_attrs_low", type=int, default=0, help=""
    )
    dataset_arg.add_argument(
        "--v_net_setting_node_resource_attrs_high", type=int, default=20, help=""
    )
    dataset_arg.add_argument(
        "--v_net_setting_link_resource_attrs_low", type=int, default=0, help=""
    )
    dataset_arg.add_argument(
        "--v_net_setting_link_resource_attrs_high", type=int, default=20, help=""
    )
    dataset_arg.add_argument(
        "--v_net_setting_aver_arrival_rate", type=float, default=0.08, help=""
    )
    dataset_arg.add_argument(
        "--v_net_setting_aver_lifetime", type=int, default=500, help=""
    )


def get_system_args():
    """Get system args

    system args: summary, save, record, and so on.
    """
    sys_arg = parser.add_argument_group("system")

    sys_arg.add_argument(
        "--summary_dir",
        type=str,
        default="save/",
        help="File Directory to save summary and records",
    )
    sys_arg.add_argument(
        "--summary_file_name",
        type=str,
        default="global_summary.csv",
        help="Summary file name",
    )
    sys_arg.add_argument(
        "--if_save_records", type=str2bool, default=True, help="Whether to save records"
    )
    sys_arg.add_argument(
        "--if_temp_save_records",
        type=str2bool,
        default=True,
        help="Whether to temporarily save records",
    )
    sys_arg.add_argument(
        "--p_net_name", type=str2bool, default=True, help="Name of the physical network"
    )
    sys_arg.add_argument(
        "--r2c_ratio_threshold",
        type=float,
        default=0.0,
        help="Threshold of revenue-to-cost ratio",
    )
    sys_arg.add_argument(
        "--vn_size_threshold",
        type=float,
        default=0.0,
        help="Threshold of virtual network size",
    )


def get_solver_args():
    """Get solver args: name, verbose, reusable"""
    solver_arg = parser.add_argument_group("solver")

    solver_arg.add_argument(
        "--solver_name",
        type=str,
        default="node_rank",
        help="Name of the solver selected to run",
    )
    solver_arg.add_argument(
        "--verbose", type=str2bool, default=1, help="Level of showing information"
    )
    solver_arg.add_argument(
        "--reusable",
        type=str2bool,
        default=False,
        help="Whether or not to allow deploy several VN nodes on the same p_node",
    )


def get_all_arg_group():
    get_dataset_args()
    get_system_args()
    get_solver_args()


def get_args(args=None):
    get_all_arg_group()
    config = parser.parse_args(args)
    print("sdfgvdsfgv")
    return config
