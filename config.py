import os
import json
import yaml
import networkx as nx
import pprint
import sys
from VNE_model.utils.class_dict import ClassDict


class Config(ClassDict):
    """
    Config Class for all the settings
    """

    def __init__(self, p_net_setting_path, v_net_setting_path, **kwargs) -> None:
        super().__init__()

        self.p_net_setting_path = p_net_setting_path
        self.v_net_setting_path = v_net_setting_path

        # Env
        self.time_window_size = kwargs.get("time_window_size", 100)
        self.renew_v_net_simulator = False
        self.node_resource_unit_price = kwargs.get("node_resource_unit_price", 1.0)
        self.link_resource_unit_price = kwargs.get("link_resource_unit_price", 1.0)
        self.revenue_service_time_weight = kwargs.get(
            "revenue_service_time_weight", 0.001
        )
        self.revenue_start_price_weight = kwargs.get("revenue_start_price_weight", 1.0)
        self.r2c_ratio_threshold = kwargs.get("r2c_ratio_threshold", 0.0)
        self.vn_size_threshold = kwargs.get("vn_size_threshold", 10000)

        # Log & Save
        self.root_path = os.path.join(sys.argv[0], os.pardir)
        self.if_save_records = kwargs.get("if_save_records", True)
        self.if_temp_save_records = kwargs.get("if_temp_save_records", True)
        self.if_save_config = kwargs.get("if_save_config", True)
        self.summary_dir = kwargs.get("summary_dir", "save/")
        self.save_dir = kwargs.get("save_dir", "save/")
        self.summary_file_name = kwargs.get("summary_file_name", "global_summary.csv")

        # VNE Solver
        self.solver_name = kwargs.get("solver_name", "nrm_rank")
        self.sub_solvr_name = kwargs.get("sub_solvr_name", None)
        self.pretrained_model_path = kwargs.get("pretrained_model_path", "")
        self.pretrained_subsolver_model_path = kwargs.get(
            "pretrained_subsolver_model_path", ""
        )
        self.verbose = kwargs.get(
            "verbose", 1
        )  # Level of showing information 0: no output, 1: output summary, 2: output detailed info
        self.reusable = kwargs.get(
            "reusable", False
        )  # Whether or not to allow to deploy several virtual nodes on the same physical node

        # Node Ranking & Link Mapping
        self.node_ranking_method = (
            "order"  # Method of node ranking: 'order' or 'greedy'
        )
        self.link_ranking_method = (
            "order"  # Method of link ranking: 'order' or 'greedy'
        )
        self.matching_mathod = "greedy"  # Method of node matching: 'greedy' or 'l2s2'
        self.shortest_method = (
            "k_shortest"  # Method of path finding: 'bfs_shortest' or 'k_shortest'
        )
        self.k_shortest = 10  # Number of shortest paths to be found
        self.allow_revocable = False  # Whether or not to allow to revoke a virtual node
        self.allow_rejection = False  # Whether or not to allow to reject a virtual node

        self.batch_size = 128
        self.target_steps = self.batch_size * 2

        # Read settings
        self.read_net_settings(p_net=True, v_net=True)
        self.create_dirs()
        # self.get_run_id()
        self.check_config()

    def update(self, new_args):
        """Update Default Config

        Args:
            args(dict): new config
        """
        if not isinstance(new_args, dict):
            new_args = vars(new_args)
        # (
        #     print(f"=" * 20 + " Update Default Config " + "=" * 20)
        #     if self.verbose > 0
        #     else None
        # )
        self.recursive_update(new_args=new_args)
        self.update_net_settings(new_args=new_args)
        self.choose_p_net_topology(new_args=new_args)

        self.create_dirs()
        self.check_config()
        # (
        #     print(f"=" * 20 + "=======================" + "=" * 20)
        #     if self.verbose > 0
        #     else None
        # )

    def update_net_settings(self, new_args):
        """Update p_net or v_net settings from new_args"""
        if "p_net_setting_path" in new_args:
            self.read_net_settings(p_net=True, v_net=False)
        if "v_net_setting_path" in new_args:
            self.read_net_settings(p_net=False, v_net=True)
        self.update_v_net_setting(new_args)

    def save_config(self, fname="config.yaml"):
        """Save config to `config.yaml`"""
        fpath = os.path.join(self.save_dir, self.solver_name)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fpath = os.path.join(fpath, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            if fpath[-4:] == "json":
                json.dump(vars(self), f)
            elif fpath[-4:] == "yaml":
                yaml.dump(vars(self), f)
            else:
                return ValueError("Only supports setting files in yaml or json format!")
        print(f"Save config.yaml in {fpath}")

    def read_net_settings(self, p_net: bool, v_net: bool):
        """Read p_net or v_net settings"""
        if p_net:
            self.read_p_net_config()
        if v_net:
            self.read_v_net_config()

    def read_p_net_config(self):
        """Read p_net config from p_net_setting_path"""
        # (
        #     print(f"Load p_net_setting from {self.p_net_setting_path}")
        #     if self.verbose > 0
        #     else None
        # )
        self.p_net_setting = self.read_setting(self.p_net_setting_path)

        if "file_path" in self.p_net_setting["topology"] and os.path.exists(
            self.p_net_setting["topology"]["file_path"]
        ):
            G = nx.read_gml(self.p_net_setting["topology"]["file_path"], label="id")
            self.p_net_setting["num_nodes"] = G.number_of_nodes()
        self.p_net_setting_num_nodes = self.p_net_setting["num_nodes"]
        self.p_net_setting_num_node_attrs = len(
            self.p_net_setting["node_attrs_setting"]
        )
        self.p_net_setting_num_link_attrs = len(
            self.p_net_setting["link_attrs_setting"]
        )
        self.p_net_setting_num_node_resource_attrs = len(
            [
                1
                for attr in self.p_net_setting["node_attrs_setting"]
                if attr["type"] == "resource"
            ]
        )
        self.p_net_setting_num_link_resource_attrs = len(
            [
                1
                for attr in self.p_net_setting["link_attrs_setting"]
                if attr["type"] == "resource"
            ]
        )
        self.p_net_setting_num_node_extrema_attrs = len(
            [
                1
                for attr in self.p_net_setting["node_attrs_setting"]
                if attr["type"] == "extrema"
            ]
        )
        self.p_net_setting_num_link_extrema_attrs = len(
            [
                1
                for attr in self.p_net_setting["link_attrs_setting"]
                if attr["type"] == "extrema"
            ]
        )

    def read_v_net_config(self):
        """Read v_net config from v_net_setting_path"""
        self.v_net_setting = self.read_setting(self.v_net_setting_path)
        self.v_net_setting_num_node_attrs = len(
            self.v_net_setting["node_attrs_setting"]
        )
        self.v_net_setting_num_link_attrs = len(
            self.v_net_setting["link_attrs_setting"]
        )
        self.v_net_setting_num_node_resource_attrs = len(
            [
                1
                for attr in self.v_net_setting["node_attrs_setting"]
                if attr["type"] == "resource"
            ]
        )
        self.v_net_setting_num_link_resource_attrs = len(
            [
                1
                for attr in self.v_net_setting["link_attrs_setting"]
                if attr["type"] == "resource"
            ]
        )

    def read_setting(self, fpath):
        """Read the setting from fpath"""
        fpath = os.path.join(self.root_path, fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            if fpath[-4:] == "json":
                setting_dict = json.load(f)
            elif fpath[-4:] == "yaml":
                setting_dict = yaml.load(f, Loader=yaml.Loader)
            else:
                return ValueError(
                    "Only supports settings files in yaml or json format!"
                )
        return setting_dict

    def create_dirs(self):
        """Create dirs to save settings, if already exist, do nothing

        Args:
            dir: self.save_dir, self.summary_dir, self.v_net_setting["save_path"] and self.p_net_setting[]"save_path"]
        """
        for dir in [
            self.save_dir,
            self.summary_dir,
            self.v_net_setting["save_dir"],
            self.p_net_setting["save_dir"],
        ]:
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)

    def check_config(self):
        """Check all configs"""
        assert self.reusable == False, "self.reusable == True Unsupported currently!"
        # if self.target_steps != -1:
        #     assert self.target_steps % self.batch_size == 0, "A should greater than b!"
        self.target_steps = self.batch_size * 2

    def recursive_update(self, new_args: dict):
        """Recursively update args

        Args:
            new_args(Dict[str, Any]): args to be updates
        """
        for key, value in new_args.items():
            if key in self.keys():
                if isinstance(value, dict):
                    self.recursive_update(self[key], value)
                else:
                    if self[key] != value:
                        self[key] = value
                        (
                            print(f"Update {key} with {value}")
                            if self.verbose > 0
                            else None
                        )
            else:
                self[key] = value
                print(f"Add {key} with {value}") if self.verbose > 0 else None

    def show_config(self):
        """Show all configs"""
        pprint.pprint(vars(self))

    def load_config(self, fpath):
        """Load config from fpath"""
        try:
            config = self.read_setting(fpath)
            print(f"Load config from {fpath}")
        except:
            print(f"No config file found in {fpath}")

    def choose_p_net_topology(self, new_args):
        """Choose p_net topology: Waxman100, Geant, or Waxman500

        Args:
            self.p_net_topology: 'wx100', 'grant', or 'wx500'
        """
        if new_args["p_net_topology"].lower() == "wx100":
            self.p_net_setting["topology"][
                "file_path"
            ] = "dataset/topology/Waxman100.gml"
            self.p_net_setting["num_nodes"] = 100
            self.p_net_setting_num_nodes = 100
        elif new_args["p_net_topology"].lower() == "geant":
            self.p_net_setting["topology"]["file_path"] = "dataset/topology/Geant.gml"
            self.p_net_setting["num_nodes"] = 40
            self.p_net_setting_num_nodes = 40
        elif new_args["p_net_topology"].lower() == "wx500":
            self.p_net_setting["topology"][
                "file_path"
            ] = "dataset/topology/Waxman500.gml"
            self.p_net_setting["num_nodes"] = 500
            self.p_net_setting_num_nodes = 500
        else:
            raise NotImplementedError

    def update_v_net_setting(self, new_args):
        """Update v_net settings by using new_args"""
        # if new_args.p_net_setting_topology_file_path is not None:
        #     assert (
        #         new_args.p_net_setting_num_nodes is None
        #     ), "p_net_setting_num_nodes and p_net_setting_topology_file_path cannot be set at the same time"
        if new_args["v_net_setting_num_v_nets"] is not None:
            self.v_net_setting["num_v_nets"] = new_args["v_net_setting_num_v_nets"]
        if new_args["v_net_setting_v_net_size_low"] is not None:
            self.v_net_setting["v_net_size"]["low"] = new_args[
                "v_net_setting_v_net_size_low"
            ]
        if new_args["v_net_setting_v_net_size_high"] is not None:
            self.v_net_setting["v_net_size"]["high"] = new_args[
                "v_net_setting_v_net_size_high"
            ]
        for i in range(len(self.v_net_setting["node_attrs_setting"])):
            if self.v_net_setting["node_attrs_setting"][i]["type"] == "resource":
                self.v_net_setting["node_attrs_setting"][i]["high"] = (
                    new_args["v_net_setting_node_resource_attrs_high"]
                    if new_args["v_net_setting_node_resource_attrs_high"] is not None
                    else self.v_net_setting["node_attrs_setting"][i]["high"]
                )
                self.v_net_setting["node_attrs_setting"][i]["low"] = (
                    new_args["v_net_setting_node_resource_attrs_low"]
                    if new_args["v_net_setting_node_resource_attrs_low"] is not None
                    else self.v_net_setting["node_attrs_setting"][i]["low"]
                )
        for i in range(len(self.v_net_setting["link_attrs_setting"])):
            if self.v_net_setting["link_attrs_setting"][i]["type"] == "resource":
                self.v_net_setting["link_attrs_setting"][i]["high"] = (
                    new_args["v_net_setting_link_resource_attrs_high"]
                    if new_args["v_net_setting_link_resource_attrs_high"] is not None
                    else self.v_net_setting["link_attrs_setting"][i]["high"]
                )
                self.v_net_setting["link_attrs_setting"][i]["low"] = (
                    new_args["v_net_setting_link_resource_attrs_low"]
                    if new_args["v_net_setting_link_resource_attrs_low"] is not None
                    else self.v_net_setting["link_attrs_setting"][i]["low"]
                )
        if new_args["v_net_setting_aver_lifetime"] is not None:
            self.v_net_setting["lifetime"]["scale"] = new_args[
                "v_net_setting_aver_lifetime"
            ]
        if new_args["v_net_setting_aver_arrival_rate"] is not None:
            self.v_net_setting["arrival_rate"]["lam"] = new_args[
                "v_net_setting_aver_arrival_rate"
            ]
        # if new_args.p_net_setting_num_nodes is not None:
        #     self.p_net_setting["num_nodes"] = new_args.p_net_setting_num_nodes
        # if new_args.p_net_setting_topology_file_path is not None:
        #     self.p_net_setting["topology"][
        #         "file_path"
        #     ] = new_args.p_net_setting_topology_file_path
        #     G = nx.read_gml(self.p_net_setting["topology"]["file_path"], label="id")
        #     self.p_net_setting["num_nodes"] = G.number_of_nodes()
        #     self.p_net_setting["num_links"] = G.number_of_edges()

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()
