from VNE_model.network.physical_network import PhysicalNetwork
from VNE_model.network.virtual_network import VirtualNetwork
from VNE_model.network.utils import path_to_links
from VNE_model.simulation.controller import Controller
from VNE_model.simulation.recorder import Recorder
from VNE_model.simulation.calculator import Calculator
from VNE_model.simulation.solution import Solution
from VNE_model.simulation.environment import SolutionStepEnvironment
from VNE_model.solver import registry

from ..solver import Solver

from ..node_rank import *
from ..link_rank import *


class NodeRankSolver(Solver):
    """
    NodeRankSolver is a base solver class that use node rank to solve the problem.
    """

    def __init__(
        self,
        controller: Controller,
        recorder: Recorder,
        calculator: Calculator,
        **kwargs
    ) -> None:
        """
        Initialize the NodeRankSolver.

        Args:
            controller: the controller to control the mapping process.
            recorder: the recorder to record the mapping process.
            calculator: the calculator to count the mapping process.
            kwargs: the keyword arguments.
        """
        super(NodeRankSolver, self).__init__(controller, recorder, calculator, **kwargs)
        # # node mapping
        # self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # # link mapping
        # self.shortest_method = kwargs.get('shortest_method', 'k_shortest')
        # self.k_shortest = kwargs.get('k_shortest', 10)

    def solve(self, instance: dict) -> Solution:
        v_net, p_net = instance["v_net"], instance["p_net"]

        solution = Solution(v_net)
        node_mapping_result = self.node_mapping(v_net, p_net, solution)
        if node_mapping_result:
            link_mapping_result = self.link_mapping(v_net, p_net, solution)
            if link_mapping_result:
                # SUCCESS
                solution["result"] = True
                return solution
            else:
                # FAILURE
                solution["route_result"] = False
        else:
            # FAILURE
            solution["place_result"] = False
        solution["result"] = False
        return solution

    def node_mapping(
        self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution
    ) -> bool:
        """Attempt to place virtual nodes onto appropriate physical nodes."""
        v_net_rank = self.node_rank(v_net)
        p_net_rank = self.node_rank(p_net)
        sorted_v_nodes = list(v_net_rank)
        sorted_p_nodes = list(p_net_rank)

        node_mapping_result = self.controller.node_mapping(
            v_net,
            p_net,
            sorted_v_nodes=sorted_v_nodes,
            sorted_p_nodes=sorted_p_nodes,
            solution=solution,
            reusable=False,
            inplace=True,
            matching_mathod=self.matching_mathod,
        )
        return node_mapping_result

    def link_mapping(
        self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution
    ) -> bool:
        """Attempt to route virtual links onto appropriate physical paths."""
        if self.link_rank is None:
            sorted_v_links = v_net.links
        else:
            v_net_edges_rank_dict = self.link_rank(v_net)
            v_net_edges_sort = sorted(
                v_net_edges_rank_dict.items(), reverse=True, key=lambda x: x[1]
            )
            sorted_v_links = [edge_value[0] for edge_value in v_net_edges_sort]

        link_mapping_result = self.controller.link_mapping(
            v_net,
            p_net,
            solution=solution,
            sorted_v_links=sorted_v_links,
            shortest_method=self.shortest_method,
            k=self.k_shortest,
            inplace=True,
        )
        return link_mapping_result


@registry.register(
    solver_name="order_rank",
    env_cls=SolutionStepEnvironment,
)
class OrderRankSolver(NodeRankSolver):
    """
    A node ranking-based solver that use the order of nodes in the graph as the rank.

    Methods:
        - solve: solve the problem instance.
        - node_mapping: place virtual nodes onto appropriate physical nodes.
        - link_mapping: route virtual links onto appropriate physical paths.
    """

    def __init__(
        self, controller: Controller, recorder: Recorder, calculator: Recorder, **kwargs
    ) -> None:
        super(OrderRankSolver, self).__init__(
            controller, recorder, calculator, **kwargs
        )
        self.node_rank = OrderNodeRank()
        self.link_rank = None
