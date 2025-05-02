import random
from line_profiler_pycharm import profile
from functools import lru_cache
from include import config
from dataclasses import dataclass
import networkx as nx
import itertools

@dataclass
class Path:
    link: (str, str)
    link_share_capacity: float
    elements: []
    link_out_type: str

class PathFinder:
    def __init__(self, allocation):
        self.allocation = allocation


    @staticmethod
    def get_graph_leafs(graph, src=None) -> []:
        if src is None:
            return [x for x in graph.nodes() if graph.out_degree(x) == 0]
        else:
            leaf_nodes = []
            for node in nx.dfs_preorder_nodes(graph, src):
                if graph.out_degree(node) == 0 and src != node:
                    leaf_nodes.append(node)
            return leaf_nodes

    @staticmethod
    def get_leafs_from_root_node(graph, root_node) -> []:
        """Return all leaf nodes starting at root_node"""
        leafs = []
        for node in graph.successors(root_node):
            if graph.out_degree(node) == 0:
                leafs.append(node)
            else:
                leafs += PathFinder.get_leafs_from_root_node(graph, node)
        return leafs

    @staticmethod
    def get_links_from_root_node(graph, root_node) -> ():
        subgraph = nx.bfs_tree(graph, root_node)
        return list(subgraph.edges())

    @staticmethod
    def get_links_between_src_dst(graph, src, dst) -> ():
        if src == dst:
            return []
        return next(nx.all_simple_edge_paths(graph, src, dst))

    @staticmethod
    def create_all_combinations(my_dict: {}):
        if len(my_dict) == 0:
            return []
        return list(itertools.product(*(my_dict[key] for key in my_dict)))


    @profile
    def find_path_and_min_demand(self, demand) -> ():
        source_link = self.allocation.get_next_link(self.allocation.root, demand)
        if source_link is None:
            return None, 0
        allocated = self._find_path_from_source(source_link[0], source_link[1], demand)
        path_with_metadata, demand,  = self._get_path_min_demand(allocated)
        return path_with_metadata, demand

    def find_random_path(self) -> ():
        leafs = self.get_graph_leafs(self.allocation.allocation_graph, self.allocation.root)
        leaf = random.choice(leafs)
        return self._find_random_path_from_source_to_leaf(self.allocation.root, leaf)

    def find_all_paths(self):
        leafs = self.get_graph_leafs(self.allocation.allocation_graph, self.allocation.root)
        all_paths = []
        for leaf in leafs:
            path = self._find_path_from_source_to_leaf(self.allocation.root, leaf)
            all_paths += path
        return all_paths

    @profile
    def _get_path_min_demand(self, path: []) -> ():
        """For each link in path, find the share on it and subtract scaled used demand.
        If negative, not available share"""
        path_with_metadata = []
        min_demand = float('inf')
        allocation_graph = self.allocation
        for link in path:
            link_demand, link_share_capacity, elements = self.allocation.get_link_residual_demand(link, True)
            if elements is None:
                continue
            link_out_type = allocation_graph.get_element_type(link[1])
            path_object = Path(link, link_share_capacity, elements, link_out_type)
            path_with_metadata.append(path_object)
            min_demand = min(min_demand, link_demand)

        return path_with_metadata, min_demand

    def _get_residual_demand(self, element, capacity):
        allocation_graph = self.allocation
        multiplier = allocation_graph.get_link_multiplier_for_aloc_element(element)
        return capacity / multiplier

    @profile
    def _find_path_from_source(self, prev_node, current_node, demand):
        allocated = [(prev_node, current_node)]
        while True:
            link = self.allocation.get_next_link(current_node, demand)
            if link is None:
                return allocated
            if self._link_is_from_tree_split(link):
                for successor in self.allocation.allocation_graph.successors(link[0]):
                    allocated += self._find_path_from_source(link[0], successor, demand)
                return allocated
            else:
                allocated.append(link)
            current_node = link[1]

    def _find_path_from_source_to_leaf(self, src, leaf):
        all_paths = list(nx.all_simple_paths(self.allocation.allocation_graph, src, leaf))
        all_paths_with_edges = []

        for path in all_paths:
            path_edges = []
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                path_edges.append(edge)
            path_with_metadata, demand, = self._get_path_min_demand(path_edges)
            all_paths_with_edges.append(path_with_metadata)

        return all_paths_with_edges

    @profile
    def _find_random_path_from_source_to_leaf(self, src, leaf):
        all_paths = list(nx.all_simple_paths(self.allocation.allocation_graph, src, leaf))
        path = random.choice(all_paths)
        path_edges = []
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            path_edges.append(edge)
        path_with_metadata, demand = self._get_path_min_demand(path_edges)
        return path_with_metadata

    @lru_cache(maxsize=100000)
    def _link_is_from_tree_split(self, link):
        node = link[0]
        node_type = self.allocation.get_element_type(node)
        return self.allocation.element_is_agg_node(node_type)

