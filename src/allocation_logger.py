import random
import pandas as pd
from line_profiler_pycharm import profile
from functools import lru_cache
from dataclasses import dataclass
from users import User
import networkx as nx
from include import config
from multipliers import DefaultMultiplierDict
from path_finder import PathFinder
from enhanced_graph import EnhancedDiGraph
from element_translator import ElementTranslator

NODE = 'node'
LINK = 'link'
AGG = 'agg'


class AllocationGraph:
    def __init__(self, req: User, alt_name: str, multiplier: DefaultMultiplierDict, app, vars, is_online=False):
        self.is_online = is_online
        self.req = req
        self.alt_name = alt_name
        self.app = app
        self.allocation_graph = nx.DiGraph()
        self.weight = 0
        self.multiplier = multiplier
        self.vars = vars
        self.root = None
        self.element: (str, str, str, str) = ()
        self.links_for_rr = {}
        self.virtual_to_physical = {}
        self.non_allocated_links = {}
        self.update_weight()
        self.share_by_element = {}
        self.collected_paths_costs = []

    @profile
    def add_element(self, i: str, j: str, m: str, n: str, demand: float):
        element = (i, j, m, n)
        self._update_element(element)
        if not self._has_allocated_share():
            self._reset_element()
            return
        if i == j:
            self._add_node_to_alloc_graph(demand)
            self.share_by_element[n] = self.share_by_element.get(n, 0) + demand
        else:
            self._add_link_to_alloc_graph(demand)
            links = self.non_allocated_links.get((i, j), [])
            links.append((element, demand))
            self.non_allocated_links[(i, j)] = links
            self.share_by_element[(m, n)] = self.share_by_element.get((m, n), 0) + demand
        self._reset_element()

    def update_weight(self):
        element = (config.user_func, config.user_func, self.req.assoc_node, self.req.assoc_node)
        self.weight = self.get_demand(element)

    def get_weight(self):
        return self.weight

    def _add_node_to_alloc_graph(self, demand):
        node = ElementTranslator.get_node(self.element)
        self._add_node(node, demand)
        self.allocation_graph.nodes[node][NODE] = True

    def _add_link_to_alloc_graph(self, demand):
        node = ElementTranslator.get_link(self.element)
        self._add_node(node, demand)
        self.allocation_graph.nodes[node][LINK] = True

    def _add_edge(self, src_node, dst_node, demand: float = 0, dst_type=NODE):
        if demand == 0:
            return 0
        demand_to_connect = demand
        if self.allocation_graph.has_edge(src_node, dst_node):
            existing_demand = self.get_link_demand(src_node, dst_node)
            demand_to_connect += existing_demand
        share = demand_to_connect * self._get_multiplier(dst_type)
        self.allocation_graph.add_edge(src_node, dst_node, capacity=share, demand=demand_to_connect)
        if dst_type == NODE:
            self._update_node_demand(dst_node, demand)
        return demand

    def _add_link_to_node_edge(self, demand: float = 0):
        src_node, trans_node1 = ElementTranslator.get_link_to_link_edge(self.element)
        trans_node2, dst_node = ElementTranslator.get_link_to_self_node_edge(self.element)
        assert trans_node1 == trans_node2
        demand = self._add_edge(src_node, trans_node1, demand=demand, dst_type=LINK)
        demand = self._add_edge(trans_node2, dst_node, demand=demand)
        self.allocation_graph.nodes[src_node][LINK] = True
        self.allocation_graph.nodes[trans_node1][LINK] = True
        self.allocation_graph.nodes[dst_node][NODE] = True
        self._update_link_demand(self.element, demand)
        return demand

    def _add_node_to_link_edge(self, demand: float = 0):
        src_node, dst_node = ElementTranslator.get_node_to_link_edge(self.element)
        if self._is_link_from_tree_split():
            agg_node = AllocationGraph._get_agg_node_name(src_node)
            self._add_agg_edge(src_node, agg_node)
            demand = self._add_edge(agg_node, dst_node, demand=demand, dst_type=LINK)
        else:
            demand = self._add_edge(src_node, dst_node, demand=demand, dst_type=LINK)
        self.allocation_graph.nodes[src_node][NODE] = True
        self.allocation_graph.nodes[dst_node][LINK] = True
        self._update_link_demand(self.element, demand)
        return demand

    def _add_link_to_link_edge(self, demand: float = 0):
        src_node, dst_node = ElementTranslator.get_link_to_link_edge(self.element)
        demand = self._add_edge(src_node, dst_node, demand=demand, dst_type=LINK)
        self.allocation_graph.nodes[src_node][LINK] = True
        self.allocation_graph.nodes[dst_node][LINK] = True
        self._update_link_demand(self.element, demand)
        return demand

    def _is_link_from_tree_split(self):
        i, j, m, n = self.element
        return self.app.is_func_tree_split(i)

    @profile
    def get_next_link(self, node: (), demand):
        if config.online_mode:
            if demand == 0:
                successors = list(self.allocation_graph.successors(node))
                if len(successors) == 0:
                    return None
                random_successor = random.choice(successors)
                return node, random_successor
            else:
                return self.get_max_demand_link(node, demand)
        else:
            successor = next(iter(self.allocation_graph.successors(node)), None)
            if successor is None:
                return None
            return node, successor

    def get_link_residual_demand(self, link, export_metadata=False):
        elements = ElementTranslator.alloc_graph_to_physical_graph_elements(self, link)
        if len(elements) == 0:
            if export_metadata:
                return None, None, None
            else:
                return None
        capacity = self.get_link_capacity(link)
        dst_element = elements[-1]
        dst_element_multiplier = self.get_link_multiplier_for_aloc_element(dst_element)
        link_demand = capacity / dst_element_multiplier
        if export_metadata:
            return link_demand, capacity, elements
        else:
            return link_demand

    def get_max_demand_link(self, node, request_demand):
        successors = list(self.allocation_graph.successors(node))
        min_demand = float('inf')
        min_demand_link = None
        if len(successors) == 1:
            return node, successors[0]
        for successor in successors:
            demand = self.get_link_residual_demand((node, successor))
            successor_successors = list(self.allocation_graph.successors(successor))
            max_successor_successors_demand = 0
            for successor_successor in successor_successors:
                successor_demand = self.get_link_residual_demand((successor, successor_successor))
                if successor_demand is None:
                    successor_demand = float('inf')
                if successor_demand > max_successor_successors_demand:
                    max_successor_successors_demand = successor_demand
            if request_demand <= demand < min_demand and max_successor_successors_demand >= request_demand:
                min_demand = demand
                min_demand_link = (node, successor)
        if min_demand_link is None and len(successors) > 0:
            min_demand_link = (node, successors[0])
        return min_demand_link

    def get_min_cost_link(self, node, request_demand):
        successors = list(self.allocation_graph.successors(node))
        min_cost = float('inf')
        min_cost_link = None
        for successor in successors:
            demand = self.get_link_residual_demand((node, successor))
            cost = self.allocation_graph.nodes[successor].get('basic_cost', 0)
            if demand > request_demand and cost < min_cost:
                min_cost = cost
                min_cost_link = (node, successor)
        if min_cost_link is None and len(successors) > 0:
            min_cost_link = (node, successors[0])
        return min_cost_link

    def _collect_all_paths_to_list(self):
        if not config.online_mode or config.skip_heuristic:
            return
        from comparison_algorithms import EmbeddingAlgorithm

        demand = self.req.demand if self.is_online else config.request_demand_size_mean
        if len(self.app.tree_splitters) > 0:
            for i in range(10):
                path, _ = self.run_path_finder()
                cost = self.get_cost_from_req_allocation_path(path)
                path_dict = EmbeddingAlgorithm.collect_path_dict_from_path_finder(path, self, demand)
                if self._no_such_dict_in_collected_paths(path_dict):
                    self.collected_paths_costs.append((path_dict, cost))
        else:
            paths = self.find_all_alloc_graph_paths()
            for path in paths:
                cost = self.get_cost_from_req_allocation_path(path)
                path_dict = EmbeddingAlgorithm.collect_path_dict_from_path_finder(path, self, demand)
                self.collected_paths_costs.append((path_dict, cost))
            self.collected_paths_costs = sorted(self.collected_paths_costs, key=lambda x: x[1])

    def _no_such_dict_in_collected_paths(self, path_dict):
        for d, _ in self.collected_paths_costs:
            if self._identical_dicts(d, path_dict):
                return False
        return True

    def get_cost_from_req_allocation_path(self, path, demand=config.request_demand_size_mean):
        cost = 0
        for path_object in path:
            for element in path_object.elements:
                # print(element)
                i, j, m, n = element
                cost += self.multiplier.get_base_cost(self.app.get_app_name(), i, j, m, n) * demand
        # print('\n\n')
        return cost

    @staticmethod
    def _identical_dicts(dict1, dict2):
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        else:
            return True


    def _connect_local_nodes(self, src_node, dst_node, src_type, dst_type):
        demand_from = self.get_node_demand(src_node)
        demand_to = self.get_node_demand(dst_node)
        demand_to_residual = self.get_node_demand(dst_node, residual=True)
        demand = min(demand_from, demand_to, demand_to_residual)
        element = src_node[0], dst_node[0], src_node[1], dst_node[1]
        self._update_element(element)
        share = demand * self._get_multiplier()
        if src_type is NODE and self._is_link_from_tree_split():
            agg_node = AllocationGraph._get_agg_node_name(src_node)
            self._add_agg_edge(src_node, agg_node)
            self.allocation_graph.add_edge(agg_node, dst_node, capacity=share, demand=demand)
        else:
            self.allocation_graph.add_edge(src_node, dst_node, capacity=share, demand=demand)
        self.allocation_graph.nodes[src_node][src_type] = True
        self.allocation_graph.nodes[dst_node][dst_type] = True
        self._update_node_demand(dst_node, demand)
        self._reset_element()
        return demand

    def add_agg_node(self, node):
        self.allocation_graph.add_node(node)
        self.residual_allocation_graph.add_node(node)
        self.allocation_graph.nodes[node][AGG] = True

    def _update_node_demand(self, node, demand):
        self.residual_allocation_graph.nodes[node]['demand'] -= demand
        if self.residual_allocation_graph.nodes[node]['demand'] < -config.allowed_error:
            raise ValueError(f"Unexpected demand: {self.residual_allocation_graph.nodes[node]['demand']}")
        if self.residual_allocation_graph.nodes[node]['demand'] < config.allowed_error:
            self.residual_allocation_graph.nodes[node]['demand'] = 0
            if len(node) == 2:
                self.allocated_nodes_j.remove(node)

    def _update_link_demand(self, link, demand):
        for link_w_demand in self.links_w_demand:
            _link, _demand = link_w_demand
            if _link == link:
                _demand -= demand
                if _demand < -config.allowed_error:
                    raise ValueError(f"Unexpected demand: {_demand}")
                self.links_w_demand.remove(link_w_demand)
                if _demand > config.allowed_error:
                    self.links_w_demand.append((_link, _demand))
                return

    def _add_node(self, node, demand):
        share = demand * self._get_multiplier()
        if self.allocation_graph.has_node(node):
            self.allocation_graph.nodes[node]['capacity'] += share
            self.allocation_graph.nodes[node]['demand'] += demand
            return
        else:
            self.allocation_graph.add_node(node, capacity=share, demand=demand)
            self._add_to_virtual_to_physical_dict(node)

    def _add_to_virtual_to_physical_dict(self, node):
        if len(node) == 3:
            func = (self.element[0], self.element[1])
        else:
            func = node[0]
        nodes = self.virtual_to_physical.get(func, [])
        nodes.append(node)
        self.virtual_to_physical[func] = nodes

    def _add_agg_edge(self, from_node, to_agg_node):
        self.add_agg_node(to_agg_node)
        nx.set_node_attributes(self.allocation_graph, {to_agg_node: {AGG: True}})
        self.allocation_graph.add_edge(from_node, to_agg_node, capacity=0)
        self.residual_allocation_graph.add_edge(from_node, to_agg_node, capacity=0)

    def _is_element_in_graph(self) -> bool:
        return self.allocation_graph.has_node(self.element)

    def get_demand(self, element=None) -> float:
        if self.is_online:
            var = self._get_variable_online(element)
        else:
            var = self._get_variable_offline(element)
        # return self.solution.get_value(var)
        if var not in self.vars:
            # print(f"Variable {var} not found in solution")
            return 0
        return self.vars[var]

    def get_link_capacity(self, link) -> float:
        return self.allocation_graph[link[0]][link[1]]['capacity']

    def get_link_demand(self, from_node, to_node) -> float:
        return self.allocation_graph.edges[from_node, to_node]['demand']

    def set_link_demand(self, from_node, to_node, demand):
        self.allocation_graph.edges[from_node, to_node]['demand'] = demand

    def get_node_capacity(self, node) -> float:
        return self.allocation_graph.nodes[node]['capacity']

    def get_node_demand(self, node, residual=False) -> float:
        if residual:
            graph = self.residual_allocation_graph
        else:
            graph = self.allocation_graph
        if node in graph.nodes() and 'demand' in graph.nodes[node]:
            return graph.nodes[node]['demand']
        else:
            element = node[0], node[0], node[1], node[1]
            return self.get_demand(element)

    def get_allocated_node_demand(self, node) -> float:
        if node in self.allocation_graph.nodes() and 'demand' in self.allocation_graph.nodes[node]:
            return self.allocation_graph.nodes[node]['demand']
        else:
            return 0

    def set_link_capacity(self, link, capacity):
        self.allocation_graph[link[0]][link[1]]['capacity'] = capacity

    def _has_allocated_share(self) -> bool:
        """Check if LP allocated share and if the share is not very small"""
        if not self.get_demand():
            return False
        else:
            return True

    def _get_variable_online(self, element=None):
        if element is None:
            element = self.element
        element = self._get_element_tuple_online(element, self.req)
        return tuple(element)

    def _get_variable_offline(self, element=None):
        if element is None:
            element = self.element
        element = self._get_element_tuple_offline(element, self.alt_name, self.req)
        return tuple(element)

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_element_tuple_offline(element, alt_name, req):
        elements = list(element)
        elements.insert(0, req.assoc_node)
        elements.insert(0, alt_name)
        elements.insert(0, req.app_name)
        return elements

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_element_tuple_online(element, req):
        elements = list(element)
        elements.insert(0, req.assoc_node)
        elements.insert(0, req.id)
        elements.insert(0, req.app_name)
        return elements

    def find_max_flow_from_root_to_leaf(self, leaf: ()) -> float:
        return nx.maximum_flow(self.allocation_graph, self.root, leaf)

    def run_path_finder(self, demand=0):
        return PathFinder(self).find_path_and_min_demand(demand)

    def find_all_alloc_graph_paths(self):
        """Use only for chain apps"""
        return PathFinder(self).find_all_paths()

    def find_random_alloc_graph_paths(self):
        return PathFinder(self).find_random_path()

    def _reset_element(self):
        self._update_element(element=())

    def _update_element(self, element: (str, str, str, str)) -> None:
        self.element = element

    @profile
    def post_process_graph(self):
        if self.weight == 0:
            return
        self.residual_allocation_graph = self.allocation_graph.copy()
        self._connect_graph_nodes(config.user_func)
        self._check_all_demand_allocated()
        self._update_graph_root()
        self._set_demand_to_agg_nodes()
        self._collect_all_paths_to_list()

    @profile
    def _connect_graph_nodes(self, i):
        successors = self.app.graph.successors(i)
        for j in successors:
            self._connect_node_pair(i, j)
            self._connect_graph_nodes(j)

    def _connect_node_pair(self, i, j):
        if j not in self.virtual_to_physical:
            return
        self.allocated_nodes_i = self.virtual_to_physical[i].copy()
        self.allocated_nodes_j = self.virtual_to_physical[j].copy()
        self.allocated_links_ij = self.virtual_to_physical.get((i, j), []).copy()
        self.links_w_demand = self.non_allocated_links.get((i, j), [])
        self._case_1_local_nodes()
        self._case_2_remote_node_to_node()
        if len(self.links_w_demand) > 0:
            raise ValueError(f"Links not allocated: {self.links_w_demand}")

    def _case_1_local_nodes(self):
        allocated_nodes_i = self.allocated_nodes_i.copy()
        allocated_nodes_j = self.allocated_nodes_j.copy()
        for node_i in allocated_nodes_i:
            for node_j in allocated_nodes_j:
                if node_i[1] == node_j[1]:
                    self._connect_local_nodes(node_i, node_j, NODE, NODE)
        allocated_links_ij = self.allocated_links_ij.copy()
        allocated_nodes_j = self.allocated_nodes_j.copy()
        for link_ij in allocated_links_ij:
            for node_j in allocated_nodes_j:
                if link_ij[2] == node_j[1]:
                    self._connect_local_nodes(link_ij, node_j, LINK, NODE)

    def _case_2_remote_node_to_node(self):
        if len(self.links_w_demand) == 0:
            return
        allocated_nodes_i = self.allocated_nodes_i.copy()
        allocated_links_ij = self.allocated_links_ij.copy()
        #node-link
        for node_i in allocated_nodes_i:
            for link_ij in allocated_links_ij:
                for link, link_demand in self.links_w_demand:
                    m = node_i[1]
                    n = link_ij[2]
                    if link[2] == m and link[3] == n:
                        self._update_element(link)
                        dst_demand = self.residual_allocation_graph.nodes[link_ij]['demand']
                        demand = min(link_demand, dst_demand)
                        self._add_node_to_link_edge(demand)
                        self._reset_element()
        if len(self.links_w_demand) == 0:
            return
        #link-link
        allocated_links_ij = self.allocated_links_ij.copy()
        for link_ij1 in allocated_links_ij:
            for link_ij2 in allocated_links_ij:
                if link_ij1 == link_ij2:
                    continue
                for link, link_demand in self.links_w_demand:
                    m = link_ij1[2]
                    n = link_ij2[2]
                    if link[2] == m and link[3] == n:
                        self._update_element(link)
                        dst_demand = self.residual_allocation_graph.nodes[link_ij2]['demand']
                        # src_demand = self.residual_allocation_graph.nodes[node_i]['demand']
                        demand = min(link_demand, dst_demand)
                        # self._add_node_to_node_edge_case_2(demand)
                        self._add_link_to_link_edge(demand)
                        self._reset_element()
        if len(self.links_w_demand) > 0:
            raise ValueError(f"Links not allocated: {self.links_w_demand}")

    def _check_all_demand_allocated(self):
        for node in self.residual_allocation_graph.nodes:
            if config.user_func == node[0]:
                continue
            # if self.residual_allocation_graph.nodes[node]['demand'] > config.allowed_error:
            if len(node) == 2 and self.residual_allocation_graph.nodes[node]['demand'] > config.allowed_error:
                print(f"Node {node} demand not allocated: {self.residual_allocation_graph.nodes[node]['demand']}, user at {self.req.assoc_node}, "
                      f"request: {self.req.id}, demand: {self.req.demand}")
                print(f'Nodes: {list(self.allocation_graph.nodes(data=True))}\n'
                      f'Links: {list(self.allocation_graph.edges(data=True))}\n')
                raise ValueError(f"Node {node} demand not allocated: {self.residual_allocation_graph.nodes[node]['demand']}")

    def _connect_path(self, path):
        min_path_demand = min([dem for _, _, dem in path])
        for ind, (m, n, _) in enumerate(path):
            link_found = False
            for link, _ in self.links_w_demand:
                if m == link[2] and n == link[3]:
                    link_found = True
                    self._update_element(link)
                    break
            if not link_found:
                raise ValueError(f"Link not found: {m}-{n}")
            if ind == 0:
                self._add_node_to_link_edge(min_path_demand)
            elif ind == len(path) - 1:
                self._add_link_to_node_edge(min_path_demand)
            else:
                self._add_link_to_link_edge(min_path_demand)

            self._reset_element()

    def _update_graph_root(self):
        root = self._get_graph_roots(self.allocation_graph)
        if len(root) > 1:
            print(f"More than one root found in allocation tree: {root}, user at {self.req.assoc_node}, "
                  f"request: {self.req.id}, demand: {self.req.demand}")
            print(f'Nodes: {list(self.allocation_graph.nodes(data=True))}\n'
                  f'Links: {list(self.allocation_graph.edges(data=True))}\n')
            raise ValueError(f"More than one root found in allocation tree: {root}, user at {self.req.assoc_node}")
        if root[0][0] != config.user_func:
            raise ValueError(f"Root is not {config.user_func}")
        if root[0][1] != self.req.assoc_node:
            raise ValueError("Root is not at assoc_node")
        self.root = root[0]

    def _set_demand_to_agg_nodes(self):
        for node in self.get_all_agg_nodes():
            capacity = 0
            for link in list(self.allocation_graph.out_edges(node)):
                capacity += self.allocation_graph[link[0]][link[1]]['capacity']
            agg_in_edge = list(self.allocation_graph.in_edges(node))
            if len(agg_in_edge) > 1:
                raise ValueError("More than one edge to agg node")
            else:
                agg_in_edge = agg_in_edge[0]
            self.allocation_graph[agg_in_edge[0]][agg_in_edge[1]]['capacity'] = capacity

    @staticmethod
    def _get_agg_node_name(from_node: ()) -> ():
        return from_node[0], from_node[1], 'AGG'

    @staticmethod
    def _get_graph_roots(graph):
        return [n for n, d in graph.in_degree() if d == 0]

    @profile
    def update_aloc_graph_link_capacity(self, allocated_share, link, capacity, zero_share_edges=[]):
        # capacity = self.get_link_capacity(link)
        new_residual_share = capacity - allocated_share
        # allow small error due to floating point arithmetic
        if new_residual_share <= config.allowed_error:
            new_residual_share = 0
            zero_share_edges.append(link)
        self.set_link_capacity(link, new_residual_share)
        return zero_share_edges

    def _get_multiplier(self, node_type=NODE, element=None):
        if element is None:
            element = self.element
        i, j, m, n = element
        if node_type == NODE:
            return self.multiplier.get_node_multiplier(self.app.app, j, m)
        else:
            return self.multiplier[self.app.app, i, j, m, n]

    @lru_cache(maxsize=10000)
    @profile
    def get_link_multiplier_for_aloc_element(self, element) -> float:
        i, j, m, n = element
        return self.multiplier[self.app.get_app_name(), i, j, m, n]

    @profile
    def remove_zero_capacity_elements(self, links: []):
        for link in links:
            self.allocation_graph.remove_edge(link[0], link[1])
        nodes_with_degree_0 = [node for node, degree in self.allocation_graph.degree() if degree == 0]
        for node in nodes_with_degree_0:
            self.allocation_graph.remove_node(node)
        return self._remove_graph_if_fully_allocated()

    def _remove_graph_if_fully_allocated(self):
        if self.root not in self.allocation_graph.nodes:
            self.allocation_graph.clear()
        elif len(self.allocation_graph.nodes) == 1:
            self.allocation_graph.remove_node(self.root)
        else:
            return False
        return True

    def get_all_agg_nodes(self):
        return list(nx.get_node_attributes(self.allocation_graph, AGG).keys())

    @lru_cache(maxsize=100000)
    @profile
    def get_element_type(self, element):
        attributes = self.allocation_graph.nodes[element]
        if NODE in attributes:
            return NODE
        elif LINK in attributes:
            return LINK
        elif AGG in attributes:
            return AGG
        else:
            raise ValueError("Unexpected node type")

    @staticmethod
    @lru_cache(maxsize=100000)
    def element_is_transient(element_type) -> bool:
        return True if element_type == LINK else False

    @staticmethod
    @lru_cache(maxsize=100000)
    def element_is_allocated_node(element_type) -> bool:
        return True if element_type == NODE else False

    @staticmethod
    @lru_cache(maxsize=100000)
    def element_is_agg_node(element_type) -> bool:
        return True if element_type == AGG else False

    @staticmethod
    def get_utilization(physical_graph: EnhancedDiGraph, allocated_virtual_nodes, allocated_virtual_links) -> (
    float, float):
        total_node_capacity = physical_graph.get_total_node_capacity()
        total_link_capacity = physical_graph.get_total_link_capacity()
        node_utilization = round(allocated_virtual_nodes / total_node_capacity, 2)
        link_utilization = round(allocated_virtual_links / total_link_capacity, 2)
        return node_utilization, link_utilization


class PrintAllocationGraph:
    def __init__(self, allocation: AllocationGraph, run_number, exp_name):
        self.allocation = allocation
        self.run_number = run_number
        self.exp_name = exp_name
        self.path_shares = []

    def _set_allocation_graph_node_colors(self, labels) -> []:
        colors = []
        added_nodes = []
        for node in self.allocation.allocation_graph.nodes:
            if labels[node] in added_nodes:
                print(f"Node {labels[node]} already added")
                continue
            added_nodes.append(labels[node])
            if len(node) == 2:
                if node == self.allocation.root:
                    colors.append("#ff2200")
                else:
                    colors.append("#7870b4")
            elif len(node) == 3:
                colors.append("#fac227")
            else:
                colors.append("#00db1d")
        return colors

    def _create_allocation_graph_illustration(self) -> []:
        labels = {}
        for node in self.allocation.allocation_graph.nodes:
            node_type = self.allocation.get_element_type(node)
            if self.allocation.element_is_allocated_node(node_type):
                labels[node] = f'{node[0]}, {node[1]}'
            elif self.allocation.element_is_agg_node(node_type):
                labels[node] = f'{node[0]}-{node[1]}-AGG'
            elif self.allocation.element_is_transient(node_type):
                labels[node] = f'{node[0]}-{node[1]}, {node[2]}'
            else:
                raise ValueError("Unexpected node type")
        graph = nx.relabel_nodes(self.allocation.allocation_graph, labels)
        return graph, self._set_allocation_graph_node_colors(labels)

    def _get_fig_name(self):
        return f"allocation_{self.allocation.app.app}_{self.allocation.req.assoc_node}.png"

    def _print_paths(self):
        req_demand = self.allocation.req.demand
        print(f"\n### Req id: {self.allocation.req.id}, app: {self.allocation.app.app}, "
              f"associated node: {self.allocation.req.assoc_node}, total demand: {req_demand} ###")
        for i, path in enumerate(self.path_shares):
            for ind, element in enumerate(path.path):
                if ind == 0:
                    share_portion = round(100 * (path.share / req_demand), 0)
                    print(f"\n- Path allocated share: {path.share} ({share_portion}%)")
                    continue
                print(f"{element}")

    def _log_all_paths(self):
        leafs = PathFinder.get_graph_leafs(self.allocation.allocation_graph)
        for leaf in leafs:
            for path in nx.all_simple_paths(self.allocation.allocation_graph, self.allocation.root, leaf):
                allocated_path = AllocatedPath(path, round(self._get_path_share(path), 2))
                self.path_shares.append(allocated_path)

    def _get_path_share(self, path: []) -> float:
        flow = self.allocation.find_max_flow_from_root_to_leaf(PrintAllocationGraph.last_path_element(path))
        share = flow[0]
        return share

    @staticmethod
    def last_path_element(path: []):
        return path[-1]


@dataclass
class AllocatedPath:
    path: []
    share: float


class OnlineAllocationLogger:
    ARRIVAL = "IN"
    DEPARTURE = "OUT"
    GREEDY = "GREEDY"
    PLAN = "PLAN"
    PREEMPT = "PREEMPT"

    def __init__(self, run_settings):
        self.run_settings = run_settings
        self.advanced_logging_mode = config.advanced_logging_mode
        self.event_entries = []
        self.stats_entries = []
        self.event_header = ['time', 'arrival', 'allocated', 'user_id', 'app_name', 'src', 'demand', 'nodes', 'type']

    def add_event_entry(self, time: int, arrival: bool, allocated: bool, id: str, app_name: str, src: str, demand: int,
                        nodes: {}, greedy=None, preempted=False):
        if not self.advanced_logging_mode:
            return ''
        if nodes is None:
            nodes = []
        else:
            nodes = [m for m, n in nodes.keys() if m == n]
        if preempted:
            type = OnlineAllocationLogger.PREEMPT
        elif greedy is None:
            type = ''
        elif greedy:
            type = OnlineAllocationLogger.GREEDY
        else:
            type = OnlineAllocationLogger.PLAN
        arrival = OnlineAllocationLogger.ARRIVAL if arrival else OnlineAllocationLogger.DEPARTURE
        # self.event_entries.append(f"{time},{arrival},{allocated},{id},{app_name},{src},{demand}\n")
        self.event_entries.append([time, arrival, allocated, id, app_name, src, demand, nodes, type])

    def export(self, dir):
        filename = f"{self.run_settings.branch_name}.csv"
        filepath = f"{dir}/{filename}"
        df = pd.DataFrame(self.event_entries, columns=self.event_header)
        df.to_csv(filepath, index=False)


class ReqReleaseLogger:
    def __init__(self, run_settings):
        self.run_settings = run_settings
        self.event_entries = []
        self.event_header = ['Run Number', 'ID', 'Arrival', 'Departure', 'Demand', 'Guaranteed', 'Allocated', 'Cost',
                             'GCost', 'NGCost', 'Preempt']

    def add_entry(self, t, request, guaranteed: bool, cost, preempt=False):
        allocated = 1 if t == request.end_time else 0
        gcost, ngcost = 0, 0
        if allocated:
            gcost = cost if guaranteed else 0
            ngcost = cost if not guaranteed else 0
        self.event_entries.append(
            [self.run_settings.run_number, request.id, request.start_time, request.end_time, request.demand, guaranteed,
             allocated, cost, gcost, ngcost, preempt])

    def post_process_cost(self, df):
        from online_handler import OnlineDataLogger, OnlineHandler
        df_acc = pd.DataFrame(self.event_entries, columns=self.event_header)
        if len(df_acc) == 0:
            print("WARNING: no logged events")
            return
        _, df_exploded_grouped = OnlineHandler.split_to_unique_timeslots(df_acc)
        unique_times = df['Time'].unique()
        df_parsed_data = []
        allocation_cost, rejection_cost, gcost, ngcost = 0, 0, 0, 0
        for t in unique_times:
            df_exp_alloc_t = self.get_df_by_allocation(df_exploded_grouped, t, True)
            df_exp_rej_t = self.get_df_by_allocation(df_exploded_grouped, t, False)
            allocation_cost, rejection_cost, gcost, ngcost = self.get_costs(df_exp_alloc_t, df_exp_rej_t,
                                                                            allocation_cost, rejection_cost, gcost, ngcost)
            data = [allocation_cost, rejection_cost]
            df_parsed_data.append(data)
        df_parsed = pd.DataFrame(df_parsed_data, columns=['Allocation Cost', 'Rejection Cost'])
        df['Allocation Cost'] = df_parsed['Allocation Cost'].astype(float).round(2)
        df['Rejection Cost'] = df_parsed['Rejection Cost'].astype(float).round(2)
        OnlineDataLogger.export(self.run_settings, df)

    def post_process_cost_from_lists(self, df, allocation_cost, rejection_cost):
        from online_handler import OnlineDataLogger
        data = []
        agg_alloc_cost = 0
        agg_rej_cost = 0
        for t in range(len(allocation_cost)):
            agg_alloc_cost += allocation_cost[t]
            agg_rej_cost += rejection_cost[t]
            data.append([agg_alloc_cost, agg_rej_cost])
        df_parsed = pd.DataFrame(data, columns=['Allocation Cost', 'Rejection Cost'])
        df['Allocation Cost'] = df_parsed['Allocation Cost'].astype(float).round(2)
        df['Rejection Cost'] = df_parsed['Rejection Cost'].astype(float).round(2)
        OnlineDataLogger.export(self.run_settings, df)


    @profile
    def post_process(self, df):
        from online_handler import OnlineDataLogger, OnlineHandler
        df_acc = pd.DataFrame(self.event_entries, columns=self.event_header)
        if len(df_acc) == 0:
            print("WARNING: no logged events")
            return
        df_grouped, df_exploded_grouped = OnlineHandler.split_to_unique_timeslots(df_acc)
        unique_times = df['Time'].unique()
        df_parsed_data = []
        allocation_cost, rejection_cost, gcost, ngcost = 0, 0, 0, 0
        g_requests, ng_requests = 0, 0
        total_alloc_requests = 0
        dealloc_requests = 0
        dealloc_next_t_requests = 0
        dealloc_next_t_demand = 0
        for t in unique_times:
            df_alloc_t = self.get_df_by_allocation(df_grouped, t, True)
            df_rej_t = self.get_df_by_allocation(df_grouped, t, False)
            df_exp_alloc_t = self.get_df_by_allocation(df_exploded_grouped, t, True)
            df_exp_rej_t = self.get_df_by_allocation(df_exploded_grouped, t, False)
            preempted_t = df_rej_t[df_rej_t['Preempt'] == True] if not df_rej_t.empty else df_rej_t
            allocated_demand = df_alloc_t['Demand'].sum() if 'Demand' in df_alloc_t.columns else 0
            allocated_active_demand = df_exp_alloc_t['Demand'].sum() if 'Demand' in df_exp_alloc_t.columns else 0
            alloc_active_requests = len(df_exp_alloc_t)
            total_alloc_requests += len(df_alloc_t)
            preempted_requests = len(preempted_t)
            allocated_requests = len(df_alloc_t) + preempted_requests
            dealloc_requests = len(preempted_t) + dealloc_next_t_requests
            dealloc_demand = preempted_t['Demand'].sum() if 'Demand' in preempted_t.columns else 0  #rejected
            dealloc_demand += dealloc_next_t_demand # plus requests that ended
            allocated_guaranteed_demand = df_alloc_t[df_alloc_t['Guaranteed']]['Demand'].sum() if not df_alloc_t.empty else 0
            allocated_non_guaranteed_demand = df_alloc_t[~df_alloc_t['Guaranteed']]['Demand'].sum() if not df_alloc_t.empty else 0
            g_requests += len(df_alloc_t[df_alloc_t['Guaranteed']]) if not df_alloc_t.empty else 0
            ng_requests += len(df_alloc_t[~df_alloc_t['Guaranteed']]) if not df_alloc_t.empty else 0
            allocation_cost, rejection_cost, gcost, ngcost = self.get_costs(df_exp_alloc_t, df_exp_rej_t,
                                                                            allocation_cost, rejection_cost, gcost, ngcost)
            runtime = df[df['Time'] == t]['Runtime'].values[0]
            next_t_end = df_exp_alloc_t[df_exp_alloc_t['Departure'] == t + 1] if not df_alloc_t.empty else pd.DataFrame()
            dealloc_next_t_requests = len(next_t_end)
            dealloc_next_t_demand = next_t_end['Demand'].sum() if not next_t_end.empty else 0

            data = [allocated_requests, allocated_demand, allocated_active_demand, alloc_active_requests,
                    total_alloc_requests, g_requests, ng_requests, allocated_guaranteed_demand,
                    allocated_non_guaranteed_demand, allocation_cost, rejection_cost, gcost, ngcost, runtime,
                    dealloc_requests, dealloc_demand, preempted_requests]
            df_parsed_data.append(data)

        df_parsed = pd.DataFrame(df_parsed_data, columns=['Allocated Requests', 'Allocated Arrived Demand',
                                                          'Allocated Active Demand', 'Allocated Active Requests',
                                                          'Total Allocated Requests',
                                                          'Alloc G Requests', 'Alloc NG Requests',
                                                          'Alloc G Demand', 'Alloc NG Demand',
                                                          'Allocation Cost', 'Rejection Cost', 'Guar Cost',
                                                          'Non-Guar Cost', 'Runtime', 'Deallocated Requests',
                                                          'Deallocated Demand', 'Preempted Requests'])
        df['Allocated Requests'] = df_parsed['Allocated Requests']
        df['Allocated Arrived Demand'] = df_parsed['Allocated Arrived Demand']
        df['Allocated Active Demand'] = df_parsed['Allocated Active Demand']
        df['Allocated Active Requests'] = df_parsed['Allocated Active Requests']
        df['Total Allocated Requests'] = df_parsed['Total Allocated Requests']
        df['Alloc G Requests'] = df_parsed['Alloc G Requests']
        df['Alloc NG Requests'] = df_parsed['Alloc NG Requests']
        df['Alloc G Demand'] = df_parsed['Alloc G Demand']
        df['Alloc NG Demand'] = df_parsed['Alloc NG Demand']
        df['Deallocated Requests'] = df_parsed['Deallocated Requests']
        df['Deallocated Demand'] = df_parsed['Deallocated Demand']
        df['Preempted Requests'] = df_parsed['Preempted Requests']
        df['Allocation Cost'] = df_parsed['Allocation Cost'].astype(float).round(2)
        df['Rejection Cost'] = df_parsed['Rejection Cost'].astype(float).round(2)
        df['Guar Cost'] = df_parsed['Guar Cost'].astype(float).round(2)
        df['Non-Guar Cost'] = df_parsed['Non-Guar Cost'].astype(float).round(2)
        df['Runtime'] = df_parsed['Runtime'].astype(float)
        OnlineDataLogger.export(self.run_settings, df)

    @staticmethod
    def get_df_by_allocation(df, t, allocated=None):
        df_t = df.get_group(t) if t in df.groups else pd.DataFrame()
        if allocated is None:
            return df_t
        allocated = 1 if allocated else 0
        return df_t[df_t['Allocated'] == allocated] if len(df_t) > 0 else pd.DataFrame()



    @staticmethod
    def get_costs(df_allocated, df_rejected, prev_allocation_cost, prev_rejection_cost, prev_gcost, previous_ngcost) -> ():
        allocation_cost = df_allocated['Cost'].sum() if not df_allocated.empty else 0
        rejection_cost = df_rejected['Cost'].sum() if not df_rejected.empty else 0
        gcost = df_allocated['GCost'].sum() if not df_allocated.empty else 0
        ngcost = df_allocated['NGCost'].sum() if not df_allocated.empty else 0
        return (allocation_cost + prev_allocation_cost, rejection_cost + prev_rejection_cost, gcost + prev_gcost,
                ngcost + previous_ngcost)