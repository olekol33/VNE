from decimal import Decimal
from functools import lru_cache
from code_testing import timer
from include.enums import OnlineExpType, SpecialNodeAttributes
from include import enums
import copy
from copy import deepcopy
import numpy as np
import include.config as config
import networkx as nx
from layout import Style as style
import time
from collections import namedtuple
import random
import math
from pathlib import Path
from file_handler import FileHandler
import os
from include.enums import UserDist, GraphAttrSource, GraphLinkCost
from dataclasses import dataclass
from application import Application
from path_finder import PathFinder
from enhanced_graph import EnhancedDiGraph


def _get_abs_yml_topology_location():
    return Path(os.path.dirname(__file__)).parent / Path(config.topology_dir) / \
        f'{config.topology_name}.yml'


def _get_abs_gml_topology_location():
    return Path(os.path.dirname(__file__)).parent / Path(config.topology_dir) / 'gml' / \
        f'{config.topology_name}.gml'


def _get_abs_edgelist_topology_location():
    return Path(os.path.dirname(__file__)).parent / Path(config.topology_dir) / 'edgelist' / \
        f'{config.topology_name}.edgelist'


def print_error_message(msg, e=None, finish=False, ):
    print(msg)
    print(style.RED, e, style.END)
    if finish:
        print('The execution is ending...')
        time.sleep(5)
        exit(0)


@dataclass
class TopologyGrid:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    step: float
    x_dim: int
    y_dim: int


class Topology:
    """
    Gets topology from file or generates it
    """
    def __init__(self, run_number: int = 0, existing_graph=None):
        print('Topology init')
        self.alpha_tier = {}
        for t in range(1, config.num_of_tiers + 1):
            if t == 1:
                self.alpha_tier[t] = 1
            else:
                self.alpha_tier[t] = self.alpha_tier[t - 1] * config.tier_scale_ratio
        self.alpha_tier_cost = {}
        self.max_tier_degree = {i: 0 for i in range(1, config.num_of_tiers + 1)}
        self.max_tier_input_cap = {i: 0 for i in range(1, config.num_of_tiers + 1)}
        self.tiers = {i: [] for i in range(1, config.num_of_tiers + 1)}
        self._invert_alpha_tier()
        self.run_number = run_number
        self.exp = None
        self.prob_sets = config.num_of_experiments if config.online_exp_type == OnlineExpType.DIFFERENT_HOTSPOTS else 1
        self.node_pop_gen = {}
        self.center_nodes = []
        self.seed = config.seed + run_number
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.graph = EnhancedDiGraph()
        self.graph_instance = namedtuple('graph_instance', ['cap', 'cost'])

        self.get_physical_graph(existing_graph)

        self.ordered_edge_nodes = []
        self.node_index_map = {}
        self.shuffle_edge_node_list()
        self._create_node_index_map()
        self.set_node_popularity()
        self.generate_topology_report()

    def _invert_alpha_tier(self):
        """ Reverse the order of keys"""
        tiers = list(self.alpha_tier.keys())
        for ind, value in enumerate(reversed(self.alpha_tier.values())):
            self.alpha_tier_cost[tiers[ind]] = value

    def shuffle_edge_node_list(self):
        edge_tier = 1
        self.ordered_edge_nodes = self.tiers[edge_tier]
        random.shuffle(self.ordered_edge_nodes)
        for ind, node in enumerate(self.ordered_edge_nodes):
            try:
                self.graph.nodes[node]['id'] = ind
            except KeyError:
                print(f"Node {node} does not exist in graph")

    def _create_node_index_map(self):
        self.node_index_map = {}
        for ind, node in enumerate(self.ordered_edge_nodes):
            self.node_index_map[node] = ind

    def get_physical_graph(self, existing_graph):
        if existing_graph:
            self.graph = existing_graph
        else:
            self.load_topology_from_file()
        self._process_topology()

    def load_topology_from_file(self):
        topology_location = _get_abs_yml_topology_location()
        if not topology_location.exists():
            if _get_abs_gml_topology_location().exists():
                self._read_gml(_get_abs_gml_topology_location())
            elif _get_abs_edgelist_topology_location().exists():
                self._read_edgelist(_get_abs_edgelist_topology_location())
            else:
                raise FileNotFoundError(f"Topology file not found: {topology_location}")
        else:
            with open(topology_location, 'r') as f:
                import yaml
                yaml_dict = yaml.load(f, Loader=yaml.Loader)
                self._convert_yml_to_graph(yaml_dict)
        self.remove_self_links()

    def _read_gml(self, gml_file):
        path, filename = os.path.split(gml_file)
        filename = filename.split('.')[0]
        unique_gml = None
        try:
            self.graph = nx.read_gml(gml_file)
            self.remove_hyphens()
        except Exception as ex:
            if "duplicated" in str(ex) and "node" in str(ex):
                unique_gml = self._create_unique_node_labels_gml(gml_file)
            elif "edge" in str(ex) and "undefined" in str(ex):
                unique_gml = self._remove_undefined_edges(gml_file)
            else:
                raise ex
            if ex is None:
                self.graph = EnhancedDiGraph(self.graph)
                return
        try:
            if unique_gml is None:
                unique_gml = self._create_unique_node_labels_gml(gml_file)
            self.graph = nx.parse_gml(unique_gml)
            self.remove_hyphens()
        except Exception as ex:
            if "duplicated" in str(ex) and "edge" in str(ex):
                unique_gml = self._create_unique_edges_gml(unique_gml)
        with open(Path(path) / str(filename + "_revised.gml"), 'w') as f:
            f.write(unique_gml)
        try:
            self.graph = nx.parse_gml(unique_gml)
            self.remove_hyphens()
        except Exception as ex:
            print_error_message("Error while parsing gml file", ex, finish=True)
        self.graph = EnhancedDiGraph(self.graph)

    def _read_edgelist(self, file):
        self.graph = nx.read_edgelist(file, create_using=nx.DiGraph, nodetype=str, data=(("distance", float),))
        np.random.seed(config.seed)
        for n in self.graph.nodes:
            self.graph.nodes[n]['Latitude'] = np.random.uniform(0, 1)
            self.graph.nodes[n]['Longitude'] = np.random.uniform(0, 1)
        self.graph = EnhancedDiGraph(self.graph)

    def _create_unique_node_labels_gml(self, gml_file):
        labels = set()
        output = []
        with open(gml_file, 'r') as f:
            for line in f:
                if 'label' in line:
                    label = line.strip().replace('"', '').split('label')[-1].lstrip()
                    if label in labels:
                        label = label + '_' + str(len(labels))
                    labels.add(label)
                    output.append('label "' + label + '"')
                else:
                    output.append(line.strip())
        return '\n'.join(output)

    def _remove_undefined_edges(self, gml_file):
        """Source and target in gml file are identified by id"""
        """For each edge in gml file, if source of target don't exist, remove edge"""
        nodes = set()
        output = []
        edge_output = []
        in_edge = False
        write_edge = True
        with open(gml_file, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith('id ') and not in_edge:
                    node = int(stripped_line.split('id ')[-1].lstrip())
                    nodes.add(node)
                    output.append(stripped_line)
                elif 'edge [' in stripped_line or in_edge:
                    in_edge = True
                    edge_output.append(stripped_line)
                    if stripped_line.startswith('source') or stripped_line.startswith('target'):
                        node = int(stripped_line.split(' ')[-1])
                        if node not in nodes:
                            if write_edge:
                                write_edge = False
                            continue
                    if stripped_line == ']':
                        in_edge = False
                        if write_edge:
                            output.extend(edge_output)
                        else:
                            write_edge = True
                        edge_output = []
                else:
                    output.append(stripped_line)
        return '\n'.join(output)

    def _create_unique_edges_gml(self, unique_gml):
        edges = set()
        output = []
        edge_output = []
        in_edge = False
        write_edge = False
        source = target = ""
        for line in unique_gml.split('\n'):
            stripped_line = line.strip()
            if stripped_line == 'edge [':
                in_edge = True
                source = target = ""
                edge_output = []
            elif in_edge and stripped_line.startswith('source'):
                source = stripped_line.split(' ')[-1]
            elif in_edge and stripped_line.startswith('target'):
                target = stripped_line.split(' ')[-1]
            elif in_edge and stripped_line == ']':
                in_edge = False
                edge = tuple(sorted([source, target]))
                edge_output.append(stripped_line)
                if edge in edges:
                    continue
                write_edge = True
                edges.add(edge)
            if not in_edge:
                if write_edge:
                    output.extend(edge_output)
                    write_edge = False
                else:
                    output.append(stripped_line)
            else:
                edge_output.append(stripped_line)
        return '\n'.join(output)

    def _set_gpu_nodes(self):
        if not config.gpu_nodes:
            return
        gpu_nodes = self._get_gpu_nodes()
        for node in gpu_nodes:
            self._create_gpu_node(node)

    def _get_gpu_nodes(self):
        # core_nodes = self.tiers[len(self.tiers)] + self.tiers[len(self.tiers)-1]
        core_nodes = self.tiers[len(self.tiers)]
        edge_nodes = self.tiers[1]
        gpu_nodes = core_nodes
        num_gpu_edge_nodes = math.ceil(config.edge_nodes_with_gpu_share * len(edge_nodes))
        gpu_edge_nodes = np.random.choice(edge_nodes, num_gpu_edge_nodes, replace=False)
        gpu_nodes.extend(gpu_edge_nodes)
        return gpu_nodes

    def _create_gpu_node(self, node):
        node_gpu = node + '_gpu'
        self.graph.add_node(node_gpu)
        self.graph.nodes[node_gpu][SpecialNodeAttributes.GPU.value] = True
        self.graph.set_node_tier(node_gpu, 0)
        link_cost = config.link_cost / config.general_cost_division_factor
        for attr in self.graph.nodes[node]:
            # if attr == 'shareCost':
                # factor = config.gpu_multiplier
                # factor = Decimal('0.5')
            if attr == 'shareCap':
                factor = Decimal('0.75')
            else:
                factor = 1
            # factor = 1
            self.graph.nodes[node_gpu][attr] = self.graph.nodes[node][attr] * factor
        for src, dst in [(node, node_gpu), (node_gpu, node)]:
            self.graph.add_edge(src, dst)
            self.graph.set_link_cost(src, dst, link_cost)
            self.graph.set_link_capacity((src, dst), self.graph.nodes[node_gpu]['shareCap'] * config.gpu_multiplier ** 2)



    def _set_graph_attributes(self):
        # self._remove_nodes_without_coordinates()
        self._remove_degree_zero_nodes()
        # self._check_upstream_flow()
        self._set_graph_capacities()
        self._set_fixed_graph_costs()
        self._set_gpu_nodes()

    def _remove_nodes_without_coordinates(self):
        for node in list(self.graph.nodes):
            if 'Latitude' not in self.graph.nodes[node] or 'Longitude' not in self.graph.nodes[node]:
                self.graph.remove_node(node)
                print(f"Removed node {node} without coordinates")

    def _remove_degree_zero_nodes(self):
        for node in list(self.graph.nodes):
            if self.graph.degree(node) == 0:
                self.graph.remove_node(node)
                print(f"Removed node {node} with degree 0")

    def _set_graph_costs_by_distance(self):
        """
        Edge (u, v) cost equal distance between nodes times a factor.
        If 0, sum of node costs equals sum of link costs
        """
        max_link_cost, min_link_cost = self._set_link_costs_by_distance()
        self._set_node_costs(max_link_cost)

    def _set_fixed_graph_costs(self):
        self._set_fixed_link_costs()
        self._set_fixed_node_costs()

    def _set_fixed_link_costs(self):
        cost = config.link_cost / config.general_cost_division_factor
        for u, v in self.graph.edges:
            self.graph.set_link_cost(u, v, cost)

    def _set_link_costs_by_distance(self) -> (float, float):
        """
        Link cost is a product of its distance alpha_tier
        Returns
        -------
        Sum of all link costs
        """
        max_link_cost = 0
        min_link_cost = float('inf')
        for u, v in self.graph.edges:
            if 'distance' not in self.graph.edges[u, v]:
                self.graph.set_link_distance(u, v, self._calculate_edge_distance(u, v))
            cost = self.graph.get_link_distance(u, v) * self.alpha_tier_cost[self.graph.get_link_tier(u, v)] * \
                   config.link_cost / config.general_cost_division_factor
            if cost == 0:
                raise ValueError(f"Distance of edge {u}-{v} is 0")
            max_link_cost = max(max_link_cost, cost)
            min_link_cost = min(min_link_cost, cost)
        for u, v in self.graph.edges:
            self.graph.set_link_cost(u, v, max_link_cost)
        return max_link_cost, min_link_cost

    def _set_fixed_node_costs(self):
        alpha_tier_cost = {1: 50, 2: 10, 3: 1}
        for node in self.graph.nodes:
            cost_factor = np.random.uniform(0.5, 1.5)
            cost = alpha_tier_cost[self.graph.get_node_tier(node)] * cost_factor / config.general_cost_division_factor
            self.graph.set_node_cost(node, cost)

    def _set_node_costs(self, max_link_cost):
        """Node cost should be tied to link costs"""
        for node in self.graph.nodes:
            cost_factor = np.random.uniform(2 / config.tier_scale_ratio, config.tier_scale_ratio / 2)
            cost = max_link_cost * self.alpha_tier_cost[self.graph.get_node_tier(node)] * cost_factor
            self.graph.set_node_cost(node, cost)

    def _get_basic_node_cost_from_equation(self, total_cost: float) -> float:
        """ node cost is derived from a sum of products of node_cost * alpha_tier * num_of_nodes_jn_tier"""
        import sympy as sp
        x = sp.symbols('x')
        equation = total_cost
        for tier in self.tiers.keys():
            const = len(self.tiers[tier]) * self.alpha_tier_cost[tier]
            equation = equation - const * x
        equation = sp.Eq(equation, 0)
        solution = sp.solve(equation, x)
        return solution[0]

    def _get_avg_graph_num_of_hops(self):
        total_hops = 0
        for node1 in self.graph.get_nodes():
            hops = 0
            for node2 in self.graph.get_nodes():
                hops += nx.shortest_path_length(self.graph, node1, node2)
            avg_node_hops = hops / self.graph.get_num_of_nodes()
            total_hops += avg_node_hops
        return total_hops / self.graph.get_num_of_nodes()

    def _set_graph_capacities(self):
        """
        tier_node_max_degree: Max degree of a node in that tier
        link_tier: is the higher tier between nodes m, n
        alpha_tier = scaling parameter between tiers. 3 ratio between tiers.
        link_capacity = C  * alpha_tier
        node_capacity = tier_node_max_degree * link_capacity

        """
        for edge in self.graph.edges:
            original_cap = self._get_link_capacity(edge)
            self.graph.set_link_capacity(edge, original_cap)
        for node in self.graph.nodes:
            tier = self.graph.get_node_tier(node)
            cap = config.base_node_capacity * self.alpha_tier[tier]
            self.graph.set_node_capacity(node, cap)

    def _get_link_capacity(self, edge):
        link_tier = self._get_link_tier(edge)
        self.graph.set_link_tier(edge[0], edge[1], link_tier)
        original_cap = config.base_link_capacity * self.alpha_tier[link_tier]
        return original_cap

    def _get_link_tier(self, edge) -> int:
        """
        Returns the tier of the link between u and v
        """
        u_tier = self.graph.get_node_tier(edge[0])
        v_tier = self.graph.get_node_tier(edge[1])
        return max(u_tier, v_tier)

    def _convert_yml_to_graph(self, yaml_dict):
        for node in yaml_dict['nodes'].values():
            node_name = node['label'].strip()
            attributes = {}
            if 'type' in node:
                attributes['type'] = node['type']
            if 'prob' in node:
                attributes['prob'] = node['prob']
            attributes['Latitude'] = node['Latitude']
            attributes['Longitude'] = node['Longitude']
            self.graph.add_node(node_name)
            for attr in attributes:
                self.graph.nodes[node_name][attr] = attributes[attr]
        for edge in yaml_dict['edges']:
            attributes = {}
            src_node = yaml_dict['nodes'][edge[0]]['label'].strip()
            dst_node = yaml_dict['nodes'][edge[1]]['label'].strip()
            attributes['distance'] = self._calculate_edge_distance(src_node, dst_node)
            self.graph.add_edge(src_node, dst_node)
            for attr in attributes:
                self.graph.edges[src_node, dst_node][attr] = attributes[attr]

    def _add_remove_bidirectional_links(self):
        if config.bidirectional_links:
            edges = list(self.graph.edges(data=True))
            for u, v, edge_data in edges:
                if (v, u) not in edges:
                    self.graph.add_edge(v, u, **edge_data)
        else:
            for u, v in list(self.graph.edges):
                if self.graph.get_node_tier(u) > self.graph.get_node_tier(v):
                    num_u_successors = len(list(self.graph.successors(u)))
                    if num_u_successors > 1 and self.graph.degree(v) > 1:
                        self.graph.remove_edge(u, v)
                        print(f"WARNING: Removed reverse edge {u}-{v}")
                    elif self.graph.get_node_tier(u) == config.num_of_tiers:
                        self.graph.remove_edge(u, v)
                        print(f"WARNING: Removed reverse edge {u}-{v}")
                else:
                    num_u_successors = len(list(self.graph.successors(u)))
                    if self.graph.get_node_tier(u) == 1 and self.graph.get_node_tier(
                            v) == config.num_of_tiers and num_u_successors > 1:
                        self.graph.remove_edge(u, v)
                        print(f"WARNING: Removed reverse edge {u}-{v}")

    def remove_hyphens(self):
        """Replaces hyphens with space in node names"""
        for node in list(self.graph.nodes):
            if '-' in node:
                new_node = node.replace('-', ' ')
                self.graph = nx.relabel_nodes(self.graph, {node: new_node}, copy=False)

    def remove_self_links(self):
        """Assume to links from node to itself in graph"""
        found_self_links = False
        for n in self.graph.nodes:
            if (n, n) in self.graph.edges:
                found_self_links = True
                self.graph.remove_edge(n, n)
        if found_self_links:
            print("Warning: Found and removed self links in topology")

    def _process_topology(self) -> None:
        self._categorize_nodes_into_tiers()
        self._add_remove_bidirectional_links()
        self._set_graph_attributes()

    def generate_topology_report(self, graph=None, is_opt=False):
        import pandas as pd
        if graph is None:
            graph = self.graph
        if is_opt:
            topo_dir = FileHandler.get_run_dir('opt')
        elif self.exp is not None:
            topo_dir = FileHandler.get_run_dir(run=self.run_number) / self.exp
        else:
            topo_dir = FileHandler.get_run_dir()
        nodes = [n for n in graph.nodes]
        edges = [e for e in graph.edges]

        data = []
        for node in nodes:
            capacity = graph.get_node_capacity(node)
            cost = graph.get_node_cost(node)
            tier = graph.get_node_tier(node)
            degree = graph.degree(node)
            data.append([node, '-1', capacity, cost, degree, tier, '-1'])
        for m, n in edges:
            capacity = graph.get_link_capacity(m, n)
            cost = graph.get_link_cost(m, n)
            tier_m = graph.get_node_tier(m)
            tier_n = graph.get_node_tier(n)
            data.append([m, n, capacity, cost, '-1', tier_m, tier_n])
        df = pd.DataFrame(data, columns=['m', 'n', 'Capacity', 'Cost', 'Degree', 'Tier_m', 'Tier_n'])
        file_path = topo_dir / 'topology_elements.csv'
        df.to_csv(file_path, sep=',', encoding='utf-8', index=False)

    def _check_upstream_flow(self):
        if config.bidirectional_links:
            return
        else:
            for m in self.graph.nodes:
                successors = list(self.graph.successors(m))
                m_tier = self.graph.get_node_tier(m)
                if m_tier == len(self.tiers):
                    continue
                if len(successors) == 0:
                    self._reverse_link_direction(m)
                    successors = list(self.graph.successors(m))
                max_n_tier = 0
                for n in successors:
                    max_n_tier = max(max_n_tier, self.graph.get_node_tier(n))
                if m_tier > max_n_tier:
                    raise ValueError(f"Node {m} no upstream link")

    def _reverse_link_direction(self, m):
        predecessors = list(self.graph.predecessors(m))
        for n in predecessors:
            self.graph.remove_edge(n, m)
            self.graph.add_edge(m, n)

    def _categorize_nodes_into_tiers(self):
        if self._categorize_nodes_into_tiers_by_type():
            return
        else:
            self._categorize_nodes_into_tiers()

    def _categorize_nodes_into_tiers(self):
        if config.use_ratios_to_assign_nodes_to_tiers:
            self._categorize_nodes_into_tiers_by_ratios()
        else:
            self._categorize_nodes_into_tiers_by_classification()

    def _categorize_nodes_into_tiers_by_ratios(self):
        if config.num_of_tiers != 3:
            raise ValueError("Setting num of tiers by degree is only implemented for 3 tiers")
        sorted_nodes = sorted(self.graph.nodes, key=lambda x: self.graph.degree(x), reverse=True)
        num_core_nodes = math.ceil(len(sorted_nodes) / config.nodes_to_core_ratio)
        num_transport_nodes = num_core_nodes * config.transport_to_core_ratio
        for i in range(num_core_nodes):
            self._set_node_tier(sorted_nodes[i], config.num_of_tiers)
        for core_node in self.tiers[config.num_of_tiers]:
            trans_count = 0
            predecessors = list(self.graph.predecessors(core_node))
            predecessors = sorted(predecessors, key=lambda x: self.graph.degree(x), reverse=True)
            for i in range(len(predecessors)):
                if self.graph.get_node_tier(predecessors[i]) is None:
                    self._set_node_tier(predecessors[i], 2)
                    trans_count += 1
                    if trans_count == config.transport_to_core_ratio:
                        break
            num_transport_nodes -= trans_count
        for node in self.graph.nodes:
            if self.graph.get_node_tier(node) is None:
                self._set_node_tier(node, 1)
        self._ensure_edge_to_transport_to_core_links()

    def _categorize_nodes_into_tiers_by_classification(self):
        """Perform Jenks Natural Breaks classification
        The method seeks to reduce the variance within classes and maximize the variance between classes"""
        from jenkspy import jenks_breaks
        node_degrees = [self.graph.degree(node) for node in self.graph.nodes]
        breaks = jenks_breaks(node_degrees, config.num_of_tiers)
        for node in self.graph.nodes:
            tier = self._get_node_tier_from_classification(node, breaks)
            self._set_node_tier(node, tier)
            degree = self.graph.degree(node) / 2 if config.bidirectional_links else self.graph.degree(node)
            self.max_tier_degree[tier] = max(self.max_tier_degree[tier], degree)
        self._ensure_has_core_node()
        self._ensure_edge_to_transport_to_core_links()

    def _set_node_tier(self, node, tier):
        self.graph.set_node_tier(node, tier)
        tier_nodes = self.tiers[tier]
        tier_nodes.append(node)
        self.tiers[tier] = tier_nodes

    def _ensure_has_core_node(self):
        if len(self.tiers[config.num_of_tiers]) == 0:
            max_degree = 0
            max_degree_node = None
            for node in self.graph.nodes:
                if self.graph.degree(node) > max_degree:
                    max_degree = self.graph.degree(node)
                    max_degree_node = node
            tier = self.graph.get_node_tier(max_degree_node)
            self.tiers[config.num_of_tiers].append(max_degree_node)
            self.graph.set_node_tier(max_degree_node, config.num_of_tiers)
            self.tiers[tier].remove(max_degree_node)

    def _ensure_edge_to_transport_to_core_links(self):
        if config.bidirectional_links or config.num_of_tiers != 3:
            return
        self._ensure_has_transport_to_core_edges()
        for core_node in self.tiers[config.num_of_tiers]:
            predecessors = list(self.graph.predecessors(core_node))
            for ind, node in enumerate(predecessors):
                if self.graph.get_node_tier(node) > 1:
                    continue
                if len(set(self.graph.predecessors(node)).union(self.graph.successors(node))) == 1:
                    nearest_node = self.graph.get_nearest_node(node, edge_or_transport=True)
                    self.graph.add_edge(node, nearest_node)
                    print(f"Added edge {node}-{nearest_node}")
                self.graph.remove_edge(node, core_node)
                print(f"Removed edge {node}-{core_node}")

    def _ensure_has_transport_to_core_edges(self):
        """If only link to core is from edge, set one edge node as transport"""
        if config.bidirectional_links or config.num_of_tiers != 3:
            return
        for core_node in self.tiers[config.num_of_tiers]:
            predecessors = list(self.graph.predecessors(core_node))
            predecessors = sorted(predecessors, key=lambda x: self.graph.degree(x), reverse=True)
            for pred in predecessors:
                if pred in self.tiers[1]:
                    for trans in self.tiers[2]:
                        if not self.graph.node_connected_to_core_node(trans):
                            self._swap_tiers(pred, trans)
                            break

    def _swap_tiers(self, edge_node, trans_node):
        self.graph.set_node_tier(edge_node, 2)
        self.tiers[1].remove(edge_node)
        self.tiers[2].append(edge_node)
        self.tiers[2].remove(trans_node)
        self.tiers[1].append(trans_node)
        self.graph.set_node_tier(trans_node, 1)
        print(f"Swapped tier between {edge_node} and {trans_node}")

    def _categorize_nodes_into_tiers_by_type(self):
        tiers = list(self.tiers.keys())
        for node in self.graph.nodes:
            tier = self.graph.get_node_type(node)
            if tier is None:
                return False
            if tier in tiers:
                tier_nodes = self.tiers[tier]
                tier_nodes.append(node)
                self.tiers[tier] = tier_nodes
                self.graph.set_node_tier(node, tier)
                self.max_tier_degree[tier] = max(self.max_tier_degree[tier], self.graph.degree(node))
            else:
                self.max_tier_degree = {i: 0 for i in range(1, config.num_of_tiers + 1)}
                self.tiers = {i: [] for i in range(1, config.num_of_tiers + 1)}
                return False
        return True

    def _get_node_tier_from_classification(self, node, breaks):
        degree = self.graph.degree(node)
        for i in range(1, len(breaks)):
            if degree <= breaks[i]:
                return i
        raise Exception("Could not find tier for node: " + node)

    def _calculate_edge_distance(self, src_node, dst_node) -> float:
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        lon1 = self.graph.nodes[src_node]['Longitude']
        lat1 = self.graph.nodes[src_node]['Latitude']
        lon2 = self.graph.nodes[dst_node]['Longitude']
        lat2 = self.graph.nodes[dst_node]['Latitude']

        earth_radius_km = 6367
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        distance = earth_radius_km * c
        return distance

    @timer("set_node_popularity")
    def set_node_popularity(self):
        if config.online_mode and not config.export_requests:
            if config.edge_node_popularity == UserDist.ZIPF:
                self.center_nodes = []
            else:
                self.center_nodes = LogNormalPopularityGenerator.create_max_dist_center_nodes(self.seed,
                                                                                              self.graph.nodes,
                                                                                              self.graph.edges,
                                                                                              config.lognormal_node_popularity_hotspots)
        elif config.edge_node_popularity == UserDist.LOGNORMAL or config.edge_node_popularity == UserDist.NORMAL:
            if config.online_mode and config.export_requests and config.online_exp_type != OnlineExpType.SAME_HOTSPOTS:
                self.node_pop_gen = {}
                for set_id in range(self.prob_sets):
                    seed = config.seed + set_id
                    self.node_pop_gen[seed] = LogNormalPopularityGenerator(self, seed)
                    self.center_nodes = self.node_pop_gen[seed].center_nodes
            else:
                self.node_pop_gen[self.run_number] = LogNormalPopularityGenerator(self, self.run_number)
                self.center_nodes = self.node_pop_gen[self.run_number].center_nodes
        elif config.edge_node_popularity == UserDist.ZIPF:
            for set_id in range(self.prob_sets):
                seed = config.seed + set_id
                self.node_pop_gen[seed] = ZipfPopularityGenerator(self, seed)
                if config.online_exp_type == OnlineExpType.DIFFERENT_HOTSPOTS:
                    hotspot = self.node_pop_gen[seed].edge_node_list[np.argmax(self.node_pop_gen[seed].node_probs)]
                    print(f"Different hotspots set {set_id} - hotspot: {hotspot}")
        elif config.edge_node_popularity == UserDist.UNIFORM:
            for set_id in range(self.prob_sets):
                seed = config.seed + set_id
                self.node_pop_gen[seed] = UniformPopularityGenerator(self, seed)
        else:
            raise ValueError(f"Unexpected ordered_edge_nodes")

    @staticmethod
    def draw_node_by_popularity_cases(num_edge_nodes, node_pop_gen, node_probs):
        if config.edge_node_popularity == UserDist.UNIFORM:
            return np.random.randint(num_edge_nodes)
        elif config.edge_node_popularity == UserDist.LOGNORMAL or config.edge_node_popularity == UserDist.NORMAL:
            return node_pop_gen.draw_node_by_popularity(node_probs)
        if config.edge_node_popularity == UserDist.ZIPF:
            raise ValueError(f"Unexpected ordered_edge_nodes")
        else:
            raise ValueError(f"Unexpected ordered_edge_nodes")

    def get_node_probs(self, seed=None):
        if seed is None:
            seed = config.seed
        if config.edge_node_popularity == UserDist.UNIFORM:
            num_edge_nodes = len(self.ordered_edge_nodes)
            return [Decimal('1') / Decimal(str(num_edge_nodes))] * num_edge_nodes
        elif config.edge_node_popularity == UserDist.INPUT:
            num_edge_nodes = []
            for node in self.ordered_edge_nodes:
                num_edge_nodes.append(self.graph.get_node_prob(node))
            return num_edge_nodes
        else:
            return self.node_pop_gen[seed].get_node_probs()

    def get_node_list_by_popularity(self, seed):
        nodes = []
        probs = {prob: self.ordered_edge_nodes[ind] for ind, prob in enumerate(self.get_node_probs(seed))}
        sorted_probs = sorted(probs.keys(), reverse=True)
        for prob in sorted_probs:
            nodes.append(probs[prob])
        return nodes


class GraphSizeScaler:
    def __init__(self, graph: EnhancedDiGraph, apps: {}, requests: [], nrf, erf):
        self.graph = copy.deepcopy(graph)
        self.requests = requests
        self.nrf = Decimal(str(nrf))
        self.erf = Decimal(str(erf))
        self.apps = apps
        app_sizes = self._create_app_size_dict()

        node_scaling_factor = self._get_node_size_scaling_factor(app_sizes)
        self._scale_node_sizes(node_scaling_factor)
        link_scaling_factor = self._get_link_size_scaling_factor(app_sizes)
        self._scale_link_sizes(link_scaling_factor)

    def _scale_node_sizes(self, node_scaling_factor):
        for node in self.graph.nodes:
            self.graph.scale_node_size(node, node_scaling_factor)

    def _scale_link_sizes(self, link_scaling_factor):
        for m, n in self.graph.edges:
            self.graph.scale_link_size(m, n, link_scaling_factor)

    def _create_app_size_dict(self) -> {}:
        app_sizes = {}
        for app_name in self.apps.keys():
            app = self.apps[app_name]
            app_sizes[app_name] = GraphSizeScaler.get_app_sizes_nodes_links(app)
        return app_sizes

    @staticmethod
    @lru_cache(maxsize=10)
    def get_app_sizes_nodes_links(app: Application):
        """Gets size of nodes and links in an application
        If choice, take main """
        path_nodes = []
        path_links = []
        main_alternative = app.get_main_alternative()
        if len(app.tree_splitters) > 0:
            paths_from_all_choice_nodes = {}
            for choice_node in app.choice_nodes:
                choice_paths = []
                links_to_choice = PathFinder.get_links_between_src_dst(main_alternative, config.user_func, choice_node)
                for branch_root_node in main_alternative.successors(choice_node):
                    links_from_choice = [(choice_node, branch_root_node)]
                    links_from_choice += PathFinder.get_links_from_root_node(main_alternative, branch_root_node)
                    choice_paths.append(links_to_choice + links_from_choice)
                paths_from_all_choice_nodes[choice_node] = choice_paths
            path_nodes, path_links = GraphSizeScaler.get_all_tree_combinations(paths_from_all_choice_nodes)
            num_of_paths = len(path_nodes)
        else:
            leafs = list(PathFinder.get_graph_leafs(main_alternative))
            num_of_paths = len(leafs)
            for leaf in leafs:
                path_nodes.append(next(nx.all_simple_paths(main_alternative, config.user_func, leaf)))
                path_links.append(next(nx.all_simple_edge_paths(main_alternative, config.user_func, leaf)))
        total_node_size = []
        total_link_size = []
        for p in range(num_of_paths):
            total_node_size.append(0)
            total_link_size.append(0)

            for node in path_nodes[p]:
                total_node_size[p] += Decimal(str(main_alternative.nodes[node].get('size', 0)))  #notice: no multiplier for node size
            for link in path_links[p]:
                total_link_size[p] += Decimal(str(main_alternative.edges[link].get('size', 0)))   #notice: no multiplier
                                      # DefaultMultiplierDict.get_app_link_attribute(app, link)
        return round(np.mean(total_node_size), 2), round(np.mean(total_link_size), 2)

    def _get_node_size_scaling_factor(self, app_sizes: {}):
        """Scale by nrf, physical node sizes, and virtual nodes sizes and user demand"""
        total_node_capacity = self.graph.get_total_node_capacity()
        if len(app_sizes) == 1:
            nodes_sizes, link_sizes = list(app_sizes.values())[0]
            total_function_allocation_size = sum(request.demand for request in self.requests) * nodes_sizes
            pass
        elif config.app_popularity == enums.AppDist.UNIFORM:
            total_nodes_sizes = 0
            total_link_sizes = 0
            for sizes in app_sizes.values():
                nodes_sizes, link_sizes = sizes
                total_nodes_sizes += nodes_sizes
                total_link_sizes += link_sizes
            total_function_allocation_size = (sum(request.demand for request in self.requests) *
                                              (Decimal(str(total_nodes_sizes)) / Decimal(str(len(app_sizes)))))
        else:
            raise NotImplementedError("Multiple non-uniform apps not supported")
        if total_function_allocation_size == 0:
            raise NotImplementedError("No size in app")

        return round(total_function_allocation_size / (total_node_capacity * self.nrf), 2)

    def _get_link_size_scaling_factor(self, app_sizes: {}):
        total_physical_link_capacity = self.graph.get_total_link_capacity()
        total_logical_link_allocation_size = 0
        for request in self.requests:
            nodes_sizes, link_sizes = app_sizes[request.app_name]
            total_logical_link_allocation_size += link_sizes * request.demand
        if total_logical_link_allocation_size == 0:
            raise NotImplementedError("No size in app")
        return round(total_logical_link_allocation_size / (total_physical_link_capacity * self.erf), 2)

    @staticmethod
    def get_all_tree_combinations(paths_from_all_choice_nodes: {}) -> ():
        """creates list of (nodes, links) for all combinations of tree"""
        nodes = []
        links = []
        permutations_dicts = PathFinder.create_all_combinations(paths_from_all_choice_nodes)
        for ind, path in enumerate(permutations_dicts):
            path = list(set([item for sublist in path for item in sublist]))
            links.append(path)
            path_nodes = list(set([item for pair in path for item in pair]))
            nodes.append(path_nodes)
        return nodes, links


class PopularityGenerator:
    def __init__(self, topology: Topology, run_number: int, intensity: float = None):
        if intensity is None:
            self.edge_node_uniform_probability = Decimal(str(config.edge_node_uniform_probability))
        else:
            self.edge_node_uniform_probability = 1 - Decimal(str(intensity))
        self.seed = config.seed + run_number
        self.edge_node_list = topology.ordered_edge_nodes
        self.grid_coordinated_to_node = {}
        self.nodes = topology.graph.nodes
        self.edges = topology.graph.edges
        self._create_node_popularity(topology)

    def _create_node_popularity(self, topology):
        raise NotImplementedError("Popularity generator not implemented")

    def _add_omitted_nodes_to_probs(self, node_list):
        for ind, node in enumerate(self.edge_node_list):
            if node not in node_list:
                self.node_probs = np.insert(self.node_probs, ind, 0)

    def _check_edge_tier_capacity(self, topology, demand_to_allocate, assumed_app_node_size,
                                  assumed_app_first_link_size,
                                  node_factor, link_factor):
        edge_capacity = float(topology.graph.get_node_tier_capacity(1)) * node_factor
        egress_higher_tier_link_cap = float(sum(topology.graph.get_link_capacity(m, n) for m, n
                                                in topology.graph.edges if
                                                topology.graph.get_node_tier(m) == 1 and
                                                topology.graph.get_node_tier(n) > 1)) * link_factor
        demand_to_allocate -= edge_capacity / assumed_app_node_size
        if demand_to_allocate > 0:
            demand_to_allocate -= egress_higher_tier_link_cap / assumed_app_first_link_size
            if demand_to_allocate > 0:
                demand_to_allocate /= assumed_app_node_size
                if config.nrf >= node_factor or config.erf >= link_factor:
                    print(f"ERROR: Total demand size for assumed app sizes exceeds capacity by {demand_to_allocate}. "
                          f"Check app size assumptions and capacities. ")

    def _adjust_node_popularities(self, func_input, init_total_prob_remainder=0):
        (topology, total_demand_size, assumed_app_node_size, assumed_app_first_link_size, estimated_node_scaling_factor,
         estimated_link_scaling_factor) = func_input
        if init_total_prob_remainder == 0:
            total_prob_remainder = 0
        else:
            total_prob_remainder = init_total_prob_remainder
        for node in self.edge_node_list:
            node_cap = float(topology.graph.get_node_capacity(node)) * estimated_node_scaling_factor
            out_link_cap = float(topology.graph.get_egress_link_capacity(node, avoid_gpu=True)) * estimated_link_scaling_factor
            demand_to_allocate = total_demand_size * float(self.node_probs[self.edge_node_list.index(node)])
            demand_to_allocate -= out_link_cap / assumed_app_first_link_size
            if demand_to_allocate > 0:
                demand_to_allocate -= node_cap / assumed_app_node_size
            if demand_to_allocate > 0:
                prob_remainder = demand_to_allocate / total_demand_size
                total_prob_remainder += prob_remainder
                self.node_probs[self.edge_node_list.index(node)] -= Decimal(str(prob_remainder))
                print(f"WARNING: Reduced {node} popularity by {round(prob_remainder * 100, 3)}%")
            if demand_to_allocate < 0 < total_prob_remainder:
                prob_remainder = -demand_to_allocate / total_demand_size
                prob_remainder = min(prob_remainder, total_prob_remainder)
                total_prob_remainder -= prob_remainder
                self.node_probs[self.edge_node_list.index(node)] += Decimal(str(prob_remainder))
                print(f"WARNING: Increased {node} popularity by {round(prob_remainder * 100, 3)}%")
        if total_prob_remainder > 0:
            if init_total_prob_remainder == 0:
                self._adjust_node_popularities(func_input, total_prob_remainder)
            else:
                for ind in range(len(self.node_probs)):
                    self.node_probs[ind] += Decimal(str(total_prob_remainder / len(self.node_probs)))

        assert round(sum(self.node_probs), 3) == 1, f"Total prob: {sum(self.node_probs)}"

    def get_node_probs(self):
        return self.node_probs


class ZipfPopularityGenerator(PopularityGenerator):
    def __init__(self, topology: Topology, run_number: int, intensity: float = None):
        super().__init__(topology, run_number, intensity)

    def _create_node_popularity(self, topology):
        from scipy.stats import zipfian
        max_val = 10000
        popularities = zipfian.rvs(config.edge_popularity_zipf_alpha, max_val, size=len(topology.ordered_edge_nodes),
                                   random_state=self.seed)
        self.node_probs = [Decimal(str(n)) / Decimal(str(np.sum(popularities))) for n in popularities]


class UniformPopularityGenerator(PopularityGenerator):
    def __init__(self, topology: Topology, run_number: int, intensity: float = None):
        super().__init__(topology, run_number, intensity)

    def _create_node_popularity(self, topology):
        self.node_probs = [1 / Decimal(str(len(topology.ordered_edge_nodes))) for _ in range(len(topology.ordered_edge_nodes))]


class LogNormalPopularityGenerator(PopularityGenerator):
    def __init__(self, topology: Topology, run_number: int, intensity: float = None):
        super().__init__(topology, run_number, intensity)

    def _create_node_popularity(self, topology):
        """Creates an M X N 2D array of cells based on topology node coordinates
        and maps nodes into cells"""
        np.random.seed(self.seed)
        self.center_nodes = []
        samples = []
        if config.lognormal_node_popularity_hotspots == 0:
            nodes = list(self.nodes)
            center_node = random.choice(nodes)
            while center_node not in topology.tiers[1]:
                center_node = random.choice(nodes)
            print("Center node: ", center_node)
            self.center_nodes.append(center_node)
        else:
            self.center_nodes = self.create_max_dist_center_nodes(topology.seed, self.nodes, self.edges,
                                                                  config.lognormal_node_popularity_hotspots)
        for node in self.center_nodes:
            node = self.nodes[node]
            node_samples = self._create_multivariate_log_normal_popularity(node)
            if len(samples) == 0:
                samples = node_samples
            else:
                samples = np.concatenate((samples, node_samples))
        self._create_cell_probabilities(samples, topology)

    @staticmethod
    def create_max_dist_center_nodes(seed, nodes, edges, num_nodes):
        center_nodes = []
        np.random.seed(seed)
        nodes = list(nodes)
        # edges = list(topology.graph.nodes)
        center_node = np.random.choice(nodes)
        center_nodes.append(center_node)
        nodes.remove(center_node)
        while len(center_nodes) < num_nodes:
            center_node = LogNormalPopularityGenerator._get_max_distance_node(nodes, edges, center_nodes)
            center_nodes.append(center_node)
            nodes.remove(center_node)
        return center_nodes

    @staticmethod
    def get_max_cap_center_nodes(graph, num_nodes):
        return LogNormalPopularityGenerator.get_max_capacity_nodes(graph)[:num_nodes]

    @staticmethod
    def get_max_capacity_nodes(graph):
        nodes = list(graph.nodes())
        nodes_cap = {}
        for node in nodes:
            if graph.get_node_tier(node) != 1:
                continue
            node_cap = graph.get_node_capacity(node)
            egress_link_cap = graph.get_egress_link_capacity(node)
            nodes_cap[node] = node_cap + egress_link_cap
        sorted_nodes = sorted(nodes_cap, key=nodes_cap.get, reverse=True)
        return sorted_nodes

    @staticmethod
    def _get_max_distance_node(nodes, edges, center_nodes):
        max_distance = 0
        max_node = None
        for node in nodes:
            min_distance = min(
                EnhancedDiGraph.get_distance(nodes, edges, node, center_node) for center_node in center_nodes)
            if min_distance > max_distance:
                max_distance = min_distance
                max_node = node
        return max_node

    def _set_node_grid_cells(self, topology_grid: TopologyGrid):
        for node in self.nodes:
            x = self.nodes[node]['Latitude']
            y = self.nodes[node]['Longitude']
            x_cell = math.floor((x - topology_grid.min_x) / topology_grid.step)
            y_cell = math.floor((y - topology_grid.min_y) / topology_grid.step)
            self.nodes[node]['cell'] = (x_cell, y_cell)

    def _create_multivariate_log_normal_popularity(self, center_node):
        multiply_num_users_for_samples = 1
        num_samples = config.num_of_offline_requests * multiply_num_users_for_samples
        mean_x, mean_y = center_node['Latitude'], center_node['Longitude']
        mean = np.array([mean_x, mean_y])
        sigma1 = 0.2
        sigma2 = 0.4
        rho = np.random.uniform(-1, 1)
        if config.edge_node_popularity == UserDist.LOGNORMAL:
            cov = np.array([[sigma1 ** 2, sigma1 * sigma2 * rho], [sigma1 * sigma2 * rho, sigma2 ** 2]])
            samples = np.random.multivariate_normal(mean, cov, num_samples, check_valid='raise')
        else:
            cov = np.array([[0.1, 0.03], [0.03, 0.1]])  # https://fabiandablander.com/statistics/Two-Properties.html
            samples = np.random.multivariate_normal(mean, cov, num_samples, check_valid='raise')
        return samples

    def _create_cell_probabilities(self, samples, topology):
        node_list = self._create_kd_tree(topology)
        node_sample_count = np.zeros(len(node_list))
        samples_array = np.array(samples)
        _, nearest_nodes = self.kd_tree.query(samples_array)
        unique_nodes, counts = np.unique(nearest_nodes, return_counts=True)
        node_sample_count[unique_nodes] += counts

        log_normal_share = 1 - Decimal(str(self.edge_node_uniform_probability))
        self.node_probs = [(Decimal(str(p)) / Decimal(str(np.sum(node_sample_count)))) * log_normal_share for p in
                           node_sample_count]
        self._add_uniform_distribution()

    def _add_uniform_distribution(self):
        uniform_dist_per_node = Decimal(str(self.edge_node_uniform_probability)) / Decimal(str(len(self.node_probs)))
        self.node_probs = [p + uniform_dist_per_node for p in self.node_probs]
        total_prob = sum(self.node_probs)
        self.node_probs = [p / total_prob for p in self.node_probs]

    def _get_nearest_node(self, x, y):
        _, node_index = self.kd_tree.query([x, y])
        return node_index

    def _is_edge_node(self, node, topology):
        return node in topology.tiers[1]

    def _create_kd_tree(self, topology):
        from scipy.spatial import KDTree
        node_list = [node for node in self.edge_node_list if self._is_edge_node(node, topology)]
        lats = np.array([self.nodes[node]['Latitude'] for node in node_list])
        lons = np.array([self.nodes[node]['Longitude'] for node in node_list])
        self.kd_tree = KDTree(np.vstack((lats, lons)).T)
        return node_list

    @staticmethod
    def draw_node_by_popularity(node_probs):
        """Draws a node based on node popularity"""
        return np.random.choice(len(node_probs), p=node_probs)

class GraphElementScaler:
    def __init__(self, graph: EnhancedDiGraph):
        self.graph = graph
        self.added_nodes = 0
        self.max_edge_cost = graph.get_max_edge_cost()
        random.seed(config.seed)

    def scale_num_of_nodes(self, element_count: float):
        if element_count == 0:
            return self.graph
        extra_nodes = element_count - self.graph.get_num_of_nodes()
        for i in range(extra_nodes):
            self._add_node(i)
        return self.graph

    def scale_num_of_links(self, element_count: float):
        if element_count == 0:
            return self.graph
        extra_links = element_count - self.graph.get_num_of_links()
        for i in range(extra_links):
            self._add_link()
        return self.graph

    def _add_link(self):
        u, v = self._get_random_not_fully_connected_nodes()
        if u is None:
            return
        top_tier_node = u if self.graph.get_node_tier(u) > self.graph.get_node_tier(v) else v
        top_tier_node_neighbor = list(self.graph.get_node_successors(top_tier_node))[0]
        link_capacity = self.graph.get_link_capacity(top_tier_node, top_tier_node_neighbor)
        link_cost = self.graph.get_link_cost(top_tier_node, top_tier_node_neighbor)
        self.graph.add_biderectional_edge(u, v, link_capacity, link_cost)

    def _get_random_not_fully_connected_nodes(self):
        all_nodes = list(self.graph.get_nodes())
        nodes_to_try = deepcopy(all_nodes)
        while True:
            m = random.choice(nodes_to_try)
            successors = self.graph.get_node_successors(m)
            unconnected_nodes = set(all_nodes) - set(successors) - {m}
            if len(unconnected_nodes) > 0:
                n = random.choice(list(unconnected_nodes))
                return m, n
            else:
                nodes_to_try.remove(m)
                if len(nodes_to_try) == 0:
                    self.added_nodes += 1
                    return None, None

    def _add_node(self, ind):
        random_node = random.choice(list(self.graph.get_nodes()))
        node_capacity = self.graph.get_node_capacity(random_node)
        node_cost = self.graph.get_node_cost(random_node)
        tier = self.graph.get_node_tier(random_node)
        new_node = f"extra_node_tier{tier}_{ind}"
        self.graph.add_node(new_node, shareCap=node_capacity, shareCost=node_cost, tier=tier, type='layer0')
        random_node_neighbor = random.choice(list(self.graph.get_node_successors(random_node)))

        link_capacity = self.graph.get_link_capacity(random_node, random_node_neighbor)
        link_cost = self.graph.get_link_cost(random_node, random_node_neighbor)
        self.graph.add_biderectional_edge(new_node, random_node, link_capacity, link_cost)
        return new_node