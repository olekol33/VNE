from decimal import Decimal
import math
import networkx as nx
from include.enums import SpecialNodeAttributes
from include import config
from functools import lru_cache


class EnhancedDiGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invalid_core_nodes = {}
        self.modified_graph = False

    @lru_cache(maxsize=None)
    def get_element_cost(self, m, n):
        if m == n:
            return self.get_node_cost(m)
        else:
            return self.get_link_cost(m, n)

    @lru_cache(maxsize=None)
    def get_max_element_cost(self, is_node: bool):
        if is_node:
            return max(nx.get_node_attributes(self, 'shareCost').values())
        else:
            return max(nx.get_edge_attributes(self, 'shareCost').values())

    @lru_cache(maxsize=None)
    def get_link_cost(self, m, n):
        return self.edges[(m, n)]['shareCost']

    @lru_cache(maxsize=None)
    def get_node_cost(self, m):
        return self.nodes[m]['shareCost']

    def get_node_tier(self, m):
        if 'tier' not in self.nodes[m]:
            return None
        return self.nodes[m]['tier']

    def get_link_tier(self, m, n):
        return self.edges[(m, n)]['tier']

    def get_link_distance(self, m, n):
        return self.edges[(m, n)]['distance']

    def set_link_distance(self, m, n, distance):
        self.edges[(m, n)]['distance'] = distance

    def get_element_capacity(self, element):
        if isinstance(element, tuple):
            return self.get_link_capacity(element)
        else:
            return self.get_node_capacity(element)

    def get_link_capacity(self, *args, avoid_gpu=False):
        if len(args) == 1 and isinstance(args[0], tuple):
            m, n = args[0]
        elif len(args) == 2:
            m, n = args
        else:
            raise Exception("Invalid input")
        if avoid_gpu and (SpecialNodeAttributes.GPU.value in self.nodes[n] or SpecialNodeAttributes.GPU.value in self.nodes[m]):
            return 0
        return self.edges[(m, n)]['shareCap']

    def remove_element(self, element):
        if isinstance(element, tuple):
            if (element[0], element[1]) in self.edges:
                self.remove_edge(*element)
        else:
            self.remove_node(element)

    def set_link_capacity(self, edge, original_cap):
        self.edges[edge]['shareCap'] = Decimal(str(original_cap))

    def set_node_capacity(self, m, original_cap):
        self.nodes[m]['shareCap'] = Decimal(str(original_cap))

    def set_link_cost(self, m, n, cost):
        self.edges[(m, n)]['shareCost'] = Decimal(str(cost))

    def set_link_tier(self, m, n, tier):
        self.edges[(m, n)]['tier'] = tier

    def set_node_tier(self, m, tier):
        self.nodes[m]['tier'] = tier

    def set_node_cost(self, m, cost):
        cost = Decimal(str(cost))
        self.nodes[m]['shareCost'] = cost

    def get_node_type(self, m):
        if 'type' in self.nodes[m]:
            return self.nodes[m]['type']
        else:
            return None

    def get_elements(self):
        return list(self.nodes(data=True)), list(self.edges(data=True))

    def get_node_prob(self, m):
        return Decimal(str(self.nodes[m]['prob']))

    def get_node_capacity(self, m, avoid_gpu=False):
        if avoid_gpu and SpecialNodeAttributes.GPU.value in self.nodes[m]:
            return 0
        return self.nodes[m]['shareCap']

    def get_nearest_node(self, node, edge_or_transport=False):
        min_distance = math.inf
        nearest_node = None
        for n in self.nodes:
            if n != node:
                if edge_or_transport:
                    if self.get_node_tier(n) > 2:
                        continue
                distance = self.get_distance(self.nodes, self.edges, node, n)
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = n
        return nearest_node

    def get_node_capacities_dict(self) -> {}:
        return {node: self.get_node_capacity(node) for node in self.nodes}

    def get_link_capacities_dict(self) -> {}:
        return {edge: self.get_link_capacity(*edge) for edge in self.edges}

    @lru_cache(maxsize=10)
    def get_tier_nodes(self, tier):
        return [node for node in self.nodes if self.nodes[node]['tier'] == tier]

    def get_node_tier_capacity(self, tier):
        capacity = 0
        for node in self.get_tier_nodes(tier):
            if self.nodes[node]['tier'] == tier:
                capacity += self.get_node_capacity(node)
        return capacity

    def get_link_tier_capacity(self, m_tier, n_tier):
        capacity = 0
        for m, n in self.edges:
            if self.get_node_tier(m) == m_tier and self.get_node_tier(n) == n_tier:
                capacity += self.get_link_capacity(m, n)
        return capacity if capacity > 0 else 1

    def get_node_successors(self, m):
        return [n for n in self.successors(m)]

    def add_biderectional_edge(self, m, n, link_capacity, link_cost):
        if m == n:
            raise Exception("Self loop is not allowed")
        link_capacity = Decimal(str(link_capacity))
        self.add_edge(m, n, shareCap=link_capacity, shareCost=link_cost)
        self.add_edge(n, m, shareCap=link_capacity, shareCost=link_cost)

    def scale_node_size(self, m, factor):
        self.nodes[m]['shareCap'] = self.nodes[m]['shareCap'] * factor

    def scale_link_size(self, m, n, factor):
        self.edges[m, n]['shareCap'] = self.edges[m, n]['shareCap'] * factor

    def subtract_link_capacity(self, m, n, share):
        share = Decimal(str(share))
        self.edges[(m, n)]['shareCap'] = round(self.edges[(m, n)]['shareCap'] - share, 8)
        assert self.edges[(m, n)]['shareCap'] >= 0

    def update_link_capacities(self, base_capacity):
        for m, n in self.edges:
            cap_factor = self.get_link_capacity(m, n) / config.base_link_capacity
            self.set_link_capacity((m, n), base_capacity * cap_factor)

    def subtract_node_capacity(self, m, share):
        share = Decimal(str(share))
        self.nodes[m]['shareCap'] = round(self.nodes[m]['shareCap'] - share, 8)
        assert self.nodes[m]['shareCap'] >= 0

    def set_zero_node_capacity(self, m):
        self.nodes[m]['shareCap'] = 0

    def node_connected_to_core_node(self, node):
        for n in self.successors(node):
            if self.get_node_tier(n) == config.num_of_tiers:
                return True
        return False

    def get_total_node_capacity(self, avoid_gpu=True):
        return sum(self.get_node_capacity(m, avoid_gpu) for m in self.nodes)

    def get_total_link_capacity(self, avoid_gpu=True):
        return sum(self.get_link_capacity((m, n), avoid_gpu=avoid_gpu) for m, n in self.edges)


    @lru_cache(maxsize=None)
    def get_edge_list(self):
        return list(self.edges)

    @lru_cache(maxsize=None)
    def get_node_list(self):
        return list(self.nodes)

    @lru_cache(maxsize=None)
    def get_incoming_edges_dict(self):
        physical_graph_nodes = self.get_node_list()
        physical_graph_edges = self.get_edge_list()
        return {n: [(m, n) for m in physical_graph_nodes if (m, n) in physical_graph_edges] for n in
                physical_graph_nodes}

    @lru_cache(maxsize=None)
    def get_outgoing_edges_dict(self):
        physical_graph_nodes = self.get_node_list()
        physical_graph_edges = self.get_edge_list()
        return {n: [(n, l) for l in physical_graph_nodes if (n, l) in physical_graph_edges] for n in
                physical_graph_nodes}

    def set_edge_to_core_assoc(self, edge: str, core: []):
        self.invalid_core_nodes[edge] = core

    def get_invalid_core_nodes(self, edge: str):
        return self.invalid_core_nodes[edge]

    def get_nodes(self):
        return list(self.nodes)

    def get_links(self):
        return list(self.edges)

    def get_num_of_links(self):
        if config.bidirectional_links:
            return int(len(self.get_links()) / 2)
        else:
            return len(self.get_links())

    @staticmethod
    def get_distance(nodes, edges, node1, node2):
        """Get distance based on latitude and longitude"""
        if (node1, node2) not in edges:
            return 1000
        if 'distance' in edges[(node1, node2)]:
            return edges[(node1, node2)]['distance']
        lat1 = nodes[node1]['Latitude']
        lon1 = nodes[node1]['Longitude']
        lat2 = nodes[node2]['Latitude']
        lon2 = nodes[node2]['Longitude']
        return EnhancedDiGraph.haversine(lat1, lon1, lat2, lon2)

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # radius of Earth in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def create_subgraph(G, elements):
        nodes_removed = [n for n in G.nodes if n not in elements]
        H = G.__class__()
        for el in elements:
            if isinstance(el, tuple):
                if el[0] in nodes_removed or el[1] in nodes_removed:
                    continue
                edge_attrs = G.get_edge_data(el[0], el[1])
                H.add_edge(el[0], el[1], **edge_attrs)
            else:
                node_attrs = G.nodes[el]
                H.add_node(el, **node_attrs)
        return H

    @lru_cache(maxsize=None)
    def get_num_of_nodes(self):
        return len(self.get_nodes())

    def get_node_input_link_capacity(self, m):
        return sum(self.get_link_capacity(n, m) for n in self.predecessors(m))

    def get_edge_node_cost(self):
        for node in self.nodes:
            if self.get_node_tier(node) == 1:
                return self.get_node_cost(node)

    def get_max_edge_cost(self):
        return max(nx.get_edge_attributes(self, 'shareCost').values())

    def get_num_of_tiers(self):
        return max(nx.get_node_attributes(self, 'tier').values())

    def get_egress_link_capacity(self, node, avoid_gpu=False):
        successors = self.successors(node)
        capacity = 0
        for n in successors:
            if avoid_gpu and SpecialNodeAttributes.GPU.value in self.nodes[n]:
                continue
            capacity += self.get_link_capacity(node, n)
        return capacity


