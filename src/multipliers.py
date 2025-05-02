from decimal import Decimal
from include import config
from enhanced_graph import EnhancedDiGraph
from include.enums import SpecialNodeAttributes
from file_handler import FileHandler
from functools import lru_cache

AGG_MULTIPLIER = 1


class DefaultMultiplierDict(dict):
    def __init__(self, apps: {}, requests: [], physical_graph: EnhancedDiGraph, nrf, erf, run_number, exp_name,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apps = apps
        self.requests = requests
        self.physical_graph = physical_graph
        self.gpu_nodes = self._get_gpu_nodes()
        self.nrf = nrf
        self.erf = erf
        self.run_number = run_number
        self.exp_name = exp_name

    def __missing__(self, key):
        self[key] = self._get_default_multiplier(key)
        return self[key]

    def _get_gpu_nodes(self):
        if not config.gpu_nodes:
            return []
        else:
            return [node for node in self.physical_graph.nodes if SpecialNodeAttributes.GPU.value
                    in self.physical_graph.nodes[node]]

    def get_base_cost(self, app_name, i, j, m, n):
        element = app_name, i, j, m, n
        multiplier = self._get_default_multiplier(element)
        return multiplier * self.physical_graph.get_element_cost(m, n)

    def get_base_rejection_cost(self, app_size):
        node_size, link_size = app_size
        node_size = Decimal(str(node_size))
        link_size = Decimal(str(link_size))
        return (node_size * self.physical_graph.get_max_element_cost(is_node=True) +
                link_size * self.physical_graph.get_max_element_cost(is_node=False))

    def _get_default_multiplier(self, element) -> float:
        app_name, i, j, m, n = element
        if m == n:
            element = (app_name, j, j, n, n)
            return self._get_default_node_multiplier(element)
        else:
            return self._get_default_link_multiplier(element)

    def _get_default_node_multiplier(self, element) -> float:
        app_name, i, j, m, n = element
        app = self.apps[app_name]
        if i == config.agg_node or j == config.agg_node:
            return AGG_MULTIPLIER
        elif i != j:
            raise ValueError("unexpected node indices")
        special_multiplier = self._get_special_node_multiplier(element)
        node_size = Decimal(str(app.graph.nodes[j].get('size', 0)))
        return node_size * special_multiplier

    def _get_default_link_multiplier(self, element) -> float:
        app_name, i, j, m, n = element
        app = self.apps[app_name]
        special_multiplier = self._get_special_link_multiplier(element)
        if i == config.agg_node:
            return special_multiplier
        link_size = app.graph.edges.get((i, j), {}).get('size', 1)
        return link_size if link_size == 1 else link_size * special_multiplier

    def _get_special_node_multiplier(self, element: ()) -> float:
        app_name, i, j, m, n = element
        if m != n:
            raise ValueError("unexpected node indices")
        return self.get_app_node_attribute(self.apps[app_name], j, n)

    def _get_special_link_multiplier(self, element: ()) -> float:
        app_name, i, j, m, n = element
        if m == n:
            raise ValueError("unexpected link indices")
        elif i == config.agg_node:
            return 1
        elif (m, n) not in self.physical_graph.edges:
            raise ValueError(f"({m}, {n}) not in physical graph")
        return DefaultMultiplierDict.get_app_link_attribute(self.apps[app_name], (i, j))

    def get_node_multiplier(self, app_name, virtual_node, m):
        return self[(app_name, virtual_node, virtual_node, m, m)]

    def get_link_multiplier(self, app_name, i, j, m, n):
        return self[(app_name, i, j, m, n)]

    # @lru_cache(maxsize=None)
    def get_app_node_attribute(self, app, func, node):
        if (SpecialNodeAttributes.GPU.value in app.graph.nodes[func] and node not in self.gpu_nodes):
            val = Decimal(str(config.gpu_multiplier))
        elif (SpecialNodeAttributes.GPU.value not in app.graph.nodes[func] and node in self.gpu_nodes):
            val = Decimal(str(config.gpu_multiplier))
        else:
            val = 1
        return val

    @staticmethod
    @lru_cache(maxsize=None)
    def get_app_link_attribute(app, link):
        zip_multiplier = 0.1  #compress/decompress mode
        if SpecialNodeAttributes.ZIP.value in app.graph.edges[link]:
            val = zip_multiplier
        elif SpecialNodeAttributes.ACCELERATOR.value in app.graph.edges[link]:
            val = config.acc_scaling_factor
        else:
            val = 1
        return Decimal(str(val))
