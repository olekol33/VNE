import random
import numpy as np
from functools import lru_cache
import copy
import networkx as nx
from include import config
from include.enums import SpecialNodeAttributes, ZipNodeTypes, AppDist, OnlineExpType
from pathlib import Path
import os
from file_handler import FileHandler


class ApplicationSet:
    def __init__(self, fic_layers=config.fic_layers, num_apps=None, length_apps=None, set_id=0):
        self.apps = {}
        if config.use_app_generator and config.online_mode:
            if num_apps is None and config.online_exp_type == OnlineExpType.NUMBER_OF_APPS:
                raise ValueError("Number of apps must be specified when using app generator")
            if length_apps is None and config.online_exp_type == OnlineExpType.LENGTH_OF_APPS:
                raise ValueError("Length of apps must be specified when using app generator")
            self.apps = ApplicationGenerator(num_apps=num_apps, length_apps=length_apps, fic_layers=fic_layers,
                                             set_id=set_id).get_apps()
        else:
            if len(config.applications_to_use) == 1:
                fic_layers = 0
            for app_name in config.applications_to_use:
                graph = ApplicationGraphParser(app_name, fic_layers).get_graph()
                self.apps[app_name] = Application(app_name, graph)
                self._set_app_share(app_name)

    def get_apps_dict(self):
        return self.apps

    def _set_app_share(self, app_name):
        if config.app_popularity == AppDist.UNIFORM:
            self.apps[app_name].set_app_share(1 / len(config.applications_to_use))
        else:
            raise NotImplementedError(f"App distribution {config.app_popularity} is not implemented")

    def get_app_names(self):
        return list(self.apps.keys())

    def get_apps(self):
        apps = {}
        for app in self.apps.values():
            apps[app.get_app_name()] = app
        return apps


class ApplicationGenerator:
    GPU = 'gpu'
    TREE = 'tree'
    ACC = 'acc'

    def __init__(self, num_apps=None, length_apps=None, fic_layers=config.fic_layers, set_id=0):
        np.random.seed(config.seed + set_id)
        self.apps = {}
        self.app_length = length_apps
        self.chain_apps = config.agen_chain_apps if num_apps is None else num_apps
        self.agen_acc_apps = config.agen_acc_apps
        self.agen_tree_apps = config.agen_tree_apps
        self.agen_gpu_apps = config.agen_gpu_apps
        total_num_apps = self.chain_apps + self.agen_acc_apps + self.agen_tree_apps + self.agen_gpu_apps
        self.fic_layers = 0 if total_num_apps == 1 else fic_layers
        self.generate_apps()
        self.export_apps(set_id)

    def export_apps(self, set_id):
        dir = FileHandler.get_run_dir(run=set_id)
        data = []
        for app_name, app in self.apps.items():
            num_nodes = len(app.get_main_alternative().nodes) - 1
            data.append(f'{app_name}:\n'
                        f'nodes({num_nodes}): {list(app.get_main_alternative().nodes(data=True))}\n'
                        f'links:{list(app.get_main_alternative().edges(data=True))}\n')
        filename = 'apps.txt'
        if config.online_exp_type == OnlineExpType.LENGTH_OF_APPS:
            filename = f'apps_{self.app_length}.txt'
        with open(dir / filename, 'w') as f:
            f.writelines(data)

    def get_apps(self):
        return self.apps

    def generate_apps(self):
        for i in range(self.chain_apps):
            app = self.generate_app(i)
            self.apps[app.get_app_name()] = app
        for i in range(self.agen_acc_apps):
            app = self.generate_acc_app(i)
            self.apps[app.get_app_name()] = app
        for i in range(self.agen_tree_apps):
            app = self.generate_tree_app(i)
            self.apps[app.get_app_name()] = app
        for i in range(self.agen_gpu_apps):
            app = self.generate_gpu_app(i)
            self.apps[app.get_app_name()] = app

    def generate_app(self, i):
        app_name = f'app_{i}'
        graph = self.generate_graph(app_name)
        return Application(app_name, graph)

    def generate_acc_app(self, i):
        app_name = f'app_acc_{i}'
        graph = self.generate_graph(app_name, self.ACC)
        return Application(app_name, graph)

    def generate_tree_app(self, i):
        app_name = f'app_tree_{i}'
        graph = self.generate_graph(app_name, self.TREE)
        return Application(app_name, graph)

    def generate_gpu_app(self, i):
        app_name = f'app_gpu_{i}'
        graph = self.generate_graph(app_name, self.GPU)
        return Application(app_name, graph)

    def generate_graph(self, app_name, app_type=None):
        is_tree = True if app_type == self.TREE else False
        is_acc = True if app_type == self.ACC else False
        is_gpu = True if app_type == self.GPU else False

        app_length = self.app_length
        if app_length is None:
            if config.agen_app_max_length > config.agen_app_min_length:
                app_length = np.random.randint(config.agen_app_min_length, config.agen_app_max_length + 1)
            elif config.agen_app_max_length == config.agen_app_min_length:
                app_length = config.agen_app_max_length
            else:
                raise ValueError("Max app length is smaller than min app length")
        if is_tree:
            app_length = 3 if app_length < 3 else app_length
        size_factor = (4 / app_length) if config.online_exp_type == OnlineExpType.LENGTH_OF_APPS else 1
        func_mean = int(config.agen_func_mean * size_factor)
        func_std = (config.agen_func_std * size_factor)
        func_sizes = [config.agen_func_mean] * app_length if config.agen_func_std == 0 else (
            self.get_trunc_normal(func_mean, func_std, app_length))
        link_sizes = [config.agen_link_mean] * app_length if is_acc or config.agen_link_std == 0 \
            else self.get_trunc_normal(config.agen_link_mean, config.agen_link_std, app_length)
        graph = nx.DiGraph()
        prev_func = config.user_func
        graph.add_node(config.user_func, size=0)
        prev_func0, prev_func1 = None, None
        for i in range(app_length):
            if is_acc and i == 0:
                func = f'acc_0'
                graph.add_node(func, size=func_sizes[i], acc=True)
            else:
                func = f'f{i}'
                graph.add_node(func, size=func_sizes[i])
            if is_tree and i > 0:
                branch = i % 2
                if branch == 0:
                    graph.add_edge(prev_func0, func, size=link_sizes[i - 1])
                    prev_func0 = func
                elif branch == 1:
                    graph.add_edge(prev_func1, func, size=link_sizes[i - 1])
                    prev_func1 = func
                else:
                    raise ValueError(f"Branch {branch} is not implemented")
            else:
                graph.add_edge(prev_func, func, size=link_sizes[i - 1])
                if i == 0 and is_tree:
                    prev_func0 = func
                    prev_func1 = func
                prev_func = func
        if is_gpu:
            nodes = list(graph.nodes)
            nodes.remove(config.user_func)
            random.shuffle(nodes)
            random_function = np.random.choice(nodes)
            graph.nodes[random_function]['GPU'] = True
        graph.name = app_name
        return ApplicationGraphParser(fic_layers=self.fic_layers, graph=graph).get_graph()

    def get_trunc_normal(self, mean, std, size):
        from scipy.stats import truncnorm
        a = (1 - mean) / std
        X = truncnorm(a=a, b=np.inf, loc=mean, scale=std).rvs(size=size)
        return X.round().astype(int)


class Application:
    def __init__(self, app: str, graph: nx.DiGraph):
        self.app = app
        self.graph = graph
        self.zipped_elements = []
        self.optimizer_elements = []
        self.alternatives = {}
        self.choice_nodes = []
        self.share = 0
        self.main_alternative_head = None
        self.tree_splitters = []
        self.branch_name = config.full_app
        self.max_link_size_element = self._get_max_link_size_elements()
        self._apply_special_element_attributes()
        self._update_splitters()
        self._update_alternative_list()
        self.gpu_functions = self._get_gpu_functions()
        self.path = []

    def get_path(self):
        if config.skip_greedy:
            return
        graph = self.get_main_alternative()
        app_leaf = [n for n in self.graph.nodes if graph.out_degree(n) == 0]
        app_leaf = app_leaf[0]
        self.path = list(nx.all_simple_edge_paths(self.graph, source=config.user_func, target=app_leaf))[0]

    def _get_gpu_functions(self):
        if not config.gpu_nodes:
            return
        gpu_functions = []
        for node in self.graph.nodes:
            if 'GPU' in self.graph.nodes[node]:
                gpu_functions.append(node)
        return gpu_functions

    def get_app_path(self):
        return self.path

    def _get_max_link_size_elements(self):
        max_link_size = 0
        max_link_size_element = None
        for edge in self.graph.edges:
            if 'size' in self.graph.edges[edge] and 'fic_factor' not in self.graph.edges[edge]:
                size = self.graph.edges[edge]['size']
                if size > max_link_size:
                    max_link_size = size
                    max_link_size_element = edge
        return max_link_size_element

    def remove_fic_alternatives(self):
        fic_alt_names = []
        for alt_name in self.alternatives.keys():
            if 'fic_factor' in self.graph.nodes[alt_name]:
                fic_alt_names.append(alt_name)
        for alt_name in fic_alt_names:
            del self.alternatives[alt_name]
        self.graph = self.alternatives[self.main_alternative_head]
        return self

    def _update_splitters(self):
        self.choice_nodes = list(nx.get_node_attributes(self.graph, "choice"))
        if len(self.choice_nodes) == 0:
            self.choice_nodes.append(config.user_func)
        self._update_tree_splitters()

    def get_main_alternative(self):
        return self.alternatives[self.main_alternative_head]

    def _update_alternative_list(self) -> {}:
        alternative_heads = self.get_all_choice_node_successors(self.graph)
        for head in alternative_heads:
            alternative_graph = self.get_alternative_graph(head)
            if 'main' in self.graph.nodes[head] or len(alternative_heads) == 1:
                self.main_alternative_head = head
            self.alternatives[head] = alternative_graph

    def get_alternative_graph(self, alt_head):
        """Keeps the branch from the relevant head and removes the rest"""
        graph = self.graph.copy()
        alternative_heads = self.get_all_choice_node_successors(graph)
        for head in alternative_heads:
            if head != alt_head:
                graph = self.remove_all_node_successors(graph, head)
        return graph

    def remove_all_node_successors(self, graph, node):
        subgraph = nx.bfs_tree(graph, node)
        for n in subgraph.nodes():
            graph.remove_node(n)
        return graph

    def get_all_choice_node_successors(self, graph):
        if len(self.choice_nodes) > 1:
            raise ValueError(f"Application {graph.name} has more than one choice node")
        choice_node = self.choice_nodes[0]
        return list(graph.successors(choice_node))

    def get_alternatives(self):
        return self.alternatives

    @lru_cache(maxsize=50)
    def get_alternative_edge_list(self, head):
        return list(self.alternatives[head].edges)

    @lru_cache(maxsize=50)
    def get_alternative_node_list(self, head):
        return list(self.alternatives[head].nodes)

    def _remove_successors(self, node):
        if self.graph.out_degree(node) == 0:
            self.graph.remove_node(node)
        else:
            successors = list(self.graph.successors(node))
            for successor in successors:
                self._remove_successors(successor)
            self.graph.remove_node(node)

    def set_app_share(self, share):
        self.share = share

    def get_graph(self):
        return self.graph

    def _add_branch_number_attribute(self):
        branch_id = 1
        for choice_node in self.choice_nodes:
            for node in self.graph.successors(choice_node):
                branch_elements = self._get_all_nodes_successors(node)
                branch_elements.add((choice_node, node))
                branch_elements.add(node)
                for element in branch_elements:
                    if self._is_node(element):
                        self.graph.nodes[element]['branch_id'] = branch_id
                    else:
                        self.graph.edges[element]['branch_id'] = branch_id
                branch_id += 1

    def _get_all_nodes_successors(self, node):
        successors = set()
        for successor in nx.neighbors(self.graph, node):
            successors.add(successor)
            successors.add((node, successor))
            successors.update(self._get_all_nodes_successors(successor))
        return successors

    def _update_tree_splitters(self):
        for node in self.graph.nodes:
            if self.graph.out_degree(node) > 1:
                if node not in self.choice_nodes and node != config.user_func:
                    self.tree_splitters.append(node)

    def _apply_special_element_attributes(self) -> None:
        """"Sets zip and opt attributes based on z/uz/opt nodes"""
        self._apply_acc()

    def get_app_name(self):
        return self.app

    @lru_cache()
    def get_app_size(self):
        main_alternative = self.get_main_alternative()
        node_size = sum([size for size in nx.get_node_attributes(main_alternative, 'size').values()])
        link_size = sum([size for size in nx.get_edge_attributes(main_alternative, 'size').values()])
        return node_size, link_size

    def _apply_acc(self):
        acc_nodes = list(nx.get_node_attributes(self.graph, SpecialNodeAttributes.ACCELERATOR.value).keys())
        if len(acc_nodes) == 0:
            return
        acc_elements = {}
        for node in acc_nodes:
            node_type, id = self._get_node_type_and_id(node)
            acc_elements[id] = []
            for edge in self.graph.out_edges(node):
                acc_elements[id].append(edge)
        self._update_element_attributes(acc_elements, SpecialNodeAttributes.ACCELERATOR)
        for node in acc_nodes:
            self._remove_special_node_attributes(node, SpecialNodeAttributes.ACCELERATOR)

    def _remove_special_node_attributes(self, node, attribute) -> None:
        self.graph.nodes[node].pop(attribute.value, None)

    def _update_element_attributes(self, element_groups: [], attribute: str):
        for _, elements in element_groups.items():
            for element in elements:
                if self._is_node(element):
                    self.graph.nodes[element][attribute.value] = True
                else:
                    self.graph.edges[element][attribute.value] = True

    def _is_node(self, element):
        """Element is node if it's a string. Otherwise, it should be a tuple"""
        return isinstance(element, str)

    def _scale_edge_size(self, factor):
        for edge in self.graph.edges:
            size = self.graph.edges[edge]['size']
            self.graph.edges[edge]['size'] = round(size * factor, 2)

    def _get_node_type_and_id(self, node: str):
        type_exists = False
        if '_' not in node:
            raise ValueError(f'Node {node} has no id')
        node_type, id = node.split('_')
        if node_type in [type_.value for type_ in SpecialNodeAttributes]:
            type_exists = True
        elif node_type in [type_.value for type_ in ZipNodeTypes]:
            type_exists = True
        if type_exists:
            return node_type, id
        else:
            raise ValueError(f'Node {node} has illegal type {node_type}')

    @staticmethod
    def _get_all_lists_combinations(_list):
        return [(a, b) for idx, a in enumerate(_list) for b in _list[idx + 1:]]

    def is_func_tree_split(self, func: str) -> bool:
        if func in self.tree_splitters:
            return True
        else:
            return False

    @staticmethod
    def remove_fic_alternatives_from_dict(apps):
        apps = copy.deepcopy(apps)
        for app_name, app in apps.items():
            apps[app_name] = app.remove_fic_alternatives()
        return apps


class ApplicationGraphParser:
    def __init__(self, app_name=None, fic_layers=config.fic_layers, graph=None):
        self.fic_layers = fic_layers

        if graph is None:
            filepath = Path(os.path.dirname(__file__)) / 'include' / 'applications' / f'{app_name}.edgelist'
            self.graph = nx.read_edgelist(filepath, create_using=nx.DiGraph)
            self.graph.name = app_name
            self._add_node_attributes_to_graph(filepath)
        else:
            self.graph = graph
        self.add_fic_alternatives()

    def get_graph(self):
        return self.graph

    def add_fic_alternatives(self):
        if not config.online_mode or self.fic_layers == 0:
            return
        self.graph.nodes[config.user_func]['choice'] = True
        node_sizes, first_link = self.get_node_sizes_and_first_link(self.graph)
        self.set_main_alternative()
        for l in range(self.fic_layers):
            node = f'fic_{l}'
            node_size = node_sizes * 10
            link_size = first_link * 10
            fic_factor = l + 2
            self.graph.add_node(node, fic_factor=fic_factor, size=node_size)
            self.graph.add_edge(config.user_func, node, fic_factor=1, size=link_size)

    def set_main_alternative(self):
        user_successors = list(self.graph.successors(config.user_func))
        if len(user_successors) == 1:
            self.graph.nodes[user_successors[0]]['main'] = True

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_node_sizes_and_first_link(app):
        if config.online_mode:
            graph = app
        else:
            graph = app.get_main_alternative()
        node_sizes = sum([size for node, size in nx.get_node_attributes(graph, 'size').items() if 'fic' not in node])
        first_link = (config.user_func, list(graph.successors(config.user_func))[0])
        first_link_size = graph[first_link[0]][first_link[1]]['size']
        return node_sizes, first_link_size

    def _add_node_attributes_to_graph(self, filepath):
        """Node attributes in App.edgelist are marked as '#<node> {<attribute>: <attribute value>}"""
        func_attributes = self._get_node_attributes(filepath)
        for func in func_attributes.keys():
            for attribute in func_attributes[func]:
                value = func_attributes[func][attribute]
                self.graph.nodes[func][attribute] = value

    def _get_node_attributes(self, filepath) -> {}:
        node_attributes = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    line = line.lstrip('#').replace(" ", "").rstrip()
                    index = line.find('{')
                    if index == -1:
                        raise ValueError(f"Unexpected line format for node attribute - {line}")
                    func = line[0:index]
                    if func not in self.graph.nodes:
                        raise ValueError(f"Node {func} does not exist in graph")
                    attributes = line[index:]
                    attributes_entry = eval(attributes)
                    if func in node_attributes.keys():
                        node_attributes[func].update(attributes_entry)
                    else:
                        node_attributes[func] = attributes_entry
        return node_attributes
