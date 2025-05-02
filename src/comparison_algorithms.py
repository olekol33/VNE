from collections import defaultdict
from methodtools import lru_cache
from decimal import Decimal
import time
from application import Application
from include.enums import GreedyInvalid
from fluid_model import FluidModel
from line_profiler_pycharm import profile
from dataclasses import dataclass
import networkx as nx
from file_handler import FileHandler
from online_handler import OnlineHandler, OnlineDataLogger
from include import config
from result_exporter import UserAllocationResultExporter, UserAllocationResultExporterParams, ResultExporter
from code_testing import timer


class EmbeddingAlgorithm:
    def __init__(self, exp_settings, model: FluidModel, requests: []):
        from result_exporter import AllocationStatCollectorOnline, AllocationStatCollector
        self.run_number = exp_settings.run_number
        self.exp_name = exp_settings.exp_name
        self.exp_settings = exp_settings
        self.branch_name = exp_settings.branch_name
        self.nrf = exp_settings.nrf
        self.erf = exp_settings.erf
        self.run_settings = exp_settings
        self.app_by_branches: {} = model.apps
        self.multiplier = model.multiplier
        self.requests = requests
        self.apps = model.apps
        self.alt_dist = None
        self.physical_graph = model.physical_graph.copy()
        self.static_physical_graph = model.physical_graph.copy()
        self.stats = AllocationStatCollector(
            self.physical_graph) if not config.online_mode else (
            AllocationStatCollectorOnline(self.physical_graph, exp_settings.test_duration))
        self.valid_nodes = list(self.physical_graph.nodes)

        FileHandler.create_exp_rundir(self.run_number, exp_settings.exp_name)
        self._initialize_metrics()

    def _embed_requests(self):
        raise NotImplementedError("Embedding algorithm not found")

    def _initialize_metrics(self):
        self.allocated_virtual_nodes = 0
        self.allocated_virtual_links = 0
        self.allocated_requests = 0
        self.allocated_demand = 0
        self.unallocated_demand = 0
        self.total_demand = 0
        self.cost = 0
        self.node_utilization = 0
        self.link_utilization = 0
        self.run_time = 0
        self.allocated_share = {}
        self.cost_from_user_location = {}
        self.node_tier_share_dict, self.link_tier_share_dict = ResultExporter.create_tier_dict()
        self._create_allocated_share_dict()

    def _create_allocated_share_dict(self):
        nodes = self.physical_graph.get_nodes()
        links = self.physical_graph.get_links()
        elements = nodes + links
        self.allocated_share = {i: 0 for i in elements}

    @staticmethod
    @profile
    def collect_path_dict_from_path_finder(path, allocation_graph, demand):
        path_allocated = {}
        for path_object in path:
            for element in path_object.elements:
                multiplier = allocation_graph.get_link_multiplier_for_aloc_element(element)
                share = demand * multiplier
                i, j, m, n = element
                share_to_allocate = path_allocated.get((m, n), 0) + share
                path_allocated[(m, n)] = share_to_allocate
        return path_allocated

    def deallocate_dict_path(self, path):
        for (m, n), share in path.items():
            share = -share
            if m == n:
                self.physical_graph.subtract_node_capacity(m, share)
            else:
                self.physical_graph.subtract_link_capacity(m, n, share)

    @profile
    def allocate_dict(self, path):
        for (m, n), share in path.items():
            if m == n:
                self.physical_graph.subtract_node_capacity(m, share)
            else:
                self.physical_graph.subtract_link_capacity(m, n, share)

    def _run_embedding(self):
        self._embed_requests()
        self.node_utilization = 0
        self.link_utilization = 0

    def _populate_node_cost_lists_and_app_sizes(self):
        start = time.time()
        self._create_node_cost_lists()
        self._get_application_sizes()
        end = time.time()
        self.run_time = end - start

    def _create_node_cost_lists(self):
        nodes = [node for node in self.physical_graph.nodes if
                 self.physical_graph.get_node_tier(node) == 1]
        for src in nodes:
            self.update_cost_list_for_single_node(src)

    def _get_application_sizes(self):
        raise NotImplementedError("Not implemented")

    @profile
    def update_cost_list_for_single_node(self, src):
        raise NotImplementedError("Not implemented")

    def _embed_nodes(self, path, user, fully_allocated):
        f1_list = []
        f2_list = []
        f3_list = []
        app_graph = self.apps[user.app_name].graph
        for ind, (virtual_node, physical_node) in enumerate(path):
            if type(virtual_node) is tuple or virtual_node == config.user_func:
                continue
            if config.log_alternatives:
                predecessor = next(iter(app_graph.predecessors(virtual_node)))
            else:
                predecessor = None
            f1_str, f2_str, f3_str = self._embed_element(user, virtual_node, virtual_node, physical_node, physical_node,
                                                         fully_allocated, predecessor=predecessor)
            if f3_str:
                f3_list.append(f3_str)
        return f1_list, f2_list, f3_list

    def _embed_links(self, path_links: [], user, virtual_graph, fully_allocated):
        if not config.online_mode and self._is_accelerator_in_app(virtual_graph):
            i, j = self._get_acc_link(virtual_graph)
        else:
            user_func_successors = list(virtual_graph.successors(config.user_func))
            i = config.user_func
            j = user_func_successors[0]
        f1 = []
        f2 = []
        for m, n in path_links:
            part_f1, part_f2, _ = self._embed_element(user, i, j, m, n, fully_allocated)
        return f1, f2

    def _embed_element(self, user, virtual_node, next_virtual_node, m, n, fully_allocated, predecessor=None):
        f1 = None
        f2 = None
        f3 = None
        if m == n:
            multiplier = self.multiplier.get_node_multiplier(user.app_name, virtual_node, m)
            shareCost = self.physical_graph.get_node_cost(m)
            share = user.demand * multiplier
            if fully_allocated:
                self.allocated_share[m] += share
                if config.log_alternatives and predecessor in self.apps[user.app_name].choice_nodes.keys():
                    branch_num = self.apps[user.app_name].choice_nodes[predecessor].index(virtual_node)
                    f3 = f"{user.app_name},{user.assoc_node},{m},{predecessor},{virtual_node},{branch_num},{user.demand}\n"
            self.allocated_virtual_nodes += share
            self.node_tier_share_dict = ResultExporter.update_node_tier_share_count(m, self.node_tier_share_dict, share,
                                                                                    self.physical_graph)
        else:
            multiplier = self.multiplier[(user.app_name, virtual_node, next_virtual_node, m, n)]
            shareCost = self.physical_graph.get_link_cost(m, n)
            share = user.demand * multiplier
            if fully_allocated:
                self.allocated_share[(m, n)] += share
            self.allocated_virtual_links += share
            self.link_tier_share_dict = ResultExporter.update_link_tier_share_count(m, n, self.link_tier_share_dict,
                                                                                    share, self.physical_graph)
        allocated_size = self._update_capacity(m, n, share)
        cost = shareCost * share
        if fully_allocated:
            self.cost += cost
            self.stats.add_cost_per_assoc_node(user.assoc_node, cost)
        return f1, f2, f3

    def _update_capacity(self, m, n, share, reduce_capacity=True) -> float:
        if m == n:
            if reduce_capacity:
                return self._subtract_node_capacity(m, share)
            else:
                return self._increase_node_capacity(m, share)
        else:
            if reduce_capacity:
                return self._subtract_link_capacity(m, n, share)
            else:
                return self._increase_link_capacity(m, n, share)

    def _increase_node_capacity(self, m, shar):
        raise NotImplementedError("Not implemented")

    def _subtract_node_capacity(self, m, share):
        node_cap = self.physical_graph.get_node_capacity(m)
        if node_cap == 0:
            return 0
        if share < node_cap:
            self.physical_graph.subtract_node_capacity(m, share)
            if self.physical_graph.get_node_capacity(m) <= config.allowed_error:
                self.valid_nodes.remove(m)
                self.physical_graph.set_zero_node_capacity(m)
            return round(share, 2)
        elif abs(share - node_cap) <= config.allowed_error:
            allocated_size = node_cap
            self.valid_nodes.remove(m)
            self.physical_graph.set_zero_node_capacity(m)
            return round(allocated_size, 2)
        else:
            raise Exception("Demand is greater than node capacity")

    def _subtract_link_capacity(self, m, n, share):
        raise NotImplementedError("Not implemented")

    def _increase_link_capacity(self, m, n, share):
        raise NotImplementedError("Not implemented")

    @profile
    def find_min_path(self, app, src, dst):
        try:
            shortest_path = nx.dijkstra_path(self.physical_graph, src, dst, weight='shareCost')
            if not self._is_allocation_feasible(app.graph, shortest_path):
                return None, None
            path_pairs = self._map_virtual_nodes_to_physical_nodes(app, shortest_path)
            edge_cost = self._get_app_links_allocation_cost(app.graph, path_pairs)
            node_cost = self._get_app_node_allocation_cost(app.app, path_pairs)
            total_cost = edge_cost + node_cost
            return path_pairs, total_cost
        except nx.NetworkXNoPath:
            return None, None

    @staticmethod
    def _get_app_links_allocation_cost(virtual_graph, path_pairs):
        raise NotImplementedError("Not implemented")

    def _get_app_node_allocation_cost(self, app_name, path_pairs):
        total_cost = 0
        for virtual_element, physical_node in path_pairs:
            if type(virtual_element) is tuple:
                continue
            node_cost = self.physical_graph.get_node_cost(physical_node)
            total_cost += node_cost * self.multiplier.get_node_multiplier(app_name, virtual_element, physical_node)
        return total_cost

    def _are_path_nodes_valid(self, path):
        """Checks both src and dst because greedy with acc allocated on both of them"""
        src = path[0]
        dst = path[-1]
        if src not in self.valid_nodes or dst not in self.valid_nodes:
            return False
        return True

    @staticmethod
    def get_path_links(path):
        """For each tuple in path create pairs of different physical nodes in successive tuples if they are different"""
        return [(path[i][1], path[i + 1][1]) for i in range(len(path) - 1) if path[i][1] != path[i + 1][1]]

    def _get_path_nodes(self, path):
        """Returns unique nodes in path. Physical nodes are at index 1 in each tuple pair"""
        return set(node[1] for node in path)

    def _get_available_share(self, user, path, app_graph) -> float:
        src = path[0][1]
        dst = path[-1][1]
        available_node_share = self._get_available_share_on_node(user, path)
        if src != dst:
            # available_link_share = self._get_link_capacity_to_destination(user, path, self.app_by_branches[user.app_name][user.branch_id].graph)
            available_link_share = self._get_link_capacity_to_destination(user, path, app_graph)
            return min(available_node_share, available_link_share)
        return available_node_share

    def _get_available_share_on_node(self, user, path):
        raise NotImplementedError("Not implemented")

    def _get_link_capacity_to_destination(self, user, path, virtual_graph):
        min_share = float('inf')
        if not config.online_mode and self._is_accelerator_in_app(virtual_graph):
            i, j = self._get_acc_link(virtual_graph)
        else:
            user_func_successors = list(virtual_graph.successors(config.user_func))
            i = config.user_func
            j = user_func_successors[0]
        path_links = self.get_path_links(path)
        for m, n in path_links:
            multiplier = self.multiplier[(user.app_name, i, j, m, n)]
            share = self.physical_graph.get_link_capacity(m, n) / multiplier
            min_share = min(min_share, share)
        return min_share

    def _get_acc_link(self, virtual_graph):
        raise NotImplementedError("Not implemented")

    def _is_accelerator_in_app(self, virtual_graph):
        return False

    def _is_allocation_feasible(self, graph, path):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def _map_virtual_nodes_to_physical_nodes(app, path):
        raise NotImplementedError("Not implemented")

    def _write_to_files(self, scenario_path, f3_list):
        if config.log_alternatives:
            with open(scenario_path / "branch_share.txt", "a") as f:
                f.write(f"app_name,user_node,dest_node,choice_node,branch_name,branch_num,share\n")
                f.writelines(f3_list)
                f.write(f"METADATA\n")
                f.write(f"Greedy,{len(self.requests)},{self.exp_settings.total_demand},{self.cost}\n")

    @profile
    def export_greedy_results(self):
        self.unallocated_demand = self.total_demand - self.allocated_demand
        params = UserAllocationResultExporterParams.take_input(self)
        UserAllocationResultExporter(params)
        self._export_share_utilization()

    def _export_share_utilization(self):
        import pandas as pd
        scenario_path = FileHandler.get_run_dir(self.run_number, self.exp_name) / 'share_utilization.csv'
        data_list = []
        for element, share in self.allocated_share.items():
            if type(element) is tuple:
                m, n = element
                tier_m, tier_n = self.static_physical_graph.get_node_tier(m), self.static_physical_graph.get_node_tier(
                    n)
                capacity = self.static_physical_graph.get_link_capacity(m, n)
            else:
                m, n = element, -1
                tier_m, tier_n = self.static_physical_graph.get_node_tier(m), -1
                capacity = self.static_physical_graph.get_node_capacity(m)
            utilization = round(share / capacity, 2)
            data_list.append([m, n, share, capacity, utilization, tier_m, tier_n])

        df = pd.DataFrame(data_list,
                          columns=['m', 'n', 'share', 'capacity', 'utilization', 'tier_m', 'tier_n'])
        df.to_csv(scenario_path, sep=',', encoding='utf-8', index=False)


class GreedyMinLoc(EmbeddingAlgorithm):
    @profile
    def __init__(self, exp_settings, model: FluidModel, requests: [], app_branches: {} = None):
        super().__init__(exp_settings, model, requests)
        print(f"Running greedy nrf {self.nrf}, erf {self.erf}, branch {self.branch_name}")
        self.application_size = {}
        self.pre_acc_application_size = {}
        self.post_acc_application_size = {}
        self.application_paths = {}
        self.removed_elements = []

        self.app_by_branches = {}
        if app_branches is None or len(app_branches) == 0:
            self._create_app_branches(model.apps, exp_settings.branch_id)
        else:
            self.app_by_branches = self._set_individual_alternatives_as_apps(app_branches)

        self._populate_node_cost_lists_and_app_sizes()
        self._run_embedding()
        self.export_greedy_results()

    def _set_individual_alternatives_as_apps(self, app_branches):
        self.alt_dist = {}
        nodes = list(self.physical_graph.nodes)
        self.branch_id_to_name = {}
        apps = {}

        for branch_id, app in app_branches.items():
            app_name = app.app
            app_dist = self.alt_dist.setdefault(app_name, {})
            app_dict = apps.setdefault(app_name, {})
            app_dict[branch_id] = app

            # branch_name = list(value.graph.successors(list(value.choice_nodes.keys())[0]))[0]
            branch_name = app.branch_name
            self.branch_id_to_name[branch_id] = branch_name

            for node in nodes:
                node_dict = app_dist.setdefault(node, {})
                node_dict[branch_name] = 0

        return apps

    def _create_app_branches(self, model_apps, branch_id):
        for app in model_apps.keys():
            app_branch = self.app_by_branches.get(app, {})
            app_branch[branch_id] = model_apps[app]
            self.app_by_branches[app] = app_branch

    def _get_application_sizes(self):
        m = list(self.physical_graph.nodes)[0]
        basic_demand_unit = 1
        for app_name in self.app_by_branches.keys():
            for app_branch, app in self.app_by_branches[app_name].items():
                total_pre_size = 0
                total_post_size = 0
                virtual_graph = app.graph
                pre_acc_nodes, post_acc_nodes = self._get_pre_post_acc_nodes(virtual_graph)
                for node in pre_acc_nodes:
                    total_pre_size += self.multiplier.get_node_multiplier(app_name, node, m) * basic_demand_unit
                for node in post_acc_nodes:
                    total_post_size += self.multiplier.get_node_multiplier(app_name, node, m) * basic_demand_unit
                self.pre_acc_application_size[AppIdentifier(app=app.app, branch_id=app_branch)] = total_pre_size
                self.post_acc_application_size[AppIdentifier(app=app.app, branch_id=app_branch)] = total_post_size

    @timer("_embed")
    @profile
    def _embed_requests(self):
        f1_list = []
        f2_list = []
        f3_list = []
        self.total_demand += sum(user.demand for user in self.requests)
        start = time.time()
        for request in self.requests:
            src = request.assoc_node
            app = request.app_name
            self.stats.increment_user_per_node(src)
            self.stats.add_total_demand(request.assoc_node, request.demand)
            min_cost_path, updated_user = self._get_valid_path(app, src, request)
            if min_cost_path:
                app_graph = self.app_by_branches[request.app_name][updated_user.branch_id].graph
                available_share = self._get_available_share(updated_user, min_cost_path, app_graph)
                f1_str, f2_str, f3_str = self._embed_and_update(min_cost_path, updated_user, available_share)
                f1_list.extend(f1_str)
                f2_list.extend(f2_str)
            f3_list.extend(f3_str)
        end = time.time()
        self.run_time += round(end - start, 2)
        scenario_path = FileHandler.get_run_dir(self.run_number, self.exp_name)
        self._write_to_files(scenario_path, f3_list)

    def _get_valid_path(self, app, src, user):
        if self._no_valid_paths_left(app, src, user.demand):
            return None, None
        min_cost_path, branch = self._get_min_cost_path(app, src)
        invalid_list = self._is_path_valid(min_cost_path)
        if len(invalid_list) > 0:
            self.update_cost_list_for_single_node(src)
            if self._no_valid_paths_left(app, src, user.demand):
                return None, None
            min_cost_path, branch = self._get_min_cost_path(app, src)
        new_user = user.update_branch_id(branch)
        return min_cost_path, new_user

    def update_cost_list_for_single_node(self, src):
        for app_name in self.app_by_branches.keys():
            for branch_id, app in self.app_by_branches[app_name].items():
                identifier = RequestIdentifier(app_name, src, branch_id)
                destination = []
                for dst in self.valid_nodes:
                    path, cost = self.find_min_path(app, src, dst)
                    if path is None:
                        continue
                    min_cost_path = MinCostPath(src, path, cost)
                    destination.append(min_cost_path)
                destination.sort(key=lambda x: x.cost)
                self.cost_from_user_location[identifier] = destination

    def _get_min_cost_path(self, app, src):
        branches = self.app_by_branches[app].keys()
        min_cost = float('inf')
        min_branch = None
        path = None
        for branch in branches:
            identifier = RequestIdentifier(app, src, branch)
            if not self.cost_from_user_location[identifier]:
                continue
            min_cost_path = self.cost_from_user_location[identifier][0]
            if min_cost_path.cost < min_cost:
                path = min_cost_path.path
                min_cost = min_cost_path.cost
                min_branch = branch
        return path, min_branch

    def _no_valid_paths_left(self, app, src, demand) -> bool:
        if src in self.removed_elements or not self._path_from_src_exists(app, src):
            self.unallocated_demand += demand
            return True
        return False

    def _path_from_src_exists(self, app, src):
        branches = self.app_by_branches[app].keys()
        for branch in branches:
            identifier = RequestIdentifier(app, src, branch)
            cost = self.cost_from_user_location.get(identifier)
            if cost:
                return True
        return False

    def _get_available_share_on_node(self, user, path):
        src = path[0][1]
        dst = path[-1][1]
        if config.online_mode:
            raise NotImplementedError("Recheck dynamic_physical_graph or physical_graph")
        dst_capacity = self.physical_graph.get_node_capacity(dst)
        dst_allocation_size = self.post_acc_application_size[AppIdentifier(app=user.app_name, branch_id=user.branch_id)]
        dst_share = dst_capacity / dst_allocation_size
        src_allocation_size = self.pre_acc_application_size[AppIdentifier(app=user.app_name, branch_id=user.branch_id)]
        if src_allocation_size > 0:
            if src == dst:
                allocation_size = src_allocation_size + dst_allocation_size
                return dst_capacity / allocation_size
            else:
                if config.online_mode:
                    raise NotImplementedError("Recheck dynamic_physical_graph or physical_graph")
                src_capacity = self.physical_graph.get_node_capacity(src)
                src_share = src_capacity / src_allocation_size
                return min(dst_share, src_share)
        return dst_share

    def _is_path_valid(self, path):
        if path is None:
            return [GreedyInvalid.SRC_INVALID]
        dst = path[-1][1]
        path_links = self.get_path_links(path)
        path_nodes = self._get_path_nodes(path)
        invalid_list = []

        is_nodes_valid = all(node not in self.removed_elements for node in path_nodes)
        is_dst_valid = dst in self.valid_nodes
        is_links_valid = all(link not in self.removed_elements for link in path_links)

        if not is_nodes_valid:
            invalid_list.append(GreedyInvalid.NODE_REMOVED)
        if not is_dst_valid:
            invalid_list.append(GreedyInvalid.DST_INVALID)
        if not is_links_valid:
            invalid_list.append(GreedyInvalid.LINK_REMOVED)

        return invalid_list

    def _no_paths_left(self):
        raise NotImplementedError("Embedding algorithm not found")

    @profile
    def _embed_and_update(self, path: [], request, available_share):
        allocated_request, fully_allocated = self._allocate_request(request, available_share)
        f1_list, f2_list, f3_list = self._embed_nodes(path, allocated_request, fully_allocated)
        path_links = self.get_path_links(path)
        if path_links:
            virtual_graph = self.app_by_branches[allocated_request.app_name][allocated_request.branch_id].graph
            f1_str, f2_str = self._embed_links(path_links, allocated_request, virtual_graph, fully_allocated)
            if f1_str and f2_str:
                f1_list.extend(f1_str)
                f2_list.extend(f2_str)
        if fully_allocated:
            # print(f'{request.id}: {path}')
            self._update_allocated_branches(allocated_request)
            self.stats.increment_allocated_requests_per_node(request.assoc_node)
            self.stats.increment_allocated_demand(request.assoc_node, request.demand)
            return f1_list, f2_list, f3_list
        else:
            # print(f'{path}')
            return [], [], []

    def _update_allocated_branches(self, user):
        if self.alt_dist is not None:
            branch_name = self.branch_id_to_name[user.branch_id]
            self.alt_dist[user.app_name][user.assoc_node][branch_name] += user.demand

    def _allocate_request(self, request, available_share):
        if available_share >= request.demand:
            self.allocated_demand += request.demand
            self.allocated_requests += 1
            return request, True
        else:
            return request.update_demand(available_share), False

    def _subtract_link_capacity(self, m, n, share):
        link_cap = self.physical_graph.get_link_capacity(m, n)
        if share < link_cap:
            self.physical_graph.subtract_link_capacity(m, n, share)
            if self.physical_graph.get_link_capacity(m, n) <= config.allowed_error:
                self.physical_graph.remove_edge(m, n)
                self.removed_elements.append((m, n))
            return round(share, 2)
        elif abs(share - link_cap) <= config.allowed_error:
            allocated_size = self.physical_graph.get_link_capacity(m, n)
            self.physical_graph.remove_edge(m, n)
            self.removed_elements.append((m, n))
            if self.physical_graph.degree(n) == 0:
                self.physical_graph.remove_node(n)
                self.removed_elements.append(n)
            if self.physical_graph.degree(m) == 0:
                self.physical_graph.remove_node(m)
                self.removed_elements.append(m)
                if m in self.valid_nodes:
                    self.valid_nodes.remove(m)
            return round(allocated_size, 2)
        else:
            raise Exception("Demand is greater than capacity")

    def _is_allocation_feasible(self, graph, path):
        if self._is_accelerator_in_app(graph) and not self._are_path_nodes_valid(path):
            return False
        return True

    def _is_path_with_acc_allocation_feasible(self, path):
        is_acc_path = False
        for f, node in path:
            if self._is_accelerator_node(f):
                is_acc_path = True
                break
        src = path[0][1]
        dst = path[-1][1]
        if is_acc_path and not self._are_path_nodes_valid([src, dst]):
            return False
        return True

    def _map_virtual_nodes_to_physical_nodes(self, app, path):
        # create tuples of (virtual_node, physical_node) based on physical nodes from path
        # if 'acc' in app.graph then place all pre_acc_nodes on src (path[0]) and all post_acc_nodes on dst (path[-1])
        # else place all virtual nodes on dst (path[-1])
        node_pairs = []
        pre_acc_nodes, post_acc_nodes = self._get_pre_post_acc_nodes(app.graph)
        if pre_acc_nodes:
            for node in pre_acc_nodes:
                node_pairs.append((node, path[0]))
            for i in range(1, len(path) - 1):
                node_pairs.append(((pre_acc_nodes[-1], post_acc_nodes[0]), path[i]))
        else:
            user_func = post_acc_nodes.pop(0)
            node_pairs.append((user_func, path[0]))
            for i in range(1, len(path) - 1):
                node_pairs.append(((post_acc_nodes[0], post_acc_nodes[1]), path[i]))
        for node in post_acc_nodes:
            node_pairs.append((node, path[-1]))
        return node_pairs

    def _get_pre_post_acc_nodes(self, virtual_graph):
        if not self._is_accelerator_in_app(virtual_graph):
            return [], list(virtual_graph.nodes)
        from path_finder import PathFinder
        pre_acc_nodes = []
        post_acc_nodes = []
        leafs = PathFinder.get_graph_leafs(virtual_graph)
        # if len(leaf) > 1:
        #     raise Exception("More than one leaf")
        for leaf in leafs:
            path = nx.shortest_path(virtual_graph, config.user_func, leaf)
            passed_acc = False
            for node in path:
                if not passed_acc:
                    pre_acc_nodes.append(node)
                else:
                    post_acc_nodes.append(node)
                if self._is_accelerator_node(node):
                    passed_acc = True
        # return list(set(pre_acc_nodes)), list(set(post_acc_nodes))
        return pre_acc_nodes, post_acc_nodes

    def _is_accelerator_in_app(self, virtual_graph):
        for node in virtual_graph.nodes:
            if self._is_accelerator_node(node):
                return True
        return False

    def _get_acc_link(self, virtual_graph):
        for i, j in virtual_graph.edges:
            if self._is_accelerator_node(i):
                return i, j

    def _is_accelerator_node(self, node):
        return 'acc' in node

    def _get_app_links_allocation_cost(self, virtual_graph, path_pairs):
        total_cost = 0
        if self._is_accelerator_in_app(virtual_graph):
            i, j = self._get_acc_link(virtual_graph)
        else:
            user_func_successors = list(virtual_graph.successors(config.user_func))
            i = config.user_func
            j = user_func_successors[0]
        for m, n in GreedyMinLoc.get_path_links(path_pairs):
            total_cost += self.physical_graph.get_link_cost(m, n) * self.multiplier[virtual_graph.name, i, j, m, n]
        return total_cost


class GreedyOnline(EmbeddingAlgorithm):
    @profile
    def __init__(self, exp_settings, model: FluidModel, requests, function_wise):
        from allocation_logger import OnlineAllocationLogger, ReqReleaseLogger
        super().__init__(exp_settings, model, requests)
        print(f"Running greedy nrf {self.nrf}, erf {self.erf}, branch {self.branch_name}")
        if config.gpu_nodes:
            raise NotImplementedError("ERROR: use GreedyLP for GPU")
        self.apps = Application.remove_fic_alternatives_from_dict(self.apps)
        self.alt_dist = None
        self.test_duration = exp_settings.test_duration
        self.demand_per_timeslot = OnlineHandler.get_total_demand_per_timeslot(self.requests, self.test_duration)
        self.application_size = {}
        self.pre_acc_application_size = {}
        self.post_acc_application_size = {}
        self.requests_costs = {}
        self.application_paths = {}
        self.valid_nodes = list(self.physical_graph.nodes)
        self.online_path_finder = OnlineGreedyPathFinder(self.apps, self.physical_graph, self.multiplier)
        self.scenario_path = FileHandler.get_run_dir(self.run_number, self.exp_name)
        self.invalid_links = []
        self._no_path_identifier_index = set()
        self.req_logger = ReqReleaseLogger(exp_settings)
        self.allocation_logger = OnlineAllocationLogger(exp_settings)
        self.arrivals, self.departures = OnlineHandler.get_request_arrivals_and_departures(requests, self.test_duration)
        self.online_logger_list = []
        self.request = None
        self.function_wise = function_wise
        self._populate_node_cost_lists_and_app_sizes()
        self._run_embedding()
        self.export_greedy_results(function_wise)

    @timer("_embed")
    @profile
    def _embed_requests(self):
        f3_list = []
        start = time.time()
        self.active_user_paths = {}
        for t in range(self.test_duration):
            self._handle_requests_per_t(t)
        end = time.time()
        self.run_time += round(end - start, 2)
        self._write_to_files(self.scenario_path, f3_list)
        self.allocation_logger.export(self.scenario_path)
        df = OnlineDataLogger.get_online_df(self.online_logger_list, self.run_settings, export=False)
        self.req_logger.post_process_cost(df)
        assert self.total_demand == self.allocated_demand + self.unallocated_demand

    @profile
    def _handle_requests_per_t(self, t):
        func_wise = 'FW' if self.function_wise else ''
        print(f"Greedy {func_wise}- Time {t}")
        self.online_logger = OnlineDataLogger(self.run_settings, self.multiplier)
        start_time = time.time()
        self._handle_departures(t)
        self._handle_arrivals(t)
        runtime = time.time() - start_time
        OnlineDataLogger.log_timeslot(t, self.arrivals, self.online_logger, self.online_logger_list,
                                      self.demand_per_timeslot, runtime)

    def _handle_departures(self, t):
        departures = self.departures[t]
        for request in departures:
            self._deallocate_request(t, request)
            self.stats.increment_departures(t, request.assoc_node)

    @profile
    def _handle_arrivals(self, t):
        arrivals = self.arrivals[t]
        rejected = []
        self._no_path_identifier_index = set()
        for ind, request in enumerate(arrivals):
            self.request = request
            self.online_logger.requests += 1
            self.online_logger.arrived_demand += request.demand
            assoc_node = request.assoc_node
            demand = request.demand
            self.total_demand += demand
            self.stats.arrival_logger(t, assoc_node, demand)
            min_cost_path, cost = self.online_path_finder.get_valid_path(request, self.physical_graph, run_settings=self.run_settings)
            if min_cost_path:
                self.requests_costs[request.id] = cost
                self.allocated_demand += demand
                self.allocated_requests += 1
                self._embed_and_update_online(t, min_cost_path, request)
            else:
                identifier = RequestIdentifier(app=request.app_name, src=request.assoc_node)
                self._no_path_identifier_index.add(identifier)
                rejected.append(request.id)
                self.unallocated_demand += demand
                self.req_logger.add_entry(t, request, False, cost)
                self.allocation_logger.add_event_entry(t, True, False, request.id,
                                                       request.app_name, request.assoc_node, demand, None)

    def _create_node_cost_lists(self):
        return

    @profile
    def export_greedy_results(self, function_wise):
        self.unallocated_demand = self.total_demand - self.allocated_demand
        params = UserAllocationResultExporterParams.take_input(self)
        UserAllocationResultExporter(params)

        file_path = self.scenario_path / "greedy_online_stats.csv"
        OnlineHandler.export_online_stats(self.stats, self.physical_graph.get_nodes(), file_path, 'Greedy')

    def _get_app_links_allocation_cost(self, virtual_graph, path_pairs):
        total_cost = 0
        user_func_successors = list(virtual_graph.successors(config.user_func))
        i = config.user_func
        j = user_func_successors[0]
        for m, n in self.get_path_links(path_pairs):
            total_cost += self.physical_graph.get_link_cost(m, n) * self.multiplier[virtual_graph.name, i, j, m, n]
        return total_cost

    def _get_application_sizes(self):
        m = list(self.physical_graph.nodes)[0]
        basic_demand_unit = 1
        for app_name, app in self.apps.items():
            total_size = 0
            virtual_graph = app.graph
            app_nodes = list(virtual_graph.nodes)
            for node in app_nodes:
                total_size += self.multiplier.get_node_multiplier(app_name, node, m) * basic_demand_unit
            self.application_size[AppIdentifier(app=app.app, branch_id=0)] = total_size

    def _is_allocation_feasible(self, graph, path):
        return EmbeddingAlgorithm._is_online_allocation_feasible(graph, path)

    @profile
    def _deallocate_request(self, t, request):
        if request.id not in self.active_user_paths.keys():
            self.allocation_logger.add_event_entry(request.end_time, False, False, request.id, request.app_name,
                                                   request.assoc_node, request.demand, None)
            return
        path = self.active_user_paths[request.id]
        self.deallocate_dict_path(path)
        cost = self.requests_costs[request.id]
        self.req_logger.add_entry(t, request, False, cost)
        self.allocation_logger.add_event_entry(request.end_time, False, True, request.id, request.app_name,
                                               request.assoc_node, request.demand, path)
        self.stats.increment_deallocated_requests(t, request.assoc_node)
        self.online_logger.update_deallocation(request.demand)
        del self.active_user_paths[request.id]
        del self.requests_costs[request.id]

    def _deallocate_node(self, path, user):
        virtual_node, m = path[-1]
        multiplier = self.multiplier.get_node_multiplier(user.app_name, virtual_node, m)
        share = user.demand * multiplier
        self._update_capacity(m, m, share, False)

    def _deallocate_links(self, path, user):
        path_links = self.get_path_links(path)
        i = config.user_func
        successors = list(self.apps[user.app_name].graph.successors(i))
        for j in successors:
            for link in path_links:
                m, n = link
                multiplier = self.multiplier[(user.app_name, i, j, m, n)]
                share = user.demand * multiplier
                self._update_capacity(m, n, share, False)

    @profile
    def _embed_and_update_online(self, t, path: {}, request):
        self.stats.allocated_logger(t, request.assoc_node, request.demand)
        self.active_user_paths[request.id] = path
        self.allocate_dict(path)
        self.allocation_logger.add_event_entry(t, True, True, request.id,
                                               request.app_name, request.assoc_node, request.demand, path)
        self.online_logger.update_allocation(request.demand)
        return

    @staticmethod
    @profile
    def add_element_to_path_dict(demand, multiplier, path_dict, m, n):
        path_dict[(m, n)] += demand * multiplier
        return path_dict


class OnlineGreedyPathFinder:
    def __init__(self, apps, original_graph, multipliers):
        self.original_graph = original_graph
        self.apps = apps
        self.multipliers = multipliers
        self.paths_from_src_node = {}
        self.nodes_by_cost = sorted(original_graph.nodes(), key=lambda dst: original_graph.nodes[dst]['shareCost'],
                                    reverse=False)

    @profile
    def get_valid_path(self, request, physical_graph, run_settings, alt_combinations=None):
        if config.gpu_nodes:
            return FullG.get_single_lp_path(physical_graph, self.apps, alt_combinations, request,
                                            self.multipliers, run_settings)
        else:
            return self.find_heuristic_min_path(self.apps[request.app_name], request, physical_graph)

    @staticmethod
    def adjust_path_cost_size(path, cost, demand):
        """Scale from default request size to specific request size"""
        new_path_values = {}
        factor = (Decimal(str(demand)) / Decimal(str(config.request_demand_size_mean)))
        for (m, n) in path.keys():
            new_path_values[(m, n)] = path[(m, n)] * factor
        return new_path_values, cost * factor

    @staticmethod
    def has_capacity_violation(path, physical_graph):
        for (m, n), share in path.items():
            if m == n:
                if physical_graph.get_node_capacity(m) < share:
                    return m
            else:
                if physical_graph.get_link_capacity(m, n) < share:
                    return (m, n)
        return False

    def find_heuristic_min_path_func_wise(self, request, physical_graph) -> ():
        path_allocated = defaultdict(Decimal)
        src = request.assoc_node
        app_name = request.app_name
        app = self.apps[app_name]
        cost = 0
        for i, j in app.get_app_path():
            path_dict, src = self.allocate_func(request, src, physical_graph, i, j, path_allocated)
            if path_dict is None:
                cost = self.multipliers.get_base_rejection_cost(app.get_app_size()) * request.demand
                return None, cost
            for (m, n), share in path_dict.items():
                path_allocated[(m, n)] += share
                cost += self.multipliers.get_base_cost(request.app_name, i, j, m, n) * request.demand
        return path_allocated, cost

    @profile
    def allocate_func(self, request, src, physical_graph, i, j, planned_allocation):
        demand = request.demand
        path_dict = {}
        for dst in self.nodes_by_cost:
            node_share = self.multipliers.get_node_multiplier(request.app_name, j, dst) * demand
            if not self.is_node_capacity_constraint_satisfied(node_share, physical_graph, dst, planned_allocation):
                continue
            try:
                path = self._get_shortest_distance_path(src, dst)
                if path is None:
                    continue
            except nx.NetworkXNoPath:
                continue
            shares = [demand * self.multipliers[(request.app_name, i, j, path[ind][0], path[ind][1])] for ind in
                      range(len(path))]
            link_shares = [(path[ind], shares[ind]) for ind in range(len(path))]
            if not self.is_link_capacity_constraint_satisfied(physical_graph, link_shares, planned_allocation):
                continue
            for link, share in link_shares:
                path_dict[link] = share
            path_dict[dst, dst] = node_share

            return path_dict, dst

        return None, None

    @staticmethod
    def is_node_capacity_constraint_satisfied(share, graph, dst, planned_allocation):
        planned_share = 0
        if (dst, dst) in planned_allocation:
            planned_share = planned_allocation[dst, dst]
        return graph.get_node_capacity(dst) >= share + planned_share

    @staticmethod
    def is_link_capacity_constraint_satisfied(graph, link_share, planned_allocation):
        for link, share in link_share:
            planned_share = 0
            if (link[0], link[1]) in planned_allocation:
                planned_share = planned_allocation[link[0], link[1]]
            if graph.get_link_capacity(link[0], link[1]) < share + planned_share:
                return False
        return True

    @lru_cache(maxsize=None)
    def _get_shortest_distance_path(self, src, dst):
        try:
            path = nx.dijkstra_path(self.original_graph, src, dst, weight='distance')
            edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        except nx.NetworkXNoPath:
            return None
        return edges_in_path

    @profile
    def find_heuristic_min_path(self, app, request, graph, valid_nodes=None, invalid_links=None) -> ():
        from application import ApplicationGraphParser
        src = request.assoc_node
        demand = request.demand
        app_node_sizes, first_link_size = ApplicationGraphParser.get_node_sizes_and_first_link(app.get_graph())
        if valid_nodes is None:
            valid_nodes = [node for node in graph.nodes if graph.get_node_capacity(node) >= demand * app_node_sizes]
        if invalid_links is None:
            invalid_links = [link for link in graph.edges if
                             graph.get_link_capacity(link[0], link[1]) < demand * first_link_size]
        invalid_links = tuple(invalid_links)
        min_cost = float('inf')
        min_cost_path = None
        for dst in valid_nodes:
            try:
                shortest_path = self.shortest_path_on_subgraph(invalid_links, src, dst)
            except nx.NetworkXNoPath:
                continue
            if shortest_path is None:
                continue
            physical_path = OnlineGreedyPathFinder._map_virtual_nodes_to_physical_nodes(app, shortest_path)
            path_dict = self._collect_allocation_dict(physical_path, request.app_name, demand, app.graph)
            total_cost = self.get_allocation_greedy_cost(self.original_graph, path_dict)
            if total_cost < min_cost:
                min_cost = total_cost
                min_cost_path = path_dict
        if min_cost_path is None:
            cost = self.multipliers.get_base_rejection_cost(app.get_app_size()) * request.demand
        else:
            cost = min_cost
        return min_cost_path, cost

    @staticmethod
    @profile
    def get_allocation_greedy_cost(physical_graph, path):
        total_cost = 0
        for (m, n), share in path.items():
            if m == n:
                total_cost += physical_graph.get_node_cost(m) * share
            else:
                total_cost += physical_graph.get_link_cost(m, n) * share
        return total_cost

    @staticmethod
    def _map_virtual_nodes_to_physical_nodes(app, path):
        # create tuples of (virtual_node, physical_node) based on physical nodes from path
        # place all virtual nodes on dst (path[-1])
        node_pairs = []
        app_nodes = list(app.graph.nodes)
        user_func = config.user_func
        node_pairs.append((user_func, path[0]))
        user_func_successors = list(app.graph.successors(user_func))
        for successor in user_func_successors:
            for i in range(1, len(path) - 1):
                node_pairs.append(((user_func, successor), path[i]))
        for node in app_nodes:
            if node == config.user_func:
                continue
            node_pairs.append((node, path[-1]))
        return node_pairs

    @profile
    def _collect_allocation_dict(self, path, app_name, demand, app_graph):
        path_allocated = defaultdict(Decimal)
        path_links = EmbeddingAlgorithm.get_path_links(path)
        user_func_successors = list(app_graph.successors(config.user_func))
        i = config.user_func
        j = user_func_successors[0]
        for link in path_links:
            m, n = link
            multiplier = self.multipliers[(app_name, i, j, m, n)]
            path_allocated = GreedyOnline.add_element_to_path_dict(demand, multiplier, path_allocated, m , n)
        for element in path:
            if isinstance(element[0], str) and element[0] != config.user_func:
                j, n = element
                m = n
                multiplier = self.multipliers.get_node_multiplier(app_name, j, n)
                path_allocated = GreedyOnline.add_element_to_path_dict(demand, multiplier, path_allocated, m , n)
        return path_allocated

    @lru_cache(maxsize=50000)
    @profile
    def shortest_path_on_subgraph(self, invalid_links, src, dst):
        res_graph = self.gen_valid_subgraph(invalid_links)
        return nx.dijkstra_path(res_graph, src, dst, weight='shareCost')

    @lru_cache(maxsize=50000)
    def gen_valid_subgraph(self, invalid_links):
        res_graph = self.original_graph.copy()
        res_graph.remove_edges_from(invalid_links)
        return res_graph


class FullG(EmbeddingAlgorithm):
    @profile
    def __init__(self, exp_settings, model: FluidModel, requests, alternatives):
        from result_exporter import AllocationStatCollectorOnline
        from allocation_logger import ReqReleaseLogger
        self.run_settings = exp_settings
        self.nrf = exp_settings.nrf
        self.erf = exp_settings.erf
        self.test_duration = exp_settings.test_duration
        self.demand_per_timeslot = OnlineHandler.get_total_demand_per_timeslot(requests, self.test_duration)
        self.model = None
        self.requests = requests
        self.total_demand = 0
        self.allocated_demand = 0
        self.unallocated_demand = 0
        self.unallocated_requests = 0
        self.allocated_requests = 0
        self.run_time = 0
        self.online_logger_list = []
        self.cost = 0
        self.requests_costs = {}
        self.run_number = exp_settings.run_number
        self.exp_name = exp_settings.exp_name
        self.apps = OnlineHandler.remove_fic_alternatives_from_apps_dict(model.apps)
        self.req_logger = ReqReleaseLogger(exp_settings)
        self.dirpath = FileHandler.get_run_dir(self.run_number, self.exp_name)
        self.cost_from_user_location = {}
        self.valid_nodes = list(model.physical_graph.nodes)
        self.physical_graph = model.physical_graph.copy()
        self.multiplier = model.multiplier
        self.demand_mult = 100
        self.alternatives = alternatives.copy()
        for app in self.alternatives.values():
            for k in list(app.keys()):
                if 'fic' in k:
                    del app[k]
        self.arrivals, self.departures = OnlineHandler.get_request_arrivals_and_departures(self.requests, self.test_duration)
        self.stats = AllocationStatCollectorOnline(self.physical_graph, self.test_duration)
        self.active_user_paths = {}
        self.run()

    @profile
    def run(self):
        from fluid_model import SingleLPModel
        start_time = time.time()
        test_duration = self.run_settings.test_duration
        self.active_user_paths = {}
        for t in range(test_duration):
            print(f"FullG - Time {t}")
            self.online_logger = OnlineDataLogger(self.run_settings, self.multiplier)
            start_time = time.time()
            arrivals = self.arrivals[t]
            departures = self.departures[t]
            for request in departures:
                self.process_departure(t, request)
            for request in arrivals:
                self.online_logger.requests += 1
                self.online_logger.arrived_demand += request.demand
                self.total_demand += request.demand
                fm = SingleLPModel(self.physical_graph, [request], self.apps, self.alternatives,
                                 self.run_settings, multiplier=self.multiplier, dirpath=self.dirpath)
                self.process_allocation(fm, request)
            runtime = time.time() - start_time
            OnlineDataLogger.log_timeslot(t, self.arrivals, self.online_logger, self.online_logger_list,
                                          self.demand_per_timeslot, runtime)
        self.run_time = time.time() - start_time
        self.export()

    def process_allocation(self, fm, request):
        if fm is not None and len(fm.non_zero_vars) > 0:
            allocation_graph = list(fm.allocation_graphs.values())[0]
            cost = self.allocate(request, allocation_graph)
            if cost is not None:
                self.requests_costs[request.id] = cost
                self.cost += cost
                self.allocated_requests += 1
                self.online_logger.update_allocation(request.demand)
                return
        cost = self._get_base_rejection_cost_wrap(request)
        self.req_logger.add_entry(request.start_time, request, False, cost)

    def _get_base_rejection_cost_wrap(self, request):
        return self.multiplier.get_base_rejection_cost(self.apps[request.app_name].get_app_size()) * request.demand

    def _get_dealloc_cost(self, t, request):
        if t != request.end_time:
            return self._get_base_rejection_cost_wrap(request)
        else:
            return self.requests_costs[request.id]

    def process_departure(self, t, request):
        if request.id not in self.active_user_paths:
            return
        cost = self._get_dealloc_cost(t, request)
        self.deallocate_dict_path(self.active_user_paths[request.id])
        self.online_logger.update_deallocation(request.demand)
        self.req_logger.add_entry(t, request, False, cost)
        del self.active_user_paths[request.id]
        del self.requests_costs[request.id]

    @profile
    def allocate(self, request, allocation_graph):
        path, _ = allocation_graph.run_path_finder()
        if path is not None:
            path_allocated = EmbeddingAlgorithm.collect_path_dict_from_path_finder(path, allocation_graph, request.demand)
            cost = OnlineGreedyPathFinder.get_allocation_greedy_cost(self.physical_graph, path_allocated)
            self.allocate_dict(path_allocated)
            self.active_user_paths[request.id] = path_allocated
            return cost
        return None

    def export(self):
        self.node_utilization = 0
        self.link_utilization = 0
        params = UserAllocationResultExporterParams.take_input(self)
        UserAllocationResultExporter(params)
        df = OnlineDataLogger.get_online_df(self.online_logger_list, self.run_settings, export=False)
        self.req_logger.post_process_cost(df)

    @staticmethod
    def get_single_lp_path(physical_graph, apps, alternatives, request, multiplier, run_settings):
        from fluid_model import SingleLPModel
        fm = SingleLPModel(physical_graph, [request], apps, alternatives=alternatives, exp_settings=run_settings, multiplier=multiplier)
        if fm is not None and len(fm.non_zero_vars) > 0:
            allocation_graph = list(fm.allocation_graphs.values())[0]
            path, _ = allocation_graph.run_path_finder()
            path_allocated = EmbeddingAlgorithm.collect_path_dict_from_path_finder(path, allocation_graph,
                                                                                   request.demand)
            cost = OnlineGreedyPathFinder.get_allocation_greedy_cost(physical_graph, path_allocated)
            return path_allocated, cost
        else:
            return None, None


@dataclass
class MinCostPath:
    source: str
    path: []
    cost: float


@dataclass(frozen=True)
class RequestIdentifier:
    app: str
    src: str
    branch_id: int = 0


@dataclass(frozen=True)
class AppIdentifier:
    app: str
    branch_id: int
