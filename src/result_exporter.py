from element_translator import ElementTranslator
from decimal import Decimal
import copy
import pandas as pd
from include import config
from file_handler import FileHandler
from physical_graph import GraphSizeScaler
from enhanced_graph import EnhancedDiGraph
from fluid_model import FluidModel
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass


class AllocationStatCollector:
    def __init__(self, physical_graph: EnhancedDiGraph):
        nodes = physical_graph.get_nodes()
        self.allocated_requests_per_node = {i: 0 for i in nodes}
        self.users_per_node = {i: 0 for i in nodes}
        self.cost_per_assoc_node = {i: 0 for i in nodes}
        self.cost_per_allocated_user = {i: 0 for i in nodes}
        self.rejection_rate = {i: 0 for i in nodes}
        self.rejected_demand_rate = {i: 0 for i in nodes}
        self.allocated_demand = {i: 0 for i in nodes}
        self.total_demand = {i: 0 for i in nodes}

    def increment_user_per_node(self, node):
        self.users_per_node[node] += 1

    def increment_allocated_demand(self, node, demand):
        self.allocated_demand[node] += demand

    def increment_allocated_requests_per_node(self, node):
        self.allocated_requests_per_node[node] += 1

    def add_allocated_requests_per_node(self, node, requests):
        self.allocated_requests_per_node[node] += requests

    def add_cost_per_assoc_node(self, node, cost):
        self.cost_per_assoc_node[node] += cost

    def add_total_demand(self, node, demand):
        self.total_demand[node] += demand


class AllocationStatCollectorOnline(AllocationStatCollector):
    def __init__(self, physical_graph: EnhancedDiGraph, test_duration: int):
        super().__init__(physical_graph)
        # test_duration = int(round(config.sim_time * (1 - config.train_set_ratio), 0))
        nodes = physical_graph.get_nodes()
        self.arrivals = [{i: 0 for i in nodes} for _ in range(test_duration)]
        self.arrivals_demand = [{i: 0 for i in nodes} for _ in range(test_duration)]
        self.departures = [{i: 0 for i in nodes} for _ in range(test_duration)]
        self.departures_demand = [{i: 0 for i in nodes} for _ in range(test_duration)]
        self.allocated_users_per_t = [{i: 0 for i in nodes} for _ in range(test_duration)]
        self.allocated_demand_per_t = [{i: 0 for i in nodes} for _ in range(test_duration)]
        self.deallocated_users = [{i: 0 for i in nodes} for _ in range(test_duration)]
        self.rejected_user_rate_per_t = [0] * test_duration
        self.rejected_demand_rate_per_t = [0] * test_duration

    def increment_arrivals_per_t(self, t, node):
        self.arrivals[t][node] += 1

    def increment_arrivals_demand_per_t(self, t, node, demand):
        self.arrivals_demand[t][node] += demand

    def increment_departures(self, t, node):
        self.departures[t][node] += 1

    def add_departures_demand(self, t, node, demand):
        self.departures_demand[t][node] += demand

    def increment_allocated_users_per_t(self, t, node):
        self.allocated_users_per_t[t][node] += 1

    def increment_allocated_demand_per_t(self, t, node, demand):
        self.allocated_demand_per_t[t][node] += demand

    def increment_deallocated_requests(self, t, node):
        self.deallocated_users[t][node] += 1

    def arrival_logger(self, t, node, demand):
        self.increment_arrivals_per_t(t, node)
        self.increment_arrivals_demand_per_t(t, node, demand)
        self.increment_user_per_node(node)

    def allocated_logger(self, t, node, demand):
        self.increment_allocated_users_per_t(t, node)
        self.increment_allocated_demand_per_t(t, node, demand)
        self.increment_allocated_requests_per_node(node)
        self.increment_allocated_demand(node, demand)


class AllocationStatExporter:
    def __init__(self, stats_sets, num_users: int):
        self.stats_sets = stats_sets
        self.num_users = num_users
        self._set_nrf_erf()
        self.post_process_stats()

    def _set_nrf_erf(self):
        run_settings = list(self.stats_sets[0].keys())[0]
        self.nrf = run_settings.nrf
        self.erf = run_settings.erf

    def post_process_stats(self, ):
        for set_stats in self.stats_sets.values():
            for exp, stats in set_stats.items():
                mean_allocation_cost_per_node = {node: c / u if u > 0 else 0 for node, c, u in
                                                 zip(stats.users_per_node.keys(), stats.cost_per_assoc_node.values(),
                                                     stats.users_per_node.values())}
                set_stats[exp].cost_per_allocated_user = copy.deepcopy(mean_allocation_cost_per_node)
                set_stats[exp].rejected_user_rate = {node: 1 - (a / u) if u > 0 else 0 for node, a, u in
                                                     zip(stats.users_per_node.keys(),
                                                         stats.allocated_requests_per_node.values(),
                                                         stats.users_per_node.values())}
                set_stats[exp].rejected_demand_rate = {node: 1 - (a / u) if u > 0 else 0 for node, a, u in
                                                       zip(stats.users_per_node.keys(),
                                                           stats.allocated_demand.values(),
                                                           stats.total_demand.values())}


    def _get_sorted_nodes_by_users(self):
        """Returns a list of nodes sorted by number of users"""
        nodes = list(self.stats_sets[0].values())[0].users_per_node.keys()
        users_per_node = {node: 0 for node in nodes}
        for stats in self.stats_sets[0].values():
            for node, users in stats.users_per_node.items():
                users_per_node[node] += users
        return sorted(users_per_node, key=users_per_node.get, reverse=True)


    @staticmethod
    def export_to_file(df, filename):
        """Export the df to a csv file. Create file if empty, append to file if not"""
        file_path = FileHandler.get_run_dir('summary') / filename
        if not file_path.exists():
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False, mode='a', header=False)

    @staticmethod
    def get_branch_name(branch_name) -> str:
        if 'OPT' in branch_name:
            branch_nodes = branch_name.split('OPT-')[1]
            runtype = 'OPT: '
            branch_name = runtype + branch_nodes
        elif 'Greedy' in branch_name:
            branch_nodes = branch_name.split('Greedy-')[1]
            runtype = 'Greedy: '
            branch_name = runtype + branch_nodes
        elif 'TANTO' in branch_name:
            branch_nodes = 'TANTO'
        elif 'Oracle-LP' in branch_name:
            branch_nodes = 'Oracle-LP'
        elif 'Oracle-Heuristic' in branch_name:
            branch_nodes = 'Oracle'
        else:
            raise ValueError(f"Unexpected branch name {branch_name}")
        return branch_name

    @staticmethod
    def _get_m_n_dimensions_for_fig(elements):
        """Calculate the m,n dimensions for the figure, so that it will be as square as possible"""
        import math
        n = len(elements)
        m = math.ceil(n / math.sqrt(n))
        return m, m

    def _get_stats_list(self):
        # ignore = ['cost_per_assoc_node', 'allocated_users_per_node', 'allocated_demand', 'total_demand']
        ignore = ['cost_per_assoc_node', 'allocated_users_per_node']
        """Returns a list of all dicts in AllocationStatCollector"""
        attr_list = list(vars(list(self.stats_sets[0].values())[0]).keys())
        for i in ignore:
            attr_list.remove(i)
        return attr_list

    def _get_num_of_nodes(self, stat):
        attribute = getattr(list(self.stats_sets[0].values())[0], stat)
        return len(attribute)

    def _remove_redundant_exp(self):
        """Iterate over self.stats and remove experiments in which branch is in remove_branches"""
        remove_branches = ['Greedy-acc_2']
        to_remove = []
        for set_stats in self.stats_sets.values():
            for exp_settings in set_stats:
                if exp_settings.branch_name in remove_branches:
                    to_remove.append(exp_settings)
        # separate loop to remove items
        for key in to_remove:
            for set_stats in self.stats_sets.values():
                if key in set_stats:
                    del set_stats[key]


class ResultExporter:
    def __init__(self, run_number: int, number_of_requests: int, fm: FluidModel, physical_graph: EnhancedDiGraph,
                 apps: {}, branch_name=None, export_to_file=True, oracle=False):
        print(f"Exporting results for run {run_number}")
        self.run_number = run_number
        self.number_of_requests = number_of_requests
        self.allocation = fm.allocation_graphs
        self.multiplier = fm.multiplier
        self.node_utilization = fm.node_utilization
        self.link_utilization = fm.link_utilization
        self.physical_graph = physical_graph
        self.allocated_share_by_assoc_node = {}
        self.apps = apps
        self.fm = fm
        self.oracle = oracle
        self.exp_name = fm.exp_name
        self.model = fm.model
        self.requests = fm.requests
        self.non_zero_vars = fm.non_zero_vars
        self.request_demand = fm.request_demand
        self.physical_graph = fm.physical_graph
        self.stats = AllocationStatCollector(fm.physical_graph)
        self.branch_name = branch_name
        self.export_to_file = export_to_file

        self.export_results()

    def export_results(self):
        # self._export_alternative_dist()
        self.export_share_utilization_model()
        self.export_allocation_summary()
        self.export_model_results()

    def export_model_results(self):
        total_demand = sum(self.stats.total_demand.values())
        allocated_demand = sum(self.stats.allocated_demand.values())
        unallocated_demand = total_demand - allocated_demand
        allocated_requests = sum(self.stats.allocated_requests_per_node.values())
        app = list(self.apps.values())[0]
        app_node_size = GraphSizeScaler.get_app_sizes_nodes_links(app)[0]
        run_time = self.model.get_solve_details().time
        non_alloc_cost = unallocated_demand * self.physical_graph.get_edge_node_cost() * app_node_size
        params = UserAllocationResultExporterParams(
            run_number=self.run_number, branch_name=self.branch_name, num_of_requests=self.number_of_requests,
            topology=config.topology_name, allocated_requests=allocated_requests, allocated_demand=allocated_demand,
            unallocated_demand=unallocated_demand,
            cost=self.fm.cost, non_alloc_cost=non_alloc_cost, total_demand=total_demand, nrf=self.fm.nrf,
            erf=self.fm.erf, node_utilization=self.node_utilization, link_utilization=self.link_utilization,
            run_time=run_time)
        if not self.export_to_file:
            return
        UserAllocationResultExporter(params)

    @staticmethod
    def create_dist_dict(apps, nodes):
        choice_nodes = {}
        alt_dist = {}
        for app in apps.values():
            app_name = app.app
            choice_node = list(app.choice_nodes.keys())[0]
            choice_nodes[app_name] = choice_node
            alt_dist[app_name] = {}
            for node in nodes:
                app_dist = {}
                for successor in app.graph.successors(choice_node):
                    app_dist[successor] = 0
                alt_dist[app_name][node] = app_dist
        return alt_dist, choice_nodes

    @staticmethod
    def custom_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d} ({p:.2f}%)'.format(p=pct, v=val)

        return my_autopct

    def export_allocation_summary(self):
        if not self.export_to_file:
            return
        total_original_demand = 0
        cost_on_links = 0
        cost_on_nodes = 0
        tiers_share = self._create_list_by_tiers()
        file_path = self.get_file_path('summary')
        try:
            df = pd.read_csv(file_path, index_col=0)
        except FileNotFoundError:
            df = pd.DataFrame()
        for r in self.allocation.keys():
            func_alloc = set()
            allocation_req = self.allocation[r]
            duration = allocation_req.req.end_time - allocation_req.req.start_time if config.online_mode else 1
            demand = allocation_req.req.demand * duration * allocation_req.weight
            total_original_demand += demand
            for link in allocation_req.allocation_graph.edges:
                for element in ElementTranslator.alloc_graph_to_physical_graph_elements(allocation_req, link):
                    i, j, m, n = element
                    multiplier = self.multiplier[allocation_req.app.get_app_name(), i, j, m, n]
                    share = allocation_req.get_link_capacity(link) / multiplier
                    if m == n:
                        node = link[1][1]
                        shareCost = self.physical_graph.get_node_cost(node)
                        cost = multiplier * share * shareCost
                        tiers_share = self._count_share_to_tier(tiers_share, node, share)
                        func_alloc.add(node)
                        cost_on_nodes += cost
                    else:
                        link0_type = allocation_req.get_element_type(link[0])
                        if allocation_req.element_is_agg_node(link0_type):
                            continue
                        shareCost = self.physical_graph.get_link_cost(m, n)
                        cost = multiplier * share * shareCost
                        cost_on_links += cost
            self.allocated_share_by_assoc_node[allocation_req.req.assoc_node] = allocation_req.req.demand
        total_allocated_cost = cost_on_nodes + cost_on_links
        df.loc['Nodes', str(self.number_of_requests)] = self.physical_graph.get_num_of_nodes()
        df.loc['Links', str(self.number_of_requests)] = self.physical_graph.get_num_of_links()
        df.loc['Node Utilization', str(self.number_of_requests)] = float(self.node_utilization)
        df.loc['Link Utilization', str(self.number_of_requests)] = float(self.link_utilization)
        df.loc['Total Cost', str(self.number_of_requests)] = float(total_allocated_cost)
        df.loc['Link Cost', str(self.number_of_requests)] = float(cost_on_links)
        df.loc['Node Cost', str(self.number_of_requests)] = float(cost_on_nodes)
        df.loc['Allocated Demand', str(self.number_of_requests)] = float(total_original_demand)
        df.loc['Time', str(self.number_of_requests)] = self.model.get_solve_details().time

        df.to_csv(file_path, sep=',', encoding='utf-8')

    def export_share_utilization_model(self):
        file_path = self.get_file_path('share_utilization')
        node_tier_share_dict, link_tier_share_dict = ResultExporter.create_tier_dict()
        node_capacities = self.physical_graph.get_node_capacities_dict()
        link_capacities = self.physical_graph.get_link_capacities_dict()
        node_shares = defaultdict(int)
        orig_node_shares = defaultdict(int)
        orig_link_shares = defaultdict(int)
        link_shares = defaultdict(int)
        data_list = []
        self._get_total_demand_into_stats()
        allocated_demand = {k: 0 for k in self.allocation.keys()}
        for var, value in self.non_zero_vars.items():
            app_name, alt_name, assoc_node, i, j, m, n = var
            key = (app_name, alt_name, assoc_node)
            if m == n:
                share = value * self.multiplier.get_node_multiplier(app_name, j, m)
                node_shares[m] += share
                if i != config.user_func:
                    orig_node_shares[m] += value
                else:
                    if i == j:
                        if allocated_demand[key] > 0:
                            raise ValueError(f"allocated_demand for {key} already exists")
                        allocated_demand[key] = value
                node_tier_share_dict = ResultExporter.update_node_tier_share_count(m, node_tier_share_dict, share,
                                                                                   self.physical_graph)
            else:
                share = value * self.multiplier[app_name, i, j, m, n]
                link_shares[m, n] += share
                orig_link_shares[m, n] += value
                link_tier_share_dict = ResultExporter.update_link_tier_share_count(m, n, link_tier_share_dict,
                                                                                   share, self.physical_graph)

        for k in self.allocation.keys():
            app_name, alt_name, assoc_node = k
            demand = allocated_demand[k] * (self.allocation[k].req.end_time - self.allocation[k].req.start_time)
            self.stats.increment_allocated_demand(assoc_node, demand)  # assuming single successor to 'U'
            self.stats.add_allocated_requests_per_node(assoc_node,
                                                       self._estimate_num_requests_from_requests_list(assoc_node,
                                                                                                      demand))

        if not self.export_to_file:
            return

        for m in node_capacities.keys():
            share = round(node_shares.get(m, 0), 2)
            capacity = round(node_capacities[m], 2)
            tier = self.physical_graph.get_node_tier(m)
            if capacity == 0:
                utilization = 0
            else:
                utilization = round(share / capacity, 2)

            data_list.append([m, -1, share, capacity, utilization, tier, '-1'])
        for m, n in link_capacities.keys():
            share = round(link_shares.get((m, n), 0), 2)
            capacity = round(link_capacities[m, n], 2)
            utilization = round(share / capacity, 2)
            tier_m = self.physical_graph.get_node_tier(m)
            tier_n = self.physical_graph.get_node_tier(n)
            data_list.append([m, n, share, capacity, utilization, tier_m, tier_n])
        df = pd.DataFrame(data_list, columns=['m', 'n', 'share', 'capacity', 'utilization', 'tier_m', 'tier_n'])
        df.to_csv(file_path, sep=',', encoding='utf-8', index=False)

    def _estimate_num_requests_from_requests_list(self, m, share):
        for grouped_request in self.requests:
            if m == grouped_request.assoc_node:
                demand_share = share / grouped_request.demand
                return round(grouped_request.grouped_users * demand_share)

    def _get_total_demand_into_stats(self):
        for request in self.requests:
            total_demand = request.demand * (request.end_time - request.start_time)
            self.stats.add_total_demand(request.assoc_node, total_demand)

    @staticmethod
    def update_node_tier_share_count(m, tier_share_dict, share, physical_graph):
        tier = physical_graph.get_node_tier(m)
        tier_share_dict[tier] = tier_share_dict[tier] + share
        return tier_share_dict

    @staticmethod
    def update_link_tier_share_count(m, n, tier_share_dict, share, physical_graph):
        m_tier = physical_graph.get_node_tier(m)
        n_tier = physical_graph.get_node_tier(n)
        tier_share_dict[(m_tier, n_tier)] = tier_share_dict[(m_tier, n_tier)] + share
        return tier_share_dict

    @staticmethod
    def create_tier_dict():
        import itertools
        node_tier_share_dict = {i: 0 for i in range(1, config.num_of_tiers + 1)}
        link_tier_share_dict = {perm: 0 for perm in itertools.product(range(1, config.num_of_tiers + 1), repeat=2)}

        return node_tier_share_dict, link_tier_share_dict

    def _create_list_by_tiers(self) -> []:
        tiers_allocation = [0] * self.physical_graph.get_num_of_tiers()
        return tiers_allocation

    def _count_share_to_tier(self, tiers: {}, node: str, share: float) -> []:
        tier = self.physical_graph.nodes[node]['tier']
        tiers[tier - 1] += share
        return tiers

    def get_file_path(self, file_name) -> Path:
        """
        Returning the exact location to store the performance metrics
        """
        scenario_path = FileHandler.get_run_dir(self.run_number, self.exp_name)
        filepath = scenario_path / str(file_name + '.csv')
        return filepath


@dataclass
class UserAllocationResultExporterParams:
    run_number: int = -1
    branch_name: str = ''
    num_of_requests: int = -1
    allocated_requests: int = -1
    allocated_demand: float = -1
    unallocated_demand: float = -1
    cost: float = -1
    non_alloc_cost: float = -1
    total_demand: float = -1
    nrf: float = -1
    erf: float = -1
    node_utilization: float = -1
    link_utilization: float = -1
    run_time: float = -1
    topology: str = ''
    target_utilization: float = -1
    hike: float = -1
    special_input: float = -1

    @staticmethod
    def take_input(params):
        try:
            app = list(params.apps.values())[0]
            app_node_size = GraphSizeScaler.get_app_sizes_nodes_links(app)[0]
            non_alloc_cost = params.unallocated_demand * params.physical_graph.get_edge_node_cost() * app_node_size
            num_of_requests = len(params.requests)
            params = UserAllocationResultExporterParams(
                run_number=params.run_settings.run_number, branch_name=params.run_settings.branch_name,
                num_of_requests=num_of_requests,
                topology=config.topology_name, allocated_requests=params.allocated_requests,
                allocated_demand=params.allocated_demand, unallocated_demand=params.unallocated_demand,
                cost=params.cost, non_alloc_cost=non_alloc_cost, total_demand=params.total_demand,
                nrf=params.run_settings.nrf,
                erf=params.run_settings.erf, node_utilization=params.node_utilization,
                link_utilization=params.link_utilization,
                run_time=params.run_time, target_utilization=params.run_settings.target_utilization,
                hike=params.run_settings.hike,
                special_input=params.run_settings.special_input)
        except Exception as e:
            raise ValueError(f"Error in UserAllocationResultExporterParams: {e}")
        return params

    def __post_init__(self):
        self.allocated_requests = self.allocated_requests
        self.allocated_demand = round(self.allocated_demand, 2)
        self.cost = round(self.cost, 2)
        self.penalized_cost = round(self.cost + self.non_alloc_cost)
        self.total_demand = round(self.total_demand, 2)
        self.rejected_demand_rate = round(1 - self.allocated_demand / self.total_demand, 2)
        self.run_time = round(self.run_time, 2)
        self.rejected_user_rate = round((self.num_of_requests - self.allocated_requests) / self.num_of_requests, 2) if (
                self.num_of_requests > 0) else 0


class UserAllocationResultExporter:

    def __init__(self, exporter_params: UserAllocationResultExporterParams = None):
        self.special_in_str = None
        self._set_header()
        self.export_allocation_results(exporter_params)

    def get_header(self):
        return self.header

    def _set_header(self):
        from online_handler import OnlineDataLogger
        self.special_in_str = OnlineDataLogger.get_special_exp_string()
        if config.online_mode:
            if self.special_in_str == 'None':
                self.header = ['Run Number', 'Target Utilization', 'Topology', 'Requests', 'Allocated Requests',
                               'Total Demand', 'Allocated Demand',
                               'Cost', 'Penalized Cost', 'Rejected Demand Rate', 'Time']
            else:
                self.header = ['Run Number', 'Target Utilization', f'{self.special_in_str}', 'Topology', 'Requests',
                               'Allocated Requests',
                               'Total Demand', 'Allocated Demand', 'Cost', 'Penalized Cost', 'Rejected Demand Rate',
                               'Time']

        else:
            self.header = ['Run Number', 'NRF', 'ERF', 'Topology', 'Node Utilization', 'Link Utilization',
                           'Requests', 'Allocated Requests', 'Total Demand', 'Allocated Demand',
                           'Rejected Demand Rate', 'Cost', 'Penalized Cost', 'Rejected User Rate',
                           'Time']

    def export_allocation_results(self, params):
        if params is None:
            return
        file_path = UserAllocationResultExporter._get_file_path(params.branch_name)
        data = {
            'Run Number': params.run_number,
            'Target Utilization': params.target_utilization,
            'NRF': params.nrf,
            'ERF': params.erf,
            'Topology': params.topology,
            'Node Utilization': params.node_utilization,
            'Link Utilization': params.link_utilization,
            'Requests': params.num_of_requests,
            'Allocated Requests': params.allocated_requests,
            'Total Demand': params.total_demand,
            'Allocated Demand': params.allocated_demand,
            'Rejected Demand Rate': params.rejected_demand_rate,
            'Cost': params.cost,
            'Penalized Cost': params.penalized_cost,
            'Rejected User Rate': params.rejected_user_rate,
            'Time': params.run_time,
            'Hike': params.hike,
            f'{self.special_in_str}': params.special_input
        }
        data = self._dict_values_decimal_to_float(data)

        data = {k: v for k, v in data.items() if k in self.header}

        df = pd.DataFrame([data], columns=self.header)

        if file_path.exists():
            existing_df = pd.read_csv(file_path)
            df = pd.concat([df, existing_df], ignore_index=True)

        df.to_csv(file_path, sep=',', encoding='utf-8', index=False)

    @staticmethod
    def _dict_values_decimal_to_float(data: {}):
        for k, v in data.items():
            if isinstance(v, Decimal):
                data[k] = float(v)
        return data

    @staticmethod
    def _get_file_path(exp_name) -> Path:
        scenario_path = FileHandler.get_run_dir('summary')
        if not scenario_path.exists():
            os.makedirs(scenario_path)
        if exp_name is None:
            file_name = 'allocation.csv'
        else:
            file_name = exp_name + '.csv'
        filepath = scenario_path / file_name
        return Path(filepath)


class RunTimer:
    def __init__(self, run_number: int, number_of_users: int, lp_time_sec: float, total_time: float, heur_time: float):
        self.filepath = FileHandler.get_run_dir(run_number) / "runtime.csv"
        self.number_of_users = number_of_users
        self.lp_time = lp_time_sec
        self.total_time = total_time
        self.heuristic_time = heur_time
        if not self._is_file_exists():
            self._create_file()
        self._write_to_file()

    def _is_file_exists(self):
        if os.path.exists(self.filepath):
            return True
        else:
            return False

    def _create_file(self):
        with open(self.filepath, 'w') as file:
            file.write("Users,Total,Solver,Heuristic\n")

    def _write_to_file(self):
        with open(self.filepath, 'a') as file:
            file.write(f'{self.number_of_users},{self.total_time},{self.lp_time},{self.heuristic_time}\n')
