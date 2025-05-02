from online_handler import OnlineHandler
import os
from decimal import Decimal
import polars as pl
from multiprocessing import cpu_count
from code_testing import timer
from include import config
import numpy as np
from physical_graph import Topology
from application import ApplicationSet
from file_handler import FileHandler
from users import User
from enhanced_graph import EnhancedDiGraph
from include.enums import ArrivalProcess, OnlineExpType

pl.Config.set_tbl_rows(1000)


class UsersProcessor:
    @timer('UsersProcessor')
    def __init__(self, topology: Topology, apps: ApplicationSet, number_of_requests: int, run_number: int = 0):
        self.num_of_req = number_of_requests
        self.full_user_list = []
        self.grouped_user_list = []
        self.user_demand = []
        self.assoc_nodes = []
        self.request_apps = []
        self.topology = topology
        self.apps = apps.get_apps_dict()
        self.seed = config.seed + run_number
        self.run_number = run_number

        np.random.seed(self.seed)

        self.create_user_list()

    @timer('create_user_list')
    def create_user_list(self):
        from users import RequestDemandGenerator
        dist = [app.share for app in self.apps.values()]
        self.request_apps = np.random.choice(list(self.apps.keys()), size=self.num_of_req, p=dist)
        self.create_list_of_assoc_edge_nodes(self.topology)
        request_demand = RequestDemandGenerator(self.num_of_req, self.seed).calculate_request_demand()
        self.full_user_list = [User(id=u, app_name=self.request_apps[u], demand=request_demand[u],
                                    assoc_node=self.assoc_nodes[u]) for u in range(self.num_of_req)]

    @staticmethod
    def check_request_load_on_allocated_node(requests: [], graph: EnhancedDiGraph, apps: {}):
        if not config.check_feasible_node_popularities:
            return
        from physical_graph import GraphSizeScaler
        exceeding_list = []
        app_link_sizes = {}
        app_func_sizes = {}
        total_node_demand = {}
        mean_link_size = {}
        mean_node_size = {}
        UsersProcessor.check_topology_ratios(graph)
        for app_name, app in apps.items():
            app_link_sizes[app_name] = app.graph.edges[('U', list(app.graph.successors('U'))[0])]['size']
            app_func_sizes[app_name], _ = GraphSizeScaler.get_app_sizes_nodes_links(app)
        for req in requests:
            total_node_demand[req.assoc_node] = total_node_demand.get(req.assoc_node, 0) + req.demand

        for req in requests:
            node = req.assoc_node
            demand = req.demand * Decimal(str(max(config.target_utilization)))
            relative_link_share = (demand * app_link_sizes[req.app_name]) / total_node_demand[node]
            relative_node_share = (demand * app_func_sizes[req.app_name]) / total_node_demand[node]
            mean_link_size[node] = mean_link_size.get(node, 0) + relative_link_share
            mean_node_size[node] = mean_node_size.get(node, 0) + relative_node_share

        for node in mean_link_size.keys():
            egress_link_cap = graph.get_egress_link_capacity(node)
            egress_share_on_links = mean_link_size[node] * total_node_demand[node]
            share_to_allocate_locally = egress_share_on_links - egress_link_cap
            if share_to_allocate_locally <= 0:
                continue
            node_cap = graph.get_node_capacity(node)
            share_to_allocate_locally = mean_node_size[node] * (share_to_allocate_locally / mean_link_size[node])
            remaining_share = share_to_allocate_locally - node_cap
            if remaining_share > 0:
                exceeded_cap = round(100 * (remaining_share / mean_node_size[node]) / total_node_demand[node], 0)
                exceeding_list.append(f"Error: User at {node} exceeding capacity by {exceeded_cap}%")

        if exceeding_list:
            for item in exceeding_list:
                print(item)
            # raise ValueError("Request load exceeds allocated node capacity")

    @staticmethod
    def check_topology_ratios(graph: EnhancedDiGraph):
        """For utilization 1 we want total size of request (size(R)) = cap(edge nodes) + cap(transport nodes) + cap(core nodes)
        We also approximately want size(R) <= cap(edge nodes) + cap(outgoing edge links)
        → cap(transport nodes) + cap(core nodes) <= cap(outgoing edge links)
        → Nt*Ct + Nc*Cc <= N_edge_links * Ce
        Ct = tier_ratio * Ce, Cc = tier_ratio * tier_ratio * Ce
        → tier_ratio * Ce ( Nt + tier_ratio * Nc) <= N_edge_links * Ce
        → tier_ratio * ( Nt + tier_ratio * Nc) <= N_edge_links
        assuming total edge link cap equal total edge node cap
        is required to have a non-blocked flow of demand from edge nodes"""
        EDGE = 1
        TRANSPORT = 2
        CORE = 3
        tier_count = [0] * (config.num_of_tiers + 1)
        for node in graph.nodes:
            tier = graph.get_node_tier(node)
            tier_count[tier] += 1
        if tier_count[EDGE] < config.tier_scale_ratio * (tier_count[TRANSPORT] + config.tier_scale_ratio * tier_count[CORE]):
            print(f"\nWARNING: Topology does not satisfy the required node number ratios")
        print(
            f"Edge nodes: {tier_count[EDGE]}, transport nodes: {tier_count[TRANSPORT]}, core nodes: {tier_count[CORE]}")

    def create_list_of_assoc_edge_nodes(self, topology: Topology):
        node_indices = self.get_node_index_by_popularity()
        if config.create_node_popularity_file:

            num_of_nodes = len(topology.ordered_edge_nodes)
            f1 = []
            hist, bin_edges = np.histogram(node_indices, bins=[i for i in range(0, num_of_nodes + 1)])
            for ind, node in enumerate(topology.ordered_edge_nodes):
                topology.graph.nodes[node]['popularity'] = hist[ind]
                popularity = hist[ind] / sum(hist)
                f1.append(f"{node}, {popularity}\n")
            self._export_node_popularity(f1)
        if not config.online_mode:
            for node_index in node_indices:
                self.assoc_nodes.append(topology.ordered_edge_nodes[node_index])

    @staticmethod
    def _export_node_popularity(f1: []):
        filepath = FileHandler.get_run_dir() / 'node_popularity.csv'
        with open(filepath, 'w') as f:
            f.write("node, popularity\n")
            f.writelines(f1)

    @timer('get_node_index_by_popularity')
    def get_node_index_by_popularity(self):
        """yields indexes of nodes in topology.ordered_edge_nodes by selected edge_node_popularity"""
        processes = cpu_count()
        num_edge_nodes = len(self.topology.ordered_edge_nodes)
        node_pop_gen = self.topology.node_pop_gen[self.topology.run_number]
        node_probs = self.topology.node_pop_gen[self.topology.run_number].node_probs
        if processes < 10:
            return [Topology.draw_node_by_popularity_cases(num_edge_nodes, node_pop_gen, node_probs) for _ in range(self.num_of_req)]
        else:
            from multiprocessing import Pool
            with Pool(int(cpu_count() * 0.9)) as pool:
                args_list = [(num_edge_nodes, node_pop_gen, node_probs, ) for _ in range(self.num_of_req)]
                results = pool.map(UsersProcessor.draw_node, args_list)
            return results

    @staticmethod
    def draw_node(args):
        num_edge_nodes = args[0]
        node_pop_gen = args[1]
        node_probs = args[2]
        return Topology.draw_node_by_popularity_cases(num_edge_nodes, node_pop_gen, node_probs)

    def generate_partial_request_list(self, partial_user_num_factor: float, set_id: int,
                                      num_of_sets: int = config.num_of_experiments + 1):
        """generates a partial list of users, where partial_user_num_factor is the factor of the number of users
        in the set set_id of full_user_list"""
        sliced_user_list = self._get_sliced_user_list(set_id, num_of_sets, self.full_user_list)
        request_list = sliced_user_list[:int(len(sliced_user_list) * partial_user_num_factor)]
        grouped_requests = self.generate_grouped_users(request_list)
        return request_list, grouped_requests

    def _get_sliced_user_list(self, i: int, num_of_sets: int, full_list: list):
        slice_size = len(full_list) // num_of_sets
        start_index = i * slice_size
        end_index = start_index + slice_size if i != num_of_sets - 1 else len(full_list)
        return full_list[start_index:end_index]

    def generate_grouped_users(self, user_list: []):
        """
        grouping of users that have the same application demand and they are associated to the same edge datacenter
        """
        from collections import defaultdict
        grouped_user_list = []
        combination_of_apps_edges = defaultdict(lambda: [[], 0])

        for user in user_list:
            combination_of_apps_edges = UsersProcessor.add_user_to_user_dict(user, combination_of_apps_edges)

        for app, assoc_node in combination_of_apps_edges.keys():
            num_of_users = len(combination_of_apps_edges[app, assoc_node][0])
            demand = combination_of_apps_edges.get((app, assoc_node))[1]
            g_u_id = self.create_or_get_request_id(create=True, app_node=(app, assoc_node))
            grouped_user_list.append(
                User(id=f"{g_u_id}", app_name=app, demand=demand, assoc_node=assoc_node,
                     grouped_users=num_of_users))
        return grouped_user_list

    @staticmethod
    def create_or_get_request_id(create: bool, app_node: () = None, existing_id: str = None):
        if create:
            app_name, node = app_node
            return f'{app_name}-{node}'.replace(" ", "_")
        else:
            app_name, node = existing_id.split('-')
            return app_name, node.replace('_', ' ')

    @staticmethod
    def add_user_to_user_dict(user: User, combination_of_apps_edges: {}) -> {}:
        key_pair = (user.app_name, user.assoc_node)
        combination_of_apps_edges[key_pair][0].append(user.id)
        combination_of_apps_edges[key_pair][1] += user.demand
        return combination_of_apps_edges


# class OnlineUserProcessor(UsersProcessor):
class OnlineUserProcessor():
    def __init__(self, topology: Topology, apps: ApplicationSet, run_number: int = 0, trace_special_input=None):
        if (config.request_arrival_process == ArrivalProcess.EXTERNAL and config.skip_to_exp_num is not None
                and run_number < config.skip_to_exp_num):
            return
        self.TRAIN = 'train'
        self.TEST = 'test'
        self.application_set = apps
        self.train_set_ratio = trace_special_input if config.online_exp_type == OnlineExpType.TRAIN_PERIOD_SHARE else config.train_set_ratio
        self.sim_time = config.sim_time
        self.train_duration = int(self.train_set_ratio * self.sim_time)
        self.test_duration = config.sim_time - int(config.train_set_ratio * self.sim_time)
        self.train_requests = None
        self.test_requests = {}
        self.per_timeslot_requests = None
        self.max_grouped_user_list = []
        self.train_demands = {}
        self.request_app_node = []
        self.request_working_points = {}
        self.num_train_requests = None
        self.trace_special_input = trace_special_input
        mean_req_num = OnlineHandler.get_lambda(topology.graph, lam=config.lambda_per_node) * config.sim_time

        # super().__init__(topology, apps, mean_req_num, run_number)
        self.num_of_req = mean_req_num
        self.full_user_list = []
        self.grouped_user_list = []
        self.user_demand = []
        self.assoc_nodes = []
        self.request_apps = []
        self.topology = topology
        self.apps = apps.get_apps_dict()
        self.seed = config.seed + run_number
        self.run_number = run_number

        np.random.seed(self.seed)
        self._create_online_user_list()

    def create_user_list(self, target_util):
        self.num_train_requests = self.requests.filter(pl.col('end_time') < self.train_duration).height
        if isinstance(target_util, int):
            requests = self.requests.with_columns(
                pl.col('demand').mul(target_util).cast(pl.Utf8).alias('demand')).clone()
        else:
            requests = self.requests.with_columns(
                pl.col('demand').mul(target_util).round(3).cast(pl.Utf8).alias('demand')).clone()
        self.train_demands = OnlineUserProcessor.extract_agg_demand(requests, self.train_duration)
        percentile = self.get_percentile(self.trace_special_input)
        self._create_grouped_for_offline_model(percentile)

    @staticmethod
    def get_percentile(trace_special_input):
        if config.online_exp_type == OnlineExpType.PERCENTILE:
            return trace_special_input
        return 80

    @timer("_create_online_user_list")
    def _create_online_user_list(self):
        from req_timing_gen import RequestTimingGenerator
        timing_gen = RequestTimingGenerator(self.application_set, self.topology, self.seed,
                                            self.run_number, self.trace_special_input)
        self.requests, probs = timing_gen.generate_req_timing()
        if probs is not None:
            filepath = FileHandler.get_run_dir() / 'node_popularity.csv'
            probs.write_csv(filepath)
        self._update_params_if_external()
        self._create_requests_by_working_point(timing_gen.warm_up_time)


    def _create_requests_by_working_point(self, warm_up_time):
        np.random.seed(self.seed)
        original_train_duration = int(config.train_set_ratio * self.sim_time)
        test_start_time = original_train_duration + warm_up_time
        test_end_time = test_start_time + self.test_duration
        test_requests = self.requests.lazy().filter(
            (pl.col('start_time') >= test_start_time) &
            (pl.col('start_time') < test_end_time)
        )
        test_requests = test_requests.with_columns(
            pl.col('start_time').sub(test_start_time).alias('start_time')).with_columns(
            pl.col('end_time').sub(test_start_time).alias('end_time')
        ).collect()
        shuffled_test_requests = test_requests.lazy().with_columns(
            pl.Series(np.random.permutation(test_requests.height)).alias("random_order")
        ).sort("random_order").drop("random_order").sort('start_time').drop('id').with_row_count().rename(
            {'row_nr': 'id'}).collect().clone()

        shuffled_test_requests = self._sort_by_applications(shuffled_test_requests)
        for point in config.target_utilization:
            if isinstance(point, int):
                self.test_requests[point] = shuffled_test_requests.with_columns(
                    pl.col('demand').mul(point).cast(pl.Utf8).alias('demand')).clone()
            else:
                self.test_requests[point] = shuffled_test_requests.with_columns(
                    pl.col('demand').mul(point).round(3).cast(pl.Utf8).alias('demand')).clone()

    def _sort_by_applications(self, df):
        if config.online_exp_type != OnlineExpType.CLASSES:
            return df
        return df.sort(['start_time', 'app_name']).drop('id').with_row_count().rename(
            {'row_nr': 'id'}).clone()


    def _cast_to_int64(self, df):
        df.with_columns(
            [pl.col(name).cast(pl.Int64) for name in df.columns if
             df[name].dtype == pl.Int32]
        )
        return df

    def _update_params_if_external(self):
        if config.request_arrival_process != ArrivalProcess.EXTERNAL:
            return
        self.sim_time = self.requests[self.requests.height - 1, 'start_time']
        self.train_duration = int(self.train_set_ratio * self.sim_time)
        self.test_duration = self.sim_time - self.train_duration

    def _create_grouped_for_online_mode(self):
        self.grouped_user_list = self.generate_grouped_requests_from_dict(self.per_timeslot_requests, online_model=True,
                                                                          start_time=0, end_time=self.train_duration)

    def _create_grouped_for_offline_model(self, demand_percentile):
        agg_train_demand = OnlineUserProcessor.sample_from_ci(self.train_demands, self.train_duration, self.apps,
                                                              self.topology.ordered_edge_nodes, demand_percentile)
        agg_train_demand = self._avoid_zero_demand(agg_train_demand)
        self.grouped_user_list = self.generate_grouped_requests_from_dict(agg_train_demand)
        self.shift_train_set_hotspots()

    def shift_train_set_hotspots(self):
        if not config.shift_train_set_hotspots:
            return
        print("Shifting train set hotspots")
        edge_nodes = self.topology.ordered_edge_nodes
        shifted_edge_nodes = edge_nodes[1:] + edge_nodes[:1]
        shifted_node_mapping = dict(zip(edge_nodes, shifted_edge_nodes))
        shifted_grouped_user_list = []
        for user in self.grouped_user_list:
            shited_node = shifted_node_mapping[user.assoc_node]
            g_u_id = UsersProcessor.create_or_get_request_id(create=True, app_node=(user.app_name, shited_node))
            shifted_grouped_user_list.append(
                User(id=f"{g_u_id}", app_name=user.app_name, demand=user.demand, req_demand=user.req_demand,
                     assoc_node=shited_node, grouped_users=user.grouped_users,
                     start_time=user.start_time, end_time=user.end_time))
        self.grouped_user_list = shifted_grouped_user_list

    def _avoid_zero_demand(self, agg_train_demand):
        min_non_null_val = min([x for x in agg_train_demand.values() if x > 0])
        agg_train_demand = {k: v if v > 0 else min_non_null_val for k, v in agg_train_demand.items()}
        for app_name in self.apps.keys():
            for node in self.topology.ordered_edge_nodes:
                if (app_name, node) not in agg_train_demand:
                    print(f"Warning: setting min demand for {app_name} at {node}")
                    agg_train_demand[(app_name, node)] = min_non_null_val
        return agg_train_demand

    def print_user_demand(self):
        expected_arrival_rate = sum([x.demand for x in self.expected_grouped_demand]) / (
                config.request_demand_size_mean * config.request_duration_mean)
        actual_demand = sum([x.demand for x in self.grouped_user_list]) / (
                config.request_demand_size_mean * config.request_duration_mean)
        print(
            f"Expected arrival rate: {round(expected_arrival_rate, 2)}, Actual arrival rate: {round(actual_demand, 2)} - "
            f"({round(actual_demand / expected_arrival_rate * 100, 2)}%)")

    @staticmethod
    def percentile_n(samples, n):
        return np.percentile(samples, n)

    @staticmethod
    def sample_from_ci(demand_dict, duration, apps, edge_nodes, demand_percentile):
        from scipy.stats import bootstrap
        n_resamples = 100 if cpu_count() < 16 else 1000
        apps = apps.keys()
        agg_demand = {}
        rng = np.random.RandomState(config.seed)
        for app in apps:
            for node in edge_nodes:
                demands = [demand_dict.get((app, node, t), 0) for t in range(duration)]
                res = bootstrap((demands,), lambda x: OnlineUserProcessor.percentile_n(x, demand_percentile),
                                n_resamples=n_resamples, method='percentile', confidence_level=0.95, random_state=rng)
                confidence_interval = Decimal(str(res.confidence_interval[1]))
                if confidence_interval == 0:
                    confidence_interval = Decimal(str(np.mean(demands)))
                agg_demand[(app, node)] = confidence_interval

        return agg_demand


    @staticmethod
    def generate_grouped_requests_from_df(requests: pl.DataFrame):
        request_dict = OnlineUserProcessor.aggregate_requests_by_app_node(requests)
        # t = requests[0, 'time']
        return OnlineUserProcessor.generate_grouped_requests_from_dict(request_dict)

    @staticmethod
    def generate_grouped_requests_from_dict(request_dict: {} = None, online_model=False, start_time=0, end_time=1):
        """
        grouping of users that have the same application demand and they are associated to the same edge datacenter
        """
        grouped_user_list = []
        for key, demand in request_dict.items():
            if online_model:
                app, assoc_node, req_demand = key
            else:
                app, assoc_node = key
                req_demand = 0
            num_of_users = 0
            # g_u_id = f'{app}_{assoc_node.replace(" ", "_")}'
            g_u_id = UsersProcessor.create_or_get_request_id(create=True, app_node=(app, assoc_node))
            grouped_user_list.append(
                User(id=f"{g_u_id}", app_name=app, demand=demand, req_demand=req_demand, assoc_node=assoc_node,
                     grouped_users=num_of_users, start_time=start_time, end_time=end_time))
        return grouped_user_list

    @staticmethod
    def generate_individual_requests_from_df(df, duration):
        """
        grouping of users that have the same application demand and they are associated to the same edge datacenter
        """
        requests = []
        for request in df.iter_rows(named=True):
            demand = Decimal(request['demand'])
            if request['start_time'] >= duration:
                continue
            requests.append(
                User(id=request['id'], app_name=request['app_name'], demand=demand, req_demand=demand,
                     assoc_node=request['assoc_node'], start_time=request['start_time'], end_time=request['end_time']))
        return requests

    @staticmethod
    def aggregate_requests_by_app_node(requests: pl.DataFrame):
        df = requests.groupby(['app_name', 'assoc_node']).agg(pl.col('demand').cast(pl.Float64).sum().alias('demand'))
        return {(row['app_name'], row['assoc_node']): Decimal(str(row['demand'])) for row in df.rows(named=True)}

    @staticmethod
    def extract_agg_demand(requests, duration):
        requests = requests.with_columns(pl.col('demand').cast(pl.Float64))
        per_timeslot_requests = OnlineUserProcessor.get_per_timeslot_requests(requests)
        demands = per_timeslot_requests.filter(pl.col('time') < duration)
        tuple_keys = list(zip(demands['app_name'], demands['assoc_node'], demands['time']))
        demands = dict(zip(tuple_keys, demands['demand']))
        return demands

    @staticmethod
    def get_per_timeslot_requests(requests):
        all_users = requests.with_columns(time=pl.int_ranges("start_time", "end_time"))
        demands_per_t = all_users.explode(['time'])
        demands_per_t = demands_per_t.filter(pl.col('time') >= 0)
        agg_demands = (
            demands_per_t
            .groupby(['app_name', 'assoc_node', 'time'])
            .agg(pl.col('demand').sum().alias('demand'))
        )
        return agg_demands
