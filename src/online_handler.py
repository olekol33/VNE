from include.enums import OnlineExpType
import copy
import os
from file_handler import FileHandler
from decimal import Decimal
from include import config
from dataclasses import dataclass
from users import User
from collections import deque
import polars as pl
import pandas as pd


class OnlineHandler:

    @staticmethod
    def get_requests_per_timeslot(requests, test_duration):
        requests = requests.clone()
        requests = requests.with_columns(time=pl.int_ranges("start_time", "end_time"))
        requests_exploded = requests.explode(['time'])
        requests_per_timeslot = []
        departures_per_timeslot = []
        for t in range(0, test_duration):
            df = requests_exploded.filter(pl.col('time') == t).clone()
            df = df.with_columns([pl.lit(t).cast(pl.Int64).alias('time')])
            requests_per_timeslot.append(df)
            df = requests.filter(pl.col('end_time') == t).clone()
            departures_per_timeslot.append(df)
        return requests_per_timeslot, departures_per_timeslot

    @staticmethod
    def get_total_demand_per_timeslot(requests, test_duration):
        total_demand_per_timeslot = []
        requests_per_timeslot, _ = OnlineHandler.get_requests_per_timeslot(requests, test_duration)
        for t_requests in requests_per_timeslot:
            t_requests = t_requests.with_columns(pl.col('demand').cast(pl.Decimal))
            total_demand = t_requests['demand'].sum()
            total_demand_per_timeslot.append(total_demand)
        return total_demand_per_timeslot

    @staticmethod
    def get_total_share_per_timeslot(requests, test_duration, apps):
        total_share_per_timeslot = {}
        total_share_per_timeslot_per_node = {}
        requests_per_timeslot, _ = OnlineHandler.get_requests_per_timeslot(requests, test_duration)
        for t, t_requests in enumerate(requests_per_timeslot):
            t_requests = t_requests.with_columns(pl.col('demand').cast(pl.Float64))
            grouped_df = OnlineHandler.group_and_add_app_size_to_df(t_requests, 'app_name', apps)
            total_share = grouped_df['share'].sum()
            total_share_per_timeslot[t] = total_share
            grouped_df = OnlineHandler.group_and_add_app_size_to_df(t_requests, ['app_name', 'assoc_node'], apps)
            key_column = grouped_df['assoc_node'].to_list()
            value_column = grouped_df['share'].to_list()
            res_dict = {k: v for k, v in zip(key_column, value_column)}
            total_share_per_timeslot_per_node[t] = res_dict
        return total_share_per_timeslot, total_share_per_timeslot_per_node

    @staticmethod
    def group_and_add_app_size_to_df(df, group, apps):
        grouped_df = df.groupby(group).agg(pl.col('demand').sum())
        app_sizes = OnlineHandler.get_app_sizes(grouped_df['app_name'].unique().to_list(), apps)
        grouped_df = grouped_df.with_columns(pl.col("app_name").alias('app_size'))
        df = grouped_df.with_columns(pl.col("app_size").map_dict(app_sizes))
        df = df.with_columns((pl.col("demand") * pl.col("app_size")).alias("share"))
        return df

    @staticmethod
    def get_app_sizes(app_names, apps):
        app_sizes = {}
        for app_name in app_names:
            app_sizes[app_name], _ = apps[app_name].get_app_size()
        return app_sizes

    @staticmethod
    def get_request_arrivals_and_departures(requests, test_duration, limited_test_duration=None):
        arrivals = {t: [] for t in range(test_duration)}
        departures = {t: [] for t in range(test_duration)}
        for row in requests.rows(named=True):
            request = User(id=row['id'], app_name=row['app_name'], demand=Decimal(row['demand']),
                           assoc_node=row['assoc_node'],
                           start_time=row['start_time'], end_time=row['end_time'])
            if limited_test_duration is not None and request.start_time >= limited_test_duration:
                continue
            arrivals[request.start_time].append(request)
            if request.end_time < test_duration:
                departures[request.end_time].append(request)

        return arrivals, departures

    @staticmethod
    def get_test_request_demand(request):
        end_time = min(config.sim_time, request.end_time)
        return request.demand * (end_time - request.start_time)

    @staticmethod
    def export_online_stats(stats, nodes, file_path, alg):
        if not config.online_mode:
            return
        duration = len(stats.rejected_user_rate_per_t)
        for t in range(duration):
            arrivals = sum(stats.arrivals[t].values())
            arrivals_demand = sum(stats.arrivals_demand[t].values())
            allocated_users = sum(stats.allocated_users_per_t[t].values())
            allocated_demand_per_t = sum(stats.allocated_demand_per_t[t].values())
            stats.rejected_user_rate_per_t[t] = round((arrivals - allocated_users) / arrivals, 3) if arrivals > 0 else 0
            stats.rejected_demand_rate_per_t[t] = round((arrivals_demand - allocated_demand_per_t) / arrivals_demand,
                                                        3) if arrivals_demand > 0 else 0

        header = ("algorithm,time,node,arrivals_users,arrivals_demand,departures_users,departures_demand,"
                  "allocated_users,allocated_demand,deallocated_users,rejected_demand_rate,rejected_users_rate\n")
        with open(file_path, "w") as f:
            f.write(header)
            for t in range(len(stats.rejected_user_rate_per_t)):
                for node in nodes:
                    rejected_user_rate = round(
                        (stats.arrivals[t][node] - stats.allocated_users_per_t[t][node]) / stats.arrivals[t][node],
                        3) if stats.arrivals[t][node] > 0 else 0
                    rejected_demand_rate = round(
                        (stats.arrivals_demand[t][node] - stats.allocated_demand_per_t[t][node]) /
                        stats.arrivals_demand[t][node], 3) if stats.arrivals_demand[t][node] > 0 else 0
                    f.write(f"{alg},{t},{node},{stats.arrivals[t][node]},{stats.arrivals_demand[t][node]},"
                            f"{stats.departures[t][node]},{stats.departures_demand[t][node]},"
                            f"{stats.allocated_users_per_t[t][node]},{stats.allocated_demand_per_t[t][node]},"
                            f"{stats.deallocated_users[t][node]},{rejected_demand_rate},{rejected_user_rate}\n")
                f.write(f"{alg},{t},agg,{sum(stats.arrivals[t].values())},{sum(stats.arrivals_demand[t].values())},"
                        f"{sum(stats.departures[t].values())},{sum(stats.departures_demand[t].values())},"
                        f"{sum(stats.allocated_users_per_t[t].values())},{sum(stats.allocated_demand_per_t[t].values())},"
                        f"{sum(stats.deallocated_users[t].values())},{stats.rejected_demand_rate_per_t[t]},{stats.rejected_user_rate_per_t[t]}\n")

    @staticmethod
    def remove_fic_alternatives_from_apps_dict(apps):
        no_fic_apps = {}
        for app_name, app in apps.items():
            app_cp = copy.deepcopy(app)
            app_cp.remove_fic_alternatives()
            no_fic_apps[app_name] = app_cp
        return no_fic_apps

    @staticmethod
    def get_lambda(physical_graph, lam):
        nodes = physical_graph.get_num_of_nodes()
        return lam * nodes

    @staticmethod
    def check_fixed_capacities():
        if config.use_app_generator:
            return
        mean_app_size = config.agen_func_mean * ((config.agen_app_max_length + config.agen_app_min_length) / 2)
        mean_req_size_per_node = (config.lambda_per_node * mean_app_size * config.request_duration_mean *
                                  config.request_demand_size_mean)
        if mean_req_size_per_node > config.base_node_capacity:
            print(f"WARNING: Mean request size per node: {int(mean_req_size_per_node)} is greater than base node capacity: "
                  f"{int(config.base_node_capacity)}\n"
                  f"mean_app_size: {mean_app_size}, lambda_per_node: {config.lambda_per_node}, request_duration_mean: "
                  f"{config.request_duration_mean}, demand_size_mean: {config.request_demand_size_mean}")

        mean_req_size_per_link = (config.lambda_per_node * config.agen_link_mean * config.request_duration_mean *
                                  config.request_demand_size_mean)
        if mean_req_size_per_link > config.base_link_capacity:
            print(f"WARNING: Mean request size per link: {int(mean_req_size_per_link)} is greater than base link capacity: "
                  f"{int(config.base_link_capacity)}\n"
                  f"lambda_per_node: {config.lambda_per_node}, agen_link_mean: {config.agen_link_mean}, request_duration_mean: "
                  f"{config.request_duration_mean}, demand_size_mean: {config.request_demand_size_mean}")

    @staticmethod
    def split_to_unique_timeslots(df, start_name='Arrival', end_name='Departure'):
        df['active'] = [list(range(start, end)) for start, end in zip(df[start_name], df[end_name])]
        df_active = df.explode('active')
        df_grouped = df.groupby(start_name)
        df_exploded_grouped = df_active.groupby('active')
        return df_grouped, df_exploded_grouped







class OnlineDataLogger:
    def __init__(self, run_settings, multiplier):
        self.run_number = run_settings.run_number
        self.multiplier = multiplier
        self.target_utilization = run_settings.target_utilization
        self.hike = run_settings.hike
        self.special_input = run_settings.special_input
        self.test_duration = run_settings.test_duration
        self.time: int = -1
        self.requests: int = 0
        self.allocated_requests: int = 0
        self.arrived_demand = Decimal('0')
        self.allocated_arrived_demand = Decimal('0')
        self.deallocated_demand = Decimal('0')
        self.deallocated_requests = Decimal('0')
        self.allocated_active_demand = Decimal('0')
        self.allocated_active_requests = Decimal('0')
        self.total_allocated_requests = Decimal('0')
        self.expected_allocated_requests = Decimal('0')
        self.expected_active_demand = Decimal('0')
        self.allocation_cost = Decimal('0')
        self.rejection_cost = Decimal('0')
        self.preempted_requests: int = 0
        self.preempted_demand = Decimal('0')
        self.runtime = 0

    def update_deallocation(self, demand):
        self.deallocated_demand += demand
        self.deallocated_requests += 1

    def update_allocation(self, demand):
        self.allocated_arrived_demand += demand
        self.allocated_requests += 1

    def update_active_demand(self, previous_logger=None):
        if previous_logger is None:
            previously_active_demand = 0
            previously_active_requests = 0
            previously_total_allocated_requests = 0
            previously_expected_allocated_requests = 0
        else:
            previously_active_demand = previous_logger['allocated_active_demand']
            previously_active_requests = previous_logger['allocated_active_requests']
            previously_total_allocated_requests = previous_logger['total_allocated_requests']
            previously_expected_allocated_requests = previous_logger['expected_allocated_requests']
        self.allocated_active_demand = previously_active_demand + self.allocated_arrived_demand - self.deallocated_demand
        self.allocated_active_requests = previously_active_requests + self.allocated_requests - self.deallocated_requests
        self.total_allocated_requests = previously_total_allocated_requests + self.allocated_requests
        self.expected_allocated_requests = previously_expected_allocated_requests + self.requests

    def class_to_dict(self):
        return self.__dict__

    @staticmethod
    def log_timeslot(t, arrivals, online_logger, online_logger_list, demand_per_timeslot, runtime):
        online_logger.time = t
        online_logger.expected_active_demand = demand_per_timeslot[t]
        online_logger.requests = len(arrivals[t])
        online_logger.runtime = runtime
        previous_logger = None if len(online_logger_list) == 0 else online_logger_list[-1]
        online_logger.update_active_demand(previous_logger)
        online_logger_list.append(online_logger.class_to_dict())

    @staticmethod
    def get_special_exp_string():
        special_in_str = 'None'
        if config.online_exp_type == OnlineExpType.NUMBER_OF_APPS:
            special_in_str = 'Napps'
        elif config.online_exp_type == OnlineExpType.LENGTH_OF_APPS:
            special_in_str = 'Lapps'
        elif config.online_exp_type == OnlineExpType.LINK_CAPACITY:
            special_in_str = 'Edge Link Capacity'
        elif config.online_exp_type == OnlineExpType.PERCENTILE:
            special_in_str = 'Percentile'
        elif config.online_exp_type == OnlineExpType.CLASSES:
            special_in_str = 'Layers'
        elif config.online_exp_type == OnlineExpType.MMPP_PARAMS:
            special_in_str = 'Cold-Hot Demand Ratio'
        elif config.online_exp_type == OnlineExpType.REQUEST_DURATION:
            special_in_str = 'Duration'
        elif config.online_exp_type == OnlineExpType.TRAIN_PERIOD_SHARE:
            special_in_str = 'Train Share'
        elif config.online_exp_type == OnlineExpType.DIFFERENT_HOTSPOTS:
            pass
        elif config.online_exp_type == OnlineExpType.SAME_HOTSPOTS:
            pass
        elif config.online_exp_type == OnlineExpType.LAMBDA:
            special_in_str = 'Lambda'
        elif config.online_exp_type == OnlineExpType.DIFFERENT_APPS:
            pass
        else:
            print(f'Unknown online_exp_type: {config.online_exp_type}')
        return special_in_str

    @staticmethod
    def get_online_df(logger_list, run_settings, export=True):
        df = pd.DataFrame(logger_list)
        special_in_str = OnlineDataLogger.get_special_exp_string()

        df.rename(columns={'run_number': 'Run Number', 'time': 'Time', 'target_utilization': 'Target Utilization',
                           'hike': 'Hike', 'special_input': f'{special_in_str}', 'requests': 'Requests',
                           'allocated_requests': 'Allocated Requests', 'arrived_demand': 'Arrived Demand',
                           'allocated_arrived_demand': 'Allocated Arrived Demand',
                           'deallocated_demand': 'Deallocated Demand',
                           'deallocated_requests': 'Deallocated Requests',
                           'allocated_active_demand': 'Allocated Active Demand',
                           'allocated_active_requests': 'Allocated Active Requests',
                           'total_allocated_requests': 'Total Allocated Requests',
                           'expected_active_demand': 'Expected Active Demand',
                           'expected_allocated_requests': 'Expected Allocated Requests',
                           'preempted_requests': 'Preempted Requests',
                           'preempted_demand': 'Preempted Demand',
                           'allocation_cost': 'Allocation Cost',
                           'rejection_cost': 'Rejection Cost', 'runtime': 'Runtime'}, inplace=True)
        #round columns
        df['Arrived Demand'] = df['Arrived Demand'].apply(lambda x: round(x, 3))
        df['Allocated Arrived Demand'] = df['Allocated Arrived Demand'].apply(lambda x: round(x, 3))
        df['Deallocated Demand'] = df['Deallocated Demand'].apply(lambda x: round(x, 3))
        df['Allocated Active Demand'] = df['Allocated Active Demand'].apply(lambda x: round(x, 3))
        df['Expected Active Demand'] = df['Expected Active Demand'].apply(lambda x: round(x, 3))
        df['Preempted Demand'] = df['Preempted Demand'].apply(lambda x: round(x, 3))
        df['Allocation Cost'] = df['Allocation Cost'].apply(lambda x: round(x, 3))
        df['Rejection Cost'] = df['Rejection Cost'].apply(lambda x: round(x, 3))
        df['Runtime'] = df['Runtime'].apply(lambda x: round(x, 5))

        to_drop = ['multiplier', 'test_duration']
        if special_in_str == 'None':
            to_drop.append(special_in_str)
        if run_settings.hike == -1:
            to_drop.append('Hike')
        df = df.drop(columns=to_drop)
        if export:
            OnlineDataLogger.export(run_settings, df)
        return df

    @staticmethod
    def export(run_settings, df):
        dir = FileHandler.get_run_dir('summary', 'online')
        filename = f"{run_settings.branch_name}.csv"
        filepath = f"{dir}/{filename}"
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)


class TestUtilizationAnalyzer:
    def __init__(self, fm, apps, requests, test_duration, run_number, exp_name):
        filepath = FileHandler.get_run_dir(run_number, exp_name) / "test_utilization.txt"
        if filepath.exists():
            return
        total_share_per_timeslot, total_share_per_timeslot_per_node = OnlineHandler.get_total_share_per_timeslot(
            requests, test_duration, apps)
        total_node_cap = fm.physical_graph.get_total_node_capacity(avoid_gpu=False)
        requests_per_second = len(requests) / test_duration
        data = []
        data.append(f'Requests per second: {requests_per_second}\n')
        for t in range(len(total_share_per_timeslot)):
            share = total_share_per_timeslot[t]
            utilization = Decimal(str(share)) / total_node_cap
            note = '>>' if utilization > 1 else ''
            data.append(f"\n{note}Time: {t}, Total Share: {share}, Total Node Capacity: {total_node_cap}, Utilization: "
                    f"{round(utilization, 2)}\n")
            node_share_dict = total_share_per_timeslot_per_node[t]
            for node, node_share in node_share_dict.items():
                node_cap = fm.physical_graph.get_node_capacity(node) + fm.physical_graph.get_egress_link_capacity(node)
                node_utilization = Decimal(str(node_share)) / node_cap
                note = '>>' if node_utilization > 1 else ''
                data.append(f"{note}Node: {node}, Node Share: {node_share}, Node Capacity: {node_cap}, Utilization: "
                        f"{round(node_utilization, 2)}\n")
        #write to file
        with open(filepath, "w") as f:
            f.writelines(data)
    @staticmethod
    def process_timeslot(t, total_share_per_timeslot, total_node_cap, total_share_per_timeslot_per_node, filepath, fm):
        share = total_share_per_timeslot[t]
        utilization = Decimal(str(share)) / total_node_cap
        note = '>>' if utilization > 1 else ''
        with open(filepath, "a") as f:
            f.write(f"\n{note}Time: {t}, Total Share: {share}, Total Node Capacity: {total_node_cap}, Utilization: "
                    f"{round(utilization, 2)}\n")
        node_share_dict = total_share_per_timeslot_per_node[t]
        for node, node_share in node_share_dict.items():
            node_cap = fm.physical_graph.get_node_capacity(node) + fm.physical_graph.get_egress_link_capacity(node)
            node_utilization = Decimal(str(node_share)) / node_cap
            note = '>>' if node_utilization > 1 else ''
            with open(filepath, "a") as f:
                f.write(f"{note}Node: {node}, Node Share: {node_share}, Node Capacity: {node_cap}, Utilization: "
                        f"{round(node_utilization, 2)}\n")


class EfficientDeque:
    def __init__(self):
        self.deque = deque()
        self.dictionary = {}

    def is_in_deque(self, key):
        return key in self.dictionary

    def append(self, key, value):
        if key not in self.dictionary:
            self.deque.append(key)
        self.dictionary[key] = value
        assert len(self.deque) == len(self.dictionary)

    def appendleft(self, key, value):
        if key not in self.dictionary:
            self.deque.appendleft(key)
        self.dictionary[key] = value

    def pop(self, id):
        for key in reversed(self.deque):
            if key == id:
                self.deque.remove(key)
                del self.dictionary[key]
                assert key == id
                assert len(self.deque) == len(self.dictionary)
                return
        raise ValueError("key not found")

    def get_value_sum(self):
        return sum(self.dictionary.values())

    def get_num_items(self):
        return len(self.dictionary)

    def get(self, key):
        return self.dictionary.get(key)

    def update(self, key, value):
        if key in self.dictionary:
            self.dictionary[key] += value
        else:
            self.append(key, value)
            assert len(self.deque) == len(self.dictionary)

    def peek(self, last_n):
        # Peek at the n from right item without removing it
        item = self.deque[-1 - last_n]
        value = self.dictionary[item]
        return item, value

    def get_length(self):
        return len(self.deque)

    def keys_to_list(self):
        return list(self.dictionary.keys())


@dataclass
class TraceSpecialInput:
    node: str = ''
    hike: float = -1
    intensity: float = -1
