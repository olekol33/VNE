import copy
import random
import multiprocessing
from decimal import Decimal
from code_testing import timer
from line_profiler_pycharm import profile
from result_exporter import UserAllocationResultExporter, UserAllocationResultExporterParams, \
    AllocationStatCollectorOnline
import time
import polars as pl
from include import config
from fluid_model import FluidModelSummary, OracleBasicModel
from online_handler import OnlineHandler, OnlineDataLogger
from file_handler import FileHandler
from allocation_logger import OnlineAllocationLogger, ReqReleaseLogger
from user_processor import OnlineUserProcessor

APP = 1
DEMAND = 2
ASSOC_NODE = 3


class OracleRunner:
    def __init__(self, exp_settings, model: FluidModelSummary, requests: pl.DataFrame, alternatives):
        self.run_number = exp_settings.run_number
        self.exp_name = exp_settings.exp_name
        self.run_settings = exp_settings
        self.branch_name = exp_settings.branch_name
        self.nrf = exp_settings.nrf
        self.erf = exp_settings.erf
        self.multiplier = model.multiplier
        self.rejected_requests = {}
        self.avg_req_cost_per_t = {}
        self.requests = requests
        self.online_logger = None
        self.basic_model_copies = []
        self.allocation_logger = OnlineAllocationLogger(exp_settings)
        self.test_duration = exp_settings.test_duration
        self.apps = OnlineHandler.remove_fic_alternatives_from_apps_dict(model.apps)
        self.alternatives = alternatives
        self.online_logger_list = []
        self.physical_graph = model.physical_graph.copy()
        self.basic_model = None
        self.stats = AllocationStatCollectorOnline(self.physical_graph, self.test_duration)
        self.dirpath = FileHandler.get_run_dir(self.run_number, self.exp_name)
        self.allocation_cost = [0] * self.test_duration
        self.rejection_cost = [0] * self.test_duration
        self.run()

    @profile
    def run(self):
        """Note we count demand as if the user arrives in each of the time slots"""
        active_requests, departures_per_timeslot = OnlineHandler.get_requests_per_timeslot(self.requests,
                                                                                           self.test_duration)
        copies = 5
        self.create_oracle_basic_model()
        start_time = time.time()
        for t, requests_t in enumerate(active_requests):
            self.basic_model_copies = self.basic_model.create_multiple_copies(self.basic_model_copies, copies)
            self.online_logger = OnlineDataLogger(self.run_settings, self.multiplier)
            active_requests_t = self.filter_rejected_requests(requests_t)
            self.oracle_deallocation_processor(departures_per_timeslot[t])
            t0 = time.time()
            fm = self.run_lp(active_requests_t)
            print(f"Oracle - Time {t} - LP time: {time.time() - t0}")
            self.oracle_allocation_processor(t, fm, active_requests_t)
            runtime = time.time() - t0
            self.online_logger.runtime = runtime
            self.online_logger_list.append(self.online_logger)
        self.run_time = time.time() - start_time
        self.export_oracle_results()

    def run_lp(self, active_requests_t):
        requests_t = self.generate_grouped_requests(active_requests_t)
        if len(requests_t) == 0:
            return None
        fm = self.basic_model_copies.pop(0)
        fm.run_model(requests_t)
        return fm

    def filter_rejected_requests(self, requests):
        rejected_ids = list(self.rejected_requests.keys())
        return requests.filter(~pl.col('id').is_in(rejected_ids))

    def oracle_deallocation_processor(self, departures):
        for request in departures.iter_rows(named=True):
            if request['id'] in self.rejected_requests:
                self.allocation_logger.add_event_entry(request['end_time'], False, False, request['id'],
                                                       request['app_name'], request['assoc_node'], request['demand'],
                                                       None)
                del self.rejected_requests[request['id']]
            else:
                self.allocation_logger.add_event_entry(request['end_time'], False, True, request['id'],
                                                       request['app_name'], request['assoc_node'], request['demand'],
                                                       None)
                demand = Decimal(request['demand'])
                self.online_logger.update_deallocation(demand)
                start_time = request['start_time']
                self.online_logger_list[start_time].update_allocation(Decimal(request['demand']))

    @profile
    def oracle_allocation_processor(self, t, fm, t_requests):
        self.online_logger.time = t
        self.online_logger.arrived_demand = t_requests.select(pl.col('demand').cast(pl.Float64))['demand'].sum()
        self.online_logger.requests = len(t_requests.filter(pl.col('start_time') == t))
        self.online_logger.expected_active_demand = self.requests.filter(
            (pl.col('start_time') <= t) & (pl.col('end_time') > t)).select(
            pl.col('demand').cast(pl.Float64))['demand'].sum()
        self.allocation_cost[t] = 0 if fm is None else fm.cost
        rejected = self.get_rejected_requests(fm, t_requests)
        for request in rejected:
            self.allocation_logger.add_event_entry(t, True, False, request['id'],
                                                   request['app_name'], request['assoc_node'], request['demand'], None)
            self.rejected_requests[request['id']] = request
        self.handle_costs(t, rejected)

    def handle_costs(self, t, rejected):
        self.get_mean_allocation_cost(t, rejected)
        self.count_costs(t, rejected)

    def get_mean_allocation_cost(self, t, rejected):
        num_rejected = len(rejected)
        if (self.online_logger.requests - num_rejected) == 0:
            print(f"WARNING: Oracle Time: {t} - Requests: {self.online_logger.requests} - Rejected: {num_rejected}")
            self.avg_req_cost_per_t[t] = 0
        else:
            self.avg_req_cost_per_t[t] = self.allocation_cost[t] / (self.online_logger.requests - num_rejected)

    def count_costs(self, t, rejected):
        for request in rejected:
            end_time = min(request['end_time'], self.test_duration)
            for t_req in range(request['start_time'], end_time):
                app_sizes = self.apps[request['app_name']].get_app_size()
                base_rejection_cost = self.multiplier.get_base_rejection_cost(app_sizes)
                self.rejection_cost[t_req] += (Decimal(request['demand']) * base_rejection_cost)
            for t_req in range(request['start_time'], t):
                self.allocation_cost[t_req] -= self.avg_req_cost_per_t[t_req]

    @staticmethod
    def get_rejected_requests(fm, requests):
        app_node_share = {}
        if fm is None:
            return []
        non_zero_vars = fm.non_zero_vars
        for var, value in non_zero_vars.items():
            app_name, alt, assoc_node, i, j, m, n = var
            if i == config.user_func and m == n and m == assoc_node:
                app_node_share[(app_name, assoc_node)] = value
        rejected = []
        for request in requests.iter_rows(named=True):
            allocated_share = app_node_share.get((request['app_name'], request['assoc_node']), 0)
            demand = Decimal(request['demand'])
            if allocated_share < demand:
                rejected.append(request)
            else:
                app_node_share[(request['app_name'], request['assoc_node'])] -= demand

        return rejected

    @timer('create_oracle_basic_model')
    def create_oracle_basic_model(self):
        requests = self.generate_all_possible_grouped_requests()
        self.basic_model = OracleBasicModel(self.physical_graph, requests, self.apps, self.alternatives,
                                            self.run_settings, dirpath=self.dirpath, oracle=True)

    def generate_all_possible_grouped_requests(self):
        data = []
        for node in self.physical_graph.get_tier_nodes(1):
            for app_name in self.apps.keys():
                data.append([app_name, 0, node])
        df = pl.DataFrame(data, schema=['app_name', 'demand', 'assoc_node'])
        return OnlineUserProcessor.generate_grouped_requests_from_df(df)

    def generate_grouped_requests(self, active_requests):
        return OnlineUserProcessor.generate_grouped_requests_from_df(active_requests)

    def export_oracle_results(self):
        self.total_demand = 1
        self.allocated_demand = sum([demand for demand in self.stats.allocated_demand.values()])
        self.unallocated_demand = self.total_demand - self.allocated_demand
        self.allocated_requests = 0
        self.cost = 0
        self.node_utilization = 0
        self.link_utilization = 0
        self.allocation_logger.export(self.dirpath)
        params = UserAllocationResultExporterParams.take_input(self)
        UserAllocationResultExporter(params)
        logger_list = []
        for logger in self.online_logger_list:
            if len(logger_list) == 0:
                logger.update_active_demand()
            else:
                logger.update_active_demand(logger_list[-1])
            logger_list.append(logger.class_to_dict())
        df = OnlineDataLogger.get_online_df(logger_list, self.run_settings, export=False)
        ReqReleaseLogger(self.run_settings).post_process_cost_from_lists(df, self.allocation_cost, self.rejection_cost)