from functools import lru_cache
from element_translator import ElementTranslator
from application import Application
from decimal import Decimal
import polars as pl
from comparison_algorithms import EmbeddingAlgorithm
import threading
import time
from include import config
from fluid_model import FluidModelSummary
from file_handler import FileHandler
from pathlib import Path
from code_testing import timer
from result_exporter import UserAllocationResultExporter, UserAllocationResultExporterParams, AllocationStatCollector, \
    AllocationStatCollectorOnline
from allocation_logger import AllocationGraph
from online_handler import OnlineHandler, OnlineDataLogger
from physical_graph import EnhancedDiGraph, GraphSizeScaler
from line_profiler_pycharm import profile

NO_LINK = 'NO_LINK'
file_lock = threading.Lock()


class HeuristicRunner:
    def __init__(self, fm: FluidModelSummary, requests, alt_combinations=None, run_settings=None):
        self.requests_by_app_node = {}
        self.allocation: {} = fm.allocation.copy()
        self.requests = requests
        self.stats = AllocationStatCollector(fm.physical_graph) if not config.online_mode else (
            AllocationStatCollectorOnline(fm.physical_graph, run_settings.test_duration))
        self.fm = fm
        self.apps = fm.apps
        self.alt_combinations = alt_combinations
        self.physical_graph = fm.physical_graph
        self.run_settings = run_settings
        self.allocated_demand = 0
        self.allocated_virtual_nodes = 0
        self.allocated_virtual_links = 0
        self.total_demand = 0
        self.allocated_requests = 0
        self.cost = 0
        self.unallocated_demand = 0
        self.run_time = fm.runtime
        if run_settings is not None:
            print(
                f"Experiment {run_settings.exp_name}, NRF {run_settings.nrf}, ERF {run_settings.erf}  is running heuristic")

        self._get_users_by_app_node()
        self.run_settings = run_settings
        if config.log_alternatives:
            with open(self._get_scenario_path() / "heuristic_branch_share.txt", "w") as f:
                f.write(f"app_name,user_node,dest_node,choice_node,branch_name,branch_num,share\n")
        start_time = time.time()
        self._run_heuristic()
        end_time = time.time()
        self.run_time = self.run_time + end_time - start_time
        self.export_heuristic_results()
        if config.log_alternatives:
            with open(self._get_scenario_path() / "heuristic_branch_share.txt", "a") as f:
                f.write(f"METADATA\n")
                f.write(f"{run_settings.branch_name},{len(requests)},{self.run_settings.total_demand},{self.cost}\n")

    def _get_users_by_app_node(self):
        self.requests_by_app_node = {}
        if config.online_mode:
            return
        else:
            for app_name, alt_name, assoc_node in self.allocation.keys():
                self.requests_by_app_node[(app_name, assoc_node)] = []
            for request in self.requests:
                self.requests_by_app_node[(request.app_name, request.assoc_node)].append(request)


    @timer("_run_heuristic_in_parallel")
    def _run_heuristic(self):
        if config.online_mode:
            online_vne = OnlineAllocationHeuristic(self.fm, self.requests, self.run_settings, self.alt_combinations)
            self.allocated_demand = online_vne.allocated_demand
            self.allocated_virtual_nodes = online_vne.allocated_virtual_nodes
            self.allocated_virtual_links = online_vne.allocated_virtual_links
            self.total_demand = online_vne.total_demand
            self.unallocated_requests = len(online_vne.unallocated_requests)
            self.cost = online_vne.cost
            self.unallocated_demand = online_vne.unallocated_demand
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.requests_by_app_node)) as executor:
            futures = []
            for requests in self.requests_by_app_node.values():
                future = executor.submit(SplitPerUserAllocationHeuristic, self.fm, requests,
                                         run_settings=self.run_settings)
                futures.append(future)
            for future in futures:
                allocation = future.result()
                self._aggregate_stats(allocation)
                self.allocated_demand += allocation.allocated_demand
                self.allocated_virtual_nodes += allocation.allocated_virtual_nodes
                self.allocated_virtual_links += allocation.allocated_virtual_links
                self.total_demand += allocation.total_demand
                self.allocated_requests += self.stats.allocated_requests_per_node[allocation.assoc_node]
                self.cost += allocation.cost
                self.unallocated_demand += allocation.unallocated_demand

    def _aggregate_stats(self, allocation):
        stats = allocation.stats
        node = allocation.assoc_node
        for attr_name in stats.__dict__:
            if attr_name in self.stats.__dict__:
                agg_stat = getattr(self.stats, attr_name)
                local_stat = getattr(stats, attr_name)
                if isinstance(agg_stat, list):
                    if isinstance(agg_stat[0], dict):
                        for t in range(len(agg_stat)):
                            agg_stat[t][node] += local_stat[t][node]
                else:
                    agg_stat[node] += local_stat[node]

    def export_heuristic_results(self):
        graph = self.fm.physical_graph

        self.node_utilization, self.link_utilization = AllocationGraph.get_utilization(
            graph, self.allocated_virtual_nodes, self.allocated_virtual_links)
        params = UserAllocationResultExporterParams.take_input(self)
        UserAllocationResultExporter(params)

        file_path = self._get_scenario_path() / "heuristic_online_stats.csv"
        OnlineHandler.export_online_stats(self.stats, graph.get_nodes(), file_path, 'Heuristic')

    def _get_scenario_path(self) -> Path:
        """
        Returning the exact location to store the performance metrics
        """
        run_number = self.run_settings.run_number
        scenario_path = FileHandler.get_run_dir(run_number)
        scenario_path = scenario_path / self.run_settings.exp_name
        return scenario_path


class AllocationHeuristic:
    def __init__(self, fm: FluidModelSummary, requests, run_settings=None):
        self.allocation: {} = fm.allocation.copy()
        self.graph: EnhancedDiGraph = fm.physical_graph
        self.requests: [] = requests
        self.solution_cost = fm.cost
        self.multiplier = fm.multiplier
        self.apps = Application.remove_fic_alternatives_from_dict(fm.apps)
        self.run_number = run_settings.run_number
        self.run_settings = run_settings
        self.cost = 0
        self.stats = self.get_stats_collector()
        self.node_utilization = 0
        self.link_utilization = 0
        self.unallocated_requests = {}
        self.total_demand = 0
        self.allocated_demand = 0
        self.unallocated_demand = 0
        self.allocated_on_fic = 0
        self.demand_on_zipped = 0
        self.allocated_virtual_nodes = 0
        self.allocated_virtual_links = 0
        self.run_time = fm.runtime
        self.finished_allocating = False

        self.execute_heuristic()

    def get_stats_collector(self):
        return AllocationStatCollector(self.graph)

    @profile
    def execute_heuristic(self):
        self.f1_list = []
        self.f2_list = []
        self.f3_list = []
        start_time = time.time()
        allocation_graphs = self.get_node_allocation_graphs(self.requests[0].app_name,
                                                                    self.requests[0].assoc_node)
        assoc_node = self.requests[0].assoc_node
        req_allocation = allocation_graphs[0]
        path, available_demand = req_allocation.run_path_finder()
        fully_allocated_demand = 0
        for request in self.requests:
            demand = request.demand
            self.stats.increment_user_per_node(assoc_node)
            self.stats.total_demand[assoc_node] += demand
            self.total_demand += demand
            if path is None or self.finished_allocating:
                self.unallocated_requests[request.id] = demand
                self.unallocated_demand += demand
                continue
            # if available_demand < demand:
            #     path, available_demand = req_allocation.run_path_finder()
            #     if available_demand < demand and len(allocation_graphs) > 1:
            #         allocation_graphs = allocation_graphs[1:]
            #         req_allocation = allocation_graphs[0]
            #         path, available_demand = req_allocation.run_path_finder()

            if available_demand < demand:
                fully_allocated_demand += available_demand
                self._allocate_partial_demand(available_demand, request, path, req_allocation, fully_allocated_demand)
                fully_allocated_demand = 0
                if self.finished_allocating:
                    if len(allocation_graphs) > 1:
                        allocation_graphs = allocation_graphs[1:]
                        req_allocation = allocation_graphs[0]
                        self.finished_allocating = False
                    else:
                        continue
                path, available_demand = req_allocation.run_path_finder()
            else:
                self._allocate_entire_demand(request, path, req_allocation)
                available_demand -= demand
                fully_allocated_demand += demand
        self.run_time = self.run_time + time.time() - start_time
        self._write_to_files(self._get_scenario_path())

    def get_node_allocation_graphs(self, app_name, assoc_node):
        allocation_graphs = []
        for k, v in self.allocation.items():
            k_app_name, _, k_assoc_node = k
            if k_app_name == app_name and k_assoc_node == assoc_node:
                if v.get_weight():
                    # allocation_graphs.append((v, v.get_weight()))
                    allocation_graphs.append(v)
        return allocation_graphs

    @profile
    def _allocate_entire_demand(self, user, path, req_allocation):
        self.allocate(user.demand, path, user, req_allocation)
        self.allocated_demand += user.demand
        self.stats.increment_allocated_demand(user.assoc_node, user.demand)
        self.stats.increment_allocated_requests_per_node(user.assoc_node)

    @profile
    def _allocate_partial_demand(self, share, request, path, req_allocation, fully_allocated_demand):
        self.finished_allocating = self.allocate(share, path, request, req_allocation, fully_allocated_demand=fully_allocated_demand)
        self.unallocated_demand += request.demand - share

    def _get_scenario_path(self) -> Path:
        """
        Returning the exact location to store the performance metrics
        """
        scenario_path = FileHandler.get_run_dir(self.run_number)
        scenario_path = scenario_path / self.run_settings.exp_name
        return scenario_path

    @profile
    def allocate(self, demand: float, path: [], request, req_allocation, fully_allocated_demand: float = 0):
        fully_allocated = True if fully_allocated_demand == 0 else False
        fully_allocated_demand = demand if config.online_mode or fully_allocated else fully_allocated_demand
        zero_share_edges = []
        allocation_cost = 0
        for path_object in path:
            for element in path_object.elements:
                multiplier = req_allocation.get_link_multiplier_for_aloc_element(element)
                share = demand * multiplier
                i, j, m, n = element
                if fully_allocated:
                    allocation_cost += share * self.graph.get_element_cost(m, n)
                    self.stats.cost_per_assoc_node[request.assoc_node] += allocation_cost
            if fully_allocated_demand > 0:
                fully_allocated_share = (fully_allocated_demand *
                                         ElementTranslator.phys_g_elem_to_alloc_g_mult(req_allocation,
                                                                                       path_object.elements))
                zero_share_edges = req_allocation.update_aloc_graph_link_capacity(
                    fully_allocated_share, path_object.link, path_object.link_share_capacity,
                    zero_share_edges=zero_share_edges)
        self.cost += allocation_cost
        if len(zero_share_edges) == 0 or config.online_mode:
            return False
        else:
            return req_allocation.remove_zero_capacity_elements(zero_share_edges)

    @profile
    def _write_to_files(self, scenario_path):
        if config.log_alternatives:
            with file_lock:
                with open(scenario_path / "heuristic_branch_share.txt", "a") as f:
                    f.writelines(self.f3_list)
        if not config.advanced_logging_mode:
            return
        with file_lock:
            with open(scenario_path / "heuristic_solution_share.txt", "a") as f1, \
                    open(scenario_path / "heuristic_solution_cost.txt", "a") as f2:
                f1.write("user_id, app_name, src, f1, f2, m, n, share\n")
                f2.write("user_id, app_name, src, f1, f2, m, n, cost\n")
                f1.writelines(self.f1_list)
                f2.writelines(self.f2_list)


class SplitPerUserAllocationHeuristic(AllocationHeuristic):
    def __init__(self, fm: FluidModelSummary, requests, run_settings=None):
        self.apps = fm.apps
        self.assoc_node = requests[0].assoc_node
        super().__init__(fm, requests, run_settings)


class OnlineAllocationHeuristic(AllocationHeuristic):
    def __init__(self, fm: FluidModelSummary, requests, run_settings=None, alt_combinations=None):
        from allocation_logger import ReqReleaseLogger, OnlineAllocationLogger
        self.arrivals = {}
        self.departures = {}
        self.online_logger_list = []
        self.alt_combinations = alt_combinations
        self.active_guaranteed_user_paths = {}
        self.active_non_guaranteed_user_paths = {}
        self.test_duration = run_settings.test_duration
        self.apps = Application.remove_fic_alternatives_from_dict(fm.apps)
        self.arrivals, self.departures = OnlineHandler.get_request_arrivals_and_departures(requests, self.test_duration)
        self.non_guaranteed_arrivals = []
        self.requests_costs = {}
        self.enable_preempt = config.enable_preempt
        self.id_to_request = self._create_id_to_request()
        self.allocation_logger = OnlineAllocationLogger(run_settings)
        self.req_logger = ReqReleaseLogger(run_settings)
        self.demand_per_timeslot = OnlineHandler.get_total_demand_per_timeslot(requests, self.test_duration)
        self.residual_graph = fm.physical_graph.copy()
        self.non_guaranteed_dealloc_requests = []
        self.guaranteed_alloc = self._create_guaranteed_alloc_graph()
        self.non_guaranteed_resource_usage = pl.LazyFrame({
            'element': str,
            'app_name': str,
            'assoc_node': str,
            'requests': [],
            'fraction': pl.Float64
        })
        self.preemtable_made_guaranteed_requests = []
        self.fair_share_fraction = 0
        self.residual_preemptable_graph = fm.physical_graph.copy()
        self.rr_update_rate = 10
        self.last_preempt_idx = -1

        super().__init__(fm, requests, run_settings)
        assert float(self.total_demand) == float(self.allocated_demand + self.unallocated_demand)

    def _create_guaranteed_alloc_graph(self):
        graph = self.residual_graph.copy()
        for node in graph.nodes:
            graph.set_node_capacity(node, 0)
        for link in graph.edges:
            graph.set_link_capacity(link, 0)
        return graph

    def _create_non_guaranteed_resource_usage(self):
        from online_handler import EfficientDeque
        graph_elements = list(self.graph.nodes) + list(self.graph.edges)

        element_cache = {element: self._get_element1_element2(element) for element in graph_elements}

        data_to_append = [
            {
                'element1': element_cache[element][0],
                'element2': element_cache[element][1],
                'app_name': app_name,
                'assoc_node': assoc_node,
                'requests': EfficientDeque(),
                'fraction': 0.0  # Directly use float
            }
            for (app_name, assoc_node), v in self.allocation.items()
            for element in graph_elements
        ]

        self.non_guaranteed_resource_usage = pl.DataFrame(data_to_append)

    @profile
    def execute_heuristic(self):
        from comparison_algorithms import OnlineGreedyPathFinder
        self.f1_list = []
        self.f2_list = []
        self.f3_list = []
        self.f4_arv_dep_list = []
        self.fraction_log = []
        self._no_path_identifier_index = set()
        self.stats = self.get_stats_collector(self.run_settings.test_duration)
        self._get_time_agnostic_allocation_dict()
        self.node_app_hash = self._hash_node_app_pairs()
        self.online_path_finder = OnlineGreedyPathFinder(self.apps, self.residual_graph, self.multiplier)
        self._create_non_guaranteed_resource_usage()
        start_time = time.time()
        for t in range(self.test_duration):
            self._execute_heuristic_per_t(t)
        self.run_time = self.run_time + time.time() - start_time
        df = OnlineDataLogger.get_online_df(self.online_logger_list, self.run_settings, export=False)
        self.req_logger.post_process(df)
        self._write_to_files(self._get_scenario_path())

    def _hash_node_app_pairs(self):
        ch = 'a'
        node_app_hash = {}
        for k, v in self.allocation.items():
            node_app_hash[k] = ch  #app_name, assoc_node = k
            ch = chr(ord(ch) + 1)
        return node_app_hash

    @profile
    def _execute_heuristic_per_t(self, t):
        print(f"Online 'Heuristic' - Time {t}")
        self.online_logger = OnlineDataLogger(self.run_settings, self.multiplier)
        start_time = time.time()
        self._handle_departures(t)
        self._handle_arrivals(t)
        runtime = time.time() - start_time
        OnlineDataLogger.log_timeslot(t, self.arrivals, self.online_logger, self.online_logger_list,
                                      self.demand_per_timeslot, runtime)

    @profile
    def _handle_departures(self, t):
        departures = self.departures[t]
        if len(departures) == 0:
            return
        self.non_guaranteed_dealloc_requests = []
        for request in departures:
            if request.id in self.unallocated_requests.keys():
                self.allocation_logger.add_event_entry(t, False, False, request.id, request.app_name,
                                                       request.assoc_node, request.demand, None)
                continue
            self._handle_deallocate_request(t, request)
            self.stats.increment_departures(t, request.assoc_node)
        self._batch_deallocate_non_guaranteed(t)

    @profile
    def _handle_arrivals(self, t):
        arrivals = self.arrivals[t]
        self._no_path_identifier_index = set()
        self.non_guaranteed_arrivals = []
        for request in arrivals:
            self._handle_request_allocation(t, request)
        for request in self.non_guaranteed_arrivals:
            self._handle_non_guaranteed(t, request)

    @profile
    def _handle_request_allocation(self, t, request):
        assoc_node = request.assoc_node
        app_name = request.app_name
        req_allocation = self.allocation[(app_name, assoc_node)]
        self.online_logger.arrived_demand += request.demand
        demand = request.demand
        self.stats.arrival_logger(t, assoc_node, demand)
        self.total_demand += demand
        if req_allocation.get_weight() == 0:
            raise Exception(f"Allocation graph not found for app {app_name} and node {assoc_node}")
        path, min_demand = req_allocation.run_path_finder(request.demand)
        if min_demand >= request.demand:
            is_allocated = self._handle_guaranteed(t, request, path)
            if not is_allocated: #in case self (app,node) using resource, cannot preempt
                self.non_guaranteed_arrivals.append(request)
        else:
            self.non_guaranteed_arrivals.append(request)

    @profile
    def _handle_guaranteed(self, t, request, path):
        is_allocated = False
        req_allocation = self.allocation[(request.app_name, request.assoc_node)]
        path_dict = EmbeddingAlgorithm.collect_path_dict_from_path_finder(path, req_allocation, request.demand)
        utilized_resources = self._check_utilized_physical_resources(path_dict)
        if len(utilized_resources) == 0:
            self._allocate_guaranteed(t, request, path, req_allocation)
            is_allocated = True
        else:
            if self.enable_preempt:
                is_allocated = self._preempt_and_allocate_guaranteed(t, request, path, req_allocation,
                                                                     utilized_resources)
        return is_allocated

    @profile
    def _handle_non_guaranteed(self, t, request):
        is_allocated = False
        path, cost = self.get_greedy_path(request)
        if path is not None:
            self._allocate_entire_demand_online(t, request, path, cost=cost)
            is_allocated = True
        if not is_allocated:
            self._handle_not_allocated(t, request, request.demand)

    def get_greedy_path(self, request):
        path = None
        cost = None
        path, cost = self._get_greedy_path_from_plan(request)
        if path is None:
            path, cost = self.online_path_finder.get_valid_path(request, self.residual_graph, self.run_settings, self.alt_combinations)
        return path, cost

    def _get_greedy_path_from_plan(self, request):
        from comparison_algorithms import OnlineGreedyPathFinder
        req_allocation = self.allocation[(request.app_name, request.assoc_node)]
        for path, cost in req_allocation.collected_paths_costs:
            adjusted_path, adjusted_cost = OnlineGreedyPathFinder.adjust_path_cost_size(path, cost, request.demand)
            if not OnlineGreedyPathFinder.has_capacity_violation(adjusted_path, self.residual_graph):
                return adjusted_path, adjusted_cost
        return None, None

    @profile
    def _allocate_guaranteed(self, t, request, path, req_allocation):
        self._allocate_entire_demand_online(t, request, path, req_allocation=req_allocation)

    def _handle_not_allocated(self, t, request, total_req_demand):
        self.unallocated_requests[request.id] = total_req_demand
        self.unallocated_demand += total_req_demand
        cost = self._get_base_rejection_cost_wrap(request)
        self.req_logger.add_entry(t, request, False, cost)
        self.allocation_logger.add_event_entry(t, True, False, request.id,
                                               request.app_name, request.assoc_node, request.demand, None)

    def _get_base_rejection_cost_wrap(self, request):
        return self.multiplier.get_base_rejection_cost(self.apps[request.app_name].get_app_size()) * request.demand

    @profile
    def _allocate_entire_demand_online(self, t, request, path, req_allocation=None, cost=None):
        self.stats.allocated_logger(t, request.assoc_node, request.demand)
        if req_allocation is not None:
            pre_alloc_cost = self.cost
            self.allocate(request.demand, path, request, req_allocation)
            cost = self.cost - pre_alloc_cost
            self.active_guaranteed_user_paths[request.id] = path
            path = EmbeddingAlgorithm.collect_path_dict_from_path_finder(path, req_allocation, request.demand)
            self.update_physical_graph(path, self.guaranteed_alloc, add_demand=True)
            self._update_residual_preemptable_graph(request, self.guaranteed_alloc, path)
            self.update_physical_graph(path, self.residual_graph)
            greedy = False
        else:
            self.active_non_guaranteed_user_paths[request.id] = path
            self.update_physical_graph(path, self.residual_graph)
            self._update_non_guaranteed_resource_usage(request, path)
            greedy = True
        self.allocation_logger.add_event_entry(t, True, True, request.id, request.app_name,
                                               request.assoc_node, request.demand, path, greedy)
        assert cost > 0
        self.requests_costs[request.id] = cost
        self.online_logger.update_allocation(request.demand)
        # self._update_fraction_log(t, path)
        self.allocated_demand += request.demand

    @profile
    def _preempt_and_allocate_guaranteed(self, t, request, path, req_allocation, congested_resources):
        preempted_path, _ = self._preempt_non_guaranteed_path(t, request, path, cost=0, congested_resources=congested_resources)
        if preempted_path is not None:
            self._allocate_guaranteed(t, request, path, req_allocation)
            return True
        else:
            return False

    @profile
    def _preempt_non_guaranteed_path(self, t, request, path, cost, congested_resources=None):
        preempt_for_non_guaranteed = False
        req_allocation = self.allocation[(request.app_name, request.assoc_node)]
        if congested_resources is None:
            preempt_for_non_guaranteed = True
            congested_resources = self._check_utilized_physical_resources(path)
        while len(congested_resources) > 0:
            resource, blocked_share = congested_resources[0]
            element1, element2 = self._get_element1_element2(resource)
            if (self._get_all_other_active_resource_users(request).
                    lazy().filter((pl.col('element1') == element1) &
                      (pl.col('element2') == element2)).collect().is_empty()):
                return None, None
            candidates = self._get_preemptable_requests(t, request, resource, blocked_share)
            if candidates is None:
                return None, None
            preempt_list = self._get_preempt_list(candidates, blocked_share)
            self._preempt_requests(t, preempt_list)
            if preempt_for_non_guaranteed:
                # from comparison_algorithms import OnlineGreedyPathFinder
                path_post_preempt, cost = self.online_path_finder.find_heuristic_min_path(self.apps[request.app_name], request,
                                                                           self.residual_graph)
                if path_post_preempt is not None:
                    return path_post_preempt, cost
            if isinstance(path, dict):
                path_dict = path
            else:
                path_dict = EmbeddingAlgorithm.collect_path_dict_from_path_finder(path, req_allocation, request.demand)
            congested_resources = self._check_utilized_physical_resources(path_dict)
        return path, cost

    @profile
    def _preempt_requests(self, t, preempt_list):
        for id in preempt_list:
            request = self.id_to_request[id]
            self._handle_deallocate_request(t, request, immediate_dealloc=True)
            demand = request.demand
            self.unallocated_requests[id] = demand
            self.unallocated_demand += demand
            self.allocated_demand -= demand
            self.online_logger.preempted_requests += 1
            self.online_logger.preempted_demand += request.demand

    @profile
    def _get_preempt_list(self, candidates, blocked_share):
        preempt_list = []
        ind = 0
        while blocked_share > 0 and ind < candidates.get_length():
            id, demand = candidates.peek(ind)
            ind += 1
            if id in self.preemtable_made_guaranteed_requests:
                continue
            preempt_list.append(id)
            blocked_share -= demand
        return preempt_list

    @profile
    def _get_max_fraction_on_element(self, request, element):
        element = element if isinstance(element, tuple) else (element, NO_LINK)
        df = self.non_guaranteed_resource_usage
        return df.lazy().filter(
            ~(
                    (pl.col('app_name') == request.app_name) &
                    (pl.col('assoc_node') == request.assoc_node)
            )).filter((pl.col('element1') == element[0]) &
                      (pl.col('element2') == element[1])).select(pl.col("fraction").max()).collect()[0, 'fraction']

    @profile
    def _get_all_other_active_resource_users(self, request):
        df = self.non_guaranteed_resource_usage
        return df.lazy().filter(
            ~(
                    (pl.col('app_name') == request.app_name) &
                    (pl.col('assoc_node') == request.assoc_node)
            ) & (pl.col('fraction') > 0)
        ).collect()

    def _get_preemptable_requests(self, t, request, resource, blocked_share):
        df_others = self._get_all_other_active_resource_users(request)
        total_usage_by_others = self._total_df_usage(self._get_resource_requests(df_others, resource))
        if total_usage_by_others < blocked_share:
            return None
        row = self._get_top_fraction_row(df_others, resource)
        share = self._get_share_on_non_guaranteed_resource(entry=row)
        if share < blocked_share:
            self._preempt_requests(t, row['requests'].keys_to_list())
            blocked_share -= share
            return self._get_preemptable_requests(t, request, resource, blocked_share)
        else:
            return row['requests']

    def _total_df_usage(self, df):
        return sum([self._get_share_on_non_guaranteed_resource(entry=row) for row in df.iter_rows(named=True)])

    def _create_id_to_request(self):
        id_to_request = {}
        for t in self.arrivals.values():
            for request in t:
                id_to_request[request.id] = request
        return id_to_request

    def get_stats_collector(self, test_duration=0):
        return AllocationStatCollectorOnline(self.graph, test_duration)

    @profile
    def _get_top_fraction_row(self, df, resource):
        df = self._get_resource_requests(df, resource)
        max_pos = df.lazy().select(pl.col("fraction").arg_max()).collect().to_numpy()[0][0]
        return df.row(max_pos, named=True)

    @profile
    def _get_resource_requests(self, df, element):
        element1, element2 = self._get_element1_element2(element)
        return df.lazy().filter((pl.col('element1') == element1) & (pl.col('element2') == element2)).collect()

    @staticmethod
    def _get_element1_element2(element):
        if isinstance(element, tuple):
            return element
        else:
            return element, NO_LINK

    @profile
    def _preempt_and_get_path_for_non_guaranteed(self, t, request):
        valid_nodes, invalid_links = self._get_fair_valid_elements(request)
        if len(valid_nodes) == 0 or len(invalid_links) == len(self.graph.edges):
            return None
        path, cost = self.online_path_finder.find_heuristic_min_path(self.apps[request.app_name], request, self.graph,
                                                               valid_nodes, invalid_links)
        if path is None:
            return None, None
        preempted_path, cost = self._preempt_non_guaranteed_path(t, request, path, cost)
        if preempted_path is not None:
            return preempted_path, cost
        else:
            return None, None

    @profile
    def _get_fair_valid_elements(self, request):
        valid_nodes = []
        invalid_links = []
        element_indices = self._get_static_indices_non_guaranteed_resource_usage(request.app_name, request.assoc_node)
        elements_rows = self.non_guaranteed_resource_usage.select(['element1', 'element2']).unique()
        graph_elements = [(row['element1'], row['element2']) if row['element2'] != NO_LINK else row['element1'] for row
                          in elements_rows.rows(named=True)]
        for element in graph_elements:
            valid = False
            is_link = True if isinstance(element, tuple) else False
            if self._has_capacity_for_element(request, element):
                valid = True
            elif self._deserves_fair_share(request, element, element_indices):
                valid = True
            if valid and not is_link:
                valid_nodes.append(element)
            elif not valid and is_link:
                invalid_links.append(element)
        return valid_nodes, invalid_links

    @profile
    def _has_capacity_for_element(self, request, element):
        node_size, link_size = GraphSizeScaler.get_app_sizes_nodes_links(self.apps[request.app_name])
        if isinstance(element, tuple):
            return True if self.residual_graph.get_link_capacity(element) >= link_size * request.demand else False
        else:
            return True if self.residual_graph.get_node_capacity(element) >= node_size * request.demand else False

    @profile
    def _deserves_fair_share(self, request, element, element_indices):
        target_idx = element_indices[element]
        if isinstance(element, tuple):
            link = self.apps[request.app_name].max_link_size_element
            multiplier_key = (request.app_name, link[0], link[1], element[0], element[1])
            multiplier = self.multiplier[multiplier_key]
            share = request.demand * multiplier
        else:
            share = 0
            virtual_nodes = list(self.apps[request.app_name].graph.nodes())
            for node in virtual_nodes:
                multiplier_key = (request.app_name, node, node, element, element)
                multiplier = self.multiplier[multiplier_key]
                share += request.demand * multiplier

        usage = self._get_share_on_non_guaranteed_resource(idx=target_idx) + share
        updated_fraction = self._check_updated_fraction(element, usage=usage)

        max_fraction = self._get_max_fraction_on_element(request, element)
        return True if max_fraction > updated_fraction else False

    @profile
    def _check_utilized_physical_resources(self, path):
        utilized_resources = []
        for (m, n), share in path.items():
            if m != n:
                delta = share - self.residual_graph.get_link_capacity(m, n)
                if delta > 0:
                    utilized_resources.append(((m, n), delta))
            else:
                delta = share - self.residual_graph.get_node_capacity(m)
                if delta > 0:
                    utilized_resources.append((m, delta))
        return utilized_resources

    def _get_time_agnostic_allocation_dict(self):
        allocation = {}
        for k, v in self.allocation.items():
            app_name, alt_name, assoc_node = k
            allocation[(app_name, assoc_node)] = v
        self.allocation = allocation

    @profile
    def _handle_deallocate_request(self, t, request, immediate_dealloc=False):
        if request.id in self.active_guaranteed_user_paths.keys():
            self._deallocate_guaranteed(t, request)
        elif request.id in self.active_non_guaranteed_user_paths.keys():
            if immediate_dealloc:
                self._deallocate_preempted_non_guaranteed(t, request)
            else:
                self.non_guaranteed_dealloc_requests.append(request)
        else:
            raise Exception("Request not found in active paths - need to check why")
        self.stats.increment_deallocated_requests(t, request.assoc_node)
        self.online_logger.update_deallocation(request.demand)

    @profile
    def _batch_deallocate_non_guaranteed(self, t):
        app_src_share_to_dealloc = {}
        app_src_requests = {}
        for request in self.non_guaranteed_dealloc_requests:
            if request.id in self.preemtable_made_guaranteed_requests:
                self.preemtable_made_guaranteed_requests.remove(request.id)
            path = self.active_non_guaranteed_user_paths[request.id]
            a_s_dict = app_src_share_to_dealloc.get((request.app_name, request.assoc_node), {})
            requests_dict = app_src_requests.get((request.app_name, request.assoc_node), [])
            self.aggregate_dicts(a_s_dict, path)
            requests_dict.append(request.id)
            app_src_share_to_dealloc[(request.app_name, request.assoc_node)] = a_s_dict
            app_src_requests[(request.app_name, request.assoc_node)] = requests_dict
            cost = self.requests_costs[request.id]
            self.req_logger.add_entry(t, request, False, cost)
            self.allocation_logger.add_event_entry(t, False, True, request.id, request.app_name,
                                                   request.assoc_node, request.demand, path)
            del self.active_non_guaranteed_user_paths[request.id]
            del self.requests_costs[request.id]
        if len(app_src_share_to_dealloc) > 0:
            total_share_to_dealloc = {}
            for d in app_src_share_to_dealloc.values():
                self.aggregate_dicts(total_share_to_dealloc, d)
            self.update_physical_graph(total_share_to_dealloc, self.residual_graph, add_demand=True)
            self._remove_batch_non_guaranteed_resource_usage(app_src_share_to_dealloc, app_src_requests)

    @staticmethod
    def aggregate_dicts(original_dict, path_dict):
        for element, share in path_dict.items():
            if element in original_dict:
                original_dict[element] += share
            else:
                original_dict[element] = share

    @profile
    def _deallocate_guaranteed(self, t, request):
        req_allocation = self.allocation[(request.app_name, request.assoc_node)]
        path = self.active_guaranteed_user_paths[request.id]
        for path_object in path:
            share = request.demand * ElementTranslator.phys_g_elem_to_alloc_g_mult(req_allocation, path_object.elements)
            capacity = req_allocation.get_link_capacity(path_object.link)
            req_allocation.update_aloc_graph_link_capacity(-share, path_object.link, capacity)
        path = EmbeddingAlgorithm.collect_path_dict_from_path_finder(path, req_allocation, request.demand)
        self.update_physical_graph(path, self.guaranteed_alloc)
        self._update_residual_preemptable_graph(request, self.guaranteed_alloc, path)
        self.update_physical_graph(path, self.residual_graph, add_demand=True)
        cost = self._get_dealloc_cost(t, request)
        self.req_logger.add_entry(t, request, True, cost)
        self.allocation_logger.add_event_entry(t, False, True, request.id, request.app_name,
                                               request.assoc_node, request.demand, path)
        del self.active_guaranteed_user_paths[request.id]
        del self.requests_costs[request.id]

    @profile
    def _deallocate_preempted_non_guaranteed(self, t, request):
        if request.id in self.preemtable_made_guaranteed_requests:
            self.preemtable_made_guaranteed_requests.remove(request.id)
        path = self.active_non_guaranteed_user_paths[request.id]
        self._update_non_guaranteed_resource_usage(request, path, add_request=False)
        self.update_physical_graph(path, self.residual_graph, add_demand=True)
        cost = self._get_dealloc_cost(t, request)
        self.req_logger.add_entry(t, request, False, cost, preempt=True)
        self.allocation_logger.add_event_entry(t, False, True, request.id, request.app_name,
                                               request.assoc_node, request.demand, path, preempted=True)
        del self.active_non_guaranteed_user_paths[request.id]
        del self.requests_costs[request.id]

    def _get_dealloc_cost(self, t, request):
        if t != request.end_time:
            return self._get_base_rejection_cost_wrap(request)
        else:
            return self.requests_costs[request.id]

    def _get_element_share(self, request, element, add_demand=False):
        demand = -request.demand if add_demand else request.demand
        i, j, m, n = element
        multiplier_key = (request.app_name, i, j, m, n)
        multiplier = self.multiplier[multiplier_key]
        return demand * multiplier

    @profile
    def _update_residual_preemptable_graph(self, request, guaranteed_alloc: EnhancedDiGraph, path,
                                           immediate_dealloc=True):
        for (m, n), share in path.items():
            if m == n:
                new_cap = self.graph.get_node_capacity(m) - guaranteed_alloc.get_node_capacity(m)
                self.residual_preemptable_graph.set_node_capacity(m, new_cap)
            else:
                new_cap = self.graph.get_link_capacity((m, n)) - guaranteed_alloc.get_link_capacity((m, n))
                self.residual_preemptable_graph.set_link_capacity((m, n), new_cap)
        if immediate_dealloc:
            self._update_fraction_denominator(request.app_name, request.assoc_node, path)

    @staticmethod
    def update_physical_graph(path, graph: EnhancedDiGraph, add_demand=False):
        for (m, n), share in path.items():
            share = -share if add_demand else share
            if m == n:
                graph.subtract_node_capacity(m, share)
            else:
                graph.subtract_link_capacity(m, n, share)

    @profile
    def _update_non_guaranteed_resource_usage(self, request, path, add_request=True):
        app_name = request.app_name
        assoc_node = request.assoc_node
        element_indices = self._get_static_indices_non_guaranteed_resource_usage(app_name, assoc_node)
        for (m, n), share in path.items():
            element = m if m == n else (m, n)
            share = share if add_request else -share
            target_idx = element_indices[element]
            preemptable_cap = self._get_preemptable_capacity(element)
            request_queue = self.non_guaranteed_resource_usage[target_idx, 'requests']
            request_queue.update(request.id, share)
            fraction = 0 if preemptable_cap == 0 else self._get_share_on_non_guaranteed_resource(
                idx=target_idx) / preemptable_cap
            self.non_guaranteed_resource_usage[target_idx, 'fraction'] = fraction
            if not add_request:
                if request_queue.get(request.id) == 0:
                    request_queue.pop(request.id)

    @profile
    def _remove_batch_non_guaranteed_resource_usage(self, app_src_share_to_dealloc, app_src_requests):
        for (app_name, assoc_node), elements in app_src_share_to_dealloc.items():
            element_indices = self._get_static_indices_non_guaranteed_resource_usage(app_name, assoc_node)
            requests = app_src_requests[(app_name, assoc_node)]
            for (m, n), share in elements.items():
                element = m if m == n else (m, n)
                target_idx = element_indices[element]
                request_queue = self.non_guaranteed_resource_usage[target_idx, 'requests']
                for rid in requests:
                    if request_queue.is_in_deque(rid):
                        request_queue.pop(rid)
                preemptable_cap = self._get_preemptable_capacity(element)
                fraction = 0 if preemptable_cap == 0 else self._get_share_on_non_guaranteed_resource(
                    idx=target_idx) / preemptable_cap
                self.non_guaranteed_resource_usage[target_idx, 'fraction'] = fraction

    @profile
    def _check_updated_fraction(self, element, usage):
        preemptable_cap = self._get_preemptable_capacity(element)
        return 0 if preemptable_cap == 0 else usage / preemptable_cap

    @profile
    def _update_fraction_denominator(self, app_name, assoc_node, path):
        element_indices = self._get_static_indices_non_guaranteed_resource_usage(app_name, assoc_node)
        for (m, n), share in path.items():
            element = m if m == n else (m, n)
            target_idx = element_indices[element]
            if self._is_non_guaranteed_resource_used(target_idx):
                preemptable_cap = self._get_preemptable_capacity(element)
                usage = self._get_share_on_non_guaranteed_resource(idx=target_idx)
                fraction = 0 if preemptable_cap == 0 else usage / preemptable_cap
                self.non_guaranteed_resource_usage[target_idx, 'fraction'] = fraction

    def _is_non_guaranteed_resource_used(self, target_idx):
        if self.non_guaranteed_resource_usage[target_idx, 'requests'].get_num_items() > 0:
            return True

    def _get_share_on_non_guaranteed_resource(self, idx=None, entry=None):
        if idx is not None and entry is not None:
            raise Exception("Only one of the arguments should be provided")
        if idx is not None:
            requests = self.non_guaranteed_resource_usage[idx, 'requests']
        else:
            requests = entry['requests']
        return requests.get_value_sum()

    def _get_preemptable_capacity(self, element):
        """Retrieve the preemptable capacity for an element."""
        cap = self.residual_preemptable_graph.get_link_capacity(element) if isinstance(element, tuple) else \
            self.residual_preemptable_graph.get_node_capacity(element)
        return Decimal(str(cap))

    @lru_cache(maxsize=1000)
    def _get_static_indices_non_guaranteed_resource_usage(self, app_name, assoc_node):
        resource_dict = {}
        df = self.non_guaranteed_resource_usage.with_row_index(name="row_number")
        df = df.filter((pl.col('app_name') == app_name) & (pl.col('assoc_node') == assoc_node))
        for row in df.rows(named=True):
            if row['element2'] == NO_LINK:
                element = row['element1']
            else:
                element = (row['element1'], row['element2'])
            resource_dict[element] = row['row_number']
        return resource_dict

    @profile
    def _write_to_files(self, scenario_path):
        self.allocation_logger.export(scenario_path)
        with file_lock:
            with open(scenario_path / "heuristic_solution_share.txt", "a") as f1, \
                    open(scenario_path / "heuristic_solution_cost.txt", "a") as f2, \
                    open(scenario_path / "heuristic_online_alloc.csv", "a") as f3, \
                    open(scenario_path / "preempt_arrivals_log.csv", "a") as f4:
                if f1.tell() == 0:
                    f1.write("user_id, app_name, src, f1, f2, m, n, share\n")
                if f2.tell() == 0:
                    f2.write("user_id, app_name, src, f1, f2, m, n, cost\n")
                f1.writelines(self.f1_list)
                f2.writelines(self.f2_list)
                if f4.tell() == 0:
                    f4.write(f"step,t,resource,fraction,hash,app,node\n")
                f4.writelines(self.fraction_log)
