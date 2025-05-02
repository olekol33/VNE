from decimal import Decimal
import time

from application import ApplicationSet
from include import config
from include.enums import OnlineExpType
from allocation_heuristic import HeuristicRunner
from fluid_model import OfflineFluidModel
from result_exporter import ResultExporter, AllocationStatExporter
import copy
import os
import pandas as pd
from physical_graph import Topology, GraphSizeScaler
from file_handler import FileHandler
from include.enums import ExperimentType
from dataclasses import dataclass, replace
from user_processor import UsersProcessor
from code_testing import timer


@dataclass
class ExperimentInput:
    topology: Topology
    apps: {}
    run_number: int
    users: UsersProcessor


class RunExperiment:
    def __init__(self):
        if config.experiment == ExperimentType.DONT_RUN:
            return
        elif config.experiment == ExperimentType.HEURISTIC_COMPARISON_STATIC:
            exp = HeuristicComparisonStatic()
            exp.experiment_aggregator()
        elif config.experiment == ExperimentType.MULTI_SET:
            exp = MultiSet()
            exp.experiment_aggregator()
        elif config.experiment == ExperimentType.ONLINE:
            exp = Online()
            exp.experiment_aggregator()
        elif config.experiment == ExperimentType.FIC_ALTERNATIVES:
            exp = FicAlt()
            exp.experiment_aggregator()
        else:
            raise NotImplementedError("Experiment not found")


@dataclass
class ExperimentRunSettings:
    nrf: Decimal
    erf: Decimal
    branch_id: int
    branch_name: str
    exp_name: str
    run_number: int
    total_demand: int
    target_utilization: float = -1
    hike: float = -1
    intensity: float = -1
    special_input: float = -1
    test_duration: float = 0

    def __hash__(self):
        return hash((self.nrf, self.erf, self.branch_id, self.exp_name, self.run_number, self.total_demand))

    def update_greedy(self, function_wise=False):
        func_wise = 'FW-' if function_wise else ''
        branch_name = f'Greedy-{func_wise}' + self.branch_name.split("OPT-")[1]
        split_exp_name = self.exp_name.split("_")
        split_exp_name[0] = "Greedy"
        exp_name = "_".join(split_exp_name)
        return replace(self, exp_name=exp_name, branch_name=branch_name)

    def update_fullg(self):
        branch_name = 'FullG-' + self.branch_name.split("OPT-")[1]
        split_exp_name = self.exp_name.split("_")
        split_exp_name[0] = "FullG"
        exp_name = "_".join(split_exp_name)
        return replace(self, exp_name=exp_name, branch_name=branch_name)

    def update_oracle(self):
        branch_name = 'Oracle-LP'
        split_exp_name = self.exp_name.split("_")
        split_exp_name[0] = "Oracle-LP"
        exp_name = "_".join(split_exp_name)
        return replace(self, exp_name=exp_name, branch_name=branch_name)

    def update_online_opt(self):
        branch_name = 'Online-OPT'
        split_exp_name = self.exp_name.split("_")
        split_exp_name[0] = "Online-OPT"
        exp_name = "_".join(split_exp_name)
        return replace(self, exp_name=exp_name, branch_name=branch_name)

    def update_name(self, name):
        branch_name = name
        return replace(self, branch_name=branch_name)

    def update_target_utilization(self, target_utilization):
        print(f"WARNING: Diff train util: {target_utilization}")
        return replace(self, target_utilization=target_utilization)

    def update_hike(self, hike):
        return replace(self, hike=hike)

    def update_special_input(self, val):
        return replace(self, special_input=val)

    def update_intensity(self, intensity):
        return replace(self, intensity=intensity)

    def update_exp_name(self, exp_name):
        return replace(self, exp_name=exp_name)

    def update_test_duration(self, test_duration):
        return replace(self, test_duration=test_duration)


class Experiment:
    def __init__(self, apps: ApplicationSet, topology: Topology):
        self.nrf = [config.nrf]
        self.erf = [config.erf]
        self.run_number = 0
        self.scale_ind = 0
        self.FULL_APP = -1
        self.debug_specific_branch = None
        self.debug_specific_run = None
        self.branch_id_to_node = {}
        self.app_branches = {}
        self.alt_combinations = {}
        self.allocation_stats_per_set = {}
        self.branch_name_to_node_link_capacity = {}
        self.apps = apps.get_apps()
        self.topology = topology

        self.exp_name = self.__class__.__name__

    def run_experiment(self, users):
        self._check_assumptions()
        self._extract_required_data()
        self._set_parameters(users)
        self._create_data_for_experiment()
        self._run_experiment()
        self._experiment_post_processing()

    def _check_assumptions(self):
        raise NotImplementedError("Experiment not found")

    def _extract_required_data(self):
        raise NotImplementedError("Experiment not found")

    def _set_parameters(self, users: UsersProcessor):
        self.users = users

    def _create_data_for_experiment(self):
        raise NotImplementedError("Experiment not found")

    def _run_experiment(self):
        self._run_model()
        self._run_oracle()
        self._run_comparison_algorithms()
        self._run_heuristic()

    def _experiment_post_processing(self):
        raise NotImplementedError("Experiment not found")

    def _run_oracle(self):
        pass

    def experiment_aggregator(self):
        self._get_user_allocation_results()

    def _get_user_allocation_results(self):
        data = []
        res_dir = FileHandler.get_run_dir('summary')
        header = None
        for filename in os.listdir(res_dir):
            exp_name = self._extract_experiment_name(filename)
            if exp_name is None:
                continue
            df = pd.read_csv(res_dir / filename, index_col=None)
            for ind, row in df.iterrows():
                row['Exp Type'] = exp_name
                data.append(row)
            header = df.columns.to_list()
        header.insert(0, 'Exp Type')
        self.results_df = pd.DataFrame(data, columns=header)
        AllocationStatExporter.export_to_file(self.results_df, 'results_df.csv')

    def normalize_allocated_demand(self, allocated_demand, branch_name):
        partial_app_total_node_capacity, partial_app_total_link_capacity = self.branch_name_to_node_link_capacity[
            branch_name]
        partial_app_capacity = partial_app_total_node_capacity + partial_app_total_link_capacity
        full_app_total_capacity = self.full_app_total_node_capacity + self.full_app_total_link_capacity
        return allocated_demand * full_app_total_capacity / partial_app_capacity

    def _extract_experiment_name(self, filename):
        if (filename.startswith('Greedy') or filename.startswith('OPT') or filename.startswith('TANTO') or
                filename.startswith('Oracle') or filename.startswith('FullG')):
            return filename.split('.csv')[0]

    def _create_data_for_experiment_branches(self):
        self.exp_fm = {}
        self.allocation_stats = {}
        self.exp_res = {}
        self._create_branches_dict()
        self.alt_combinations = self.get_alternative_combinations()
        for branch_id, branch_nodes in self.alt_combinations.items():
            self._extract_branch(branch_id)
            branch_name = self.get_branch_name(branch_id, branch_nodes)
            num_users_k = len(self.full_user_list) / 1000
            num_users_short = int(num_users_k) if isinstance(num_users_k, int) else round(num_users_k, 1)
            for n in self.nrf:
                for e in self.erf:
                    exp_name = f"{self.exp_name}_nrf{n}_erf{e}_{num_users_short}K_{branch_id}_{branch_name}"
                    run_setting = ExperimentRunSettings(nrf=n, erf=e, branch_id=branch_id,
                                                        branch_name=branch_name, exp_name=exp_name,
                                                        run_number=self.run_number, total_demand=self.total_demand)
                    self.exp_fm[run_setting] = None
                    if not config.skip_individual_alternatives:
                        FileHandler.create_exp_rundir(self.run_number, exp_name)

    def get_branch_name(self, branch_id, branch_nodes) -> str:
        if branch_id == self.FULL_APP:
            return 'OPT-Full'
        nodes = []
        for app, node in branch_nodes.items():
            nodes.append(f"{list(node.keys())[0]}")
        return f"OPT-{'-'.join(nodes)}"

    def _run_model(self):
        for exp_settings in self.exp_fm.keys():
            start_time = time.time()

            app_name = exp_settings.app_name
            branch_id = exp_settings.branch_id
            exp_name = exp_settings.exp_name
            branch_name = exp_settings.branch_name
            app = self.app_branches[app_name][branch_id]

            print(f"\nRunning experiment {exp_name}")

            apps = {app.graph.name: app}

            target_apps = apps
            physical_graph = GraphSizeScaler(self.topology.graph, target_apps, self.requests, exp_settings.nrf,
                                             exp_settings.erf).graph

            fm = OfflineFluidModel(physical_graph, self.requests, apps, self.run_number, exp_settings.nrf,
                                   exp_settings.erf,
                                   exp_name)
            self.exp_fm[exp_settings] = fm
            ResultExporter(self.run_number, len(self.full_user_list), fm, self.physical_graph, self.apps)

            if branch_id == self.FULL_APP:
                self.full_app_total_node_capacity = self.physical_graph.get_total_node_capacity()
                self.full_app_total_link_capacity = self.physical_graph.get_total_link_capacity()
            else:
                total_node_capacity = self.physical_graph.get_total_node_capacity()
                total_link_capacity = self.physical_graph.get_total_link_capacity()
                self.branch_name_to_node_link_capacity[branch_name] = (total_node_capacity, total_link_capacity)

            elapsed_time = time.time() - start_time
            print(f"Experiment {exp_name} took {elapsed_time} seconds")

    @timer("_run_heuristic")
    def _run_heuristic(self):
        for exp_settings in self.exp_fm.keys():
            if exp_settings.branch_id != self.FULL_APP:
                continue
            new_exp_settings = exp_settings.update_name('TANTO')
            fm = self.exp_fm[exp_settings]
            full_user_list = self.full_user_list
            heuristic = HeuristicRunner(fm.summary, full_user_list, run_settings=new_exp_settings)
            self.allocation_stats[new_exp_settings] = heuristic.stats

    def _run_comparison_algorithms(self):
        if config.skip_greedy:
            return
        for exp_settings in self.exp_fm.keys():
            if 'Oracle' in exp_settings.exp_name:
                continue
            fm = self.exp_fm[exp_settings]
            new_exp_settings = exp_settings.update_greedy()
            if exp_settings.branch_id != self.FULL_APP and config.skip_individual_alternatives:
                continue
            from comparison_algorithms import GreedyMinLoc
            app_name = list(self.apps.keys())[0]
            if exp_settings.branch_id == self.FULL_APP:
                # app_branches = copy.deepcopy(self.app_branches[app_name])
                # del app_branches[self.FULL_APP]
                app_branches = self._get_app_branches(app_name)
                greedy = GreedyMinLoc(new_exp_settings, fm, self.full_user_list, app_branches)
            else:
                app_branches = {
                    exp_settings.branch_id: self.app_branches[app_name][exp_settings.branch_id]}
                greedy = GreedyMinLoc(new_exp_settings, fm, self.full_user_list, app_branches)
            self.allocation_stats[new_exp_settings] = greedy.stats
            app_total_node_capacity = fm.physical_graph.get_total_node_capacity()
            app_total_link_capacity = fm.physical_graph.get_total_link_capacity()
            self.branch_name_to_node_link_capacity[new_exp_settings.branch_name] = (app_total_node_capacity,
                                                                                    app_total_link_capacity)

    def _get_app_branches(self, app_name):
        app_branches = copy.deepcopy(self.app_branches[app_name])
        del app_branches[self.FULL_APP]
        if len(app_branches) == 0:
            ind = 1
            for alt_name, graph in self.app_branches[app_name][self.FULL_APP].get_alternatives().items():
                app = copy.deepcopy(self.app_branches[app_name][self.FULL_APP])
                app.graph = graph
                app_branches[ind] = app
                ind += 1
        return app_branches

    def _create_branches_dict(self) -> {}:
        branch_id = 1
        branch_nodes = list(self.app.alternatives.keys())
        for branch_node in branch_nodes:
            self.branch_id_to_node[branch_id] = branch_node
            branch_id += 1
        self.branch_id_to_node[self.FULL_APP] = config.full_app

    def _create_scaled_physical_graph(self, requests, nrf, erf, apps):
        graph = GraphSizeScaler(self.topology.graph, apps, requests, nrf, erf).graph
        UsersProcessor.check_request_load_on_allocated_node(requests, graph, self.apps)
        return graph

    def _check_one_choice_node_in_app(self):
        for app in self.apps.values():
            choice_nodes = list(app.get_choice_nodes().keys())
            if 'U' in choice_nodes:
                choice_nodes.remove('U')
            if len(choice_nodes) > 1:
                raise ValueError("More than one choice node")

    def get_alternative_combinations(self):
        alternatives = self.get_all_alternatives()
        combinations = {self.FULL_APP: alternatives}
        if config.skip_individual_alternatives:
            return combinations
        if len(alternatives) > 1:
            raise ValueError("More than one app is used in experiment of single branch")

        for app_name, branches in alternatives.items():
            branch_id = 1
            for branch in branches:
                combinations[branch_id] = ({app_name: {branch: branches[branch]}})
                branch_id += 1
        return combinations

    def get_all_alternatives(self) -> {}:
        alternatives = {}
        for app_name, app in self.apps.items():
            alternatives[app_name] = app.get_alternatives()
        return alternatives

    def _check_size_defined_in_app(self):
        for app in self.apps.values():
            for node in app.graph.nodes:
                if node == config.user_func:
                    continue
                if 'size' not in app.graph.nodes[node]:
                    raise ValueError(f"No size definition in app for node {node}")
            for edge in app.graph.edges:
                if config.user_func in edge:
                    continue
                if 'size' not in app.graph.edges[edge]:
                    raise ValueError(f"No size definition in app for edge {edge}")

    def _extract_branch(self, branch_id):
        for app_name, app in self.apps.items():
            app = copy.deepcopy(app)
            branch_nodes = self.branch_id_to_node[branch_id]
            app.branch_name = branch_nodes
            if branch_id != self.FULL_APP:
                branch_nodes = branch_nodes.split('-')
                nodes = copy.deepcopy(app.graph.nodes)
                removed_nodes = []
                for node_name in nodes:
                    if node_name not in removed_nodes:
                        node = app.graph.nodes[node_name]
                        if 'branch' in node and node_name not in branch_nodes:
                            removed = app.remove_all_node_successors(node_name)
                            removed_nodes += removed
                app.graph = app.get_alternatives()[branch_nodes[0]]
            app_branches = self.app_branches.get(app_name, {})
            app_branches[branch_id] = app
            self.app_branches[app_name] = app_branches

    @staticmethod
    def _get_sorted_experiment_list(df, secondary_metric):
        return df[['Experiment', secondary_metric]].drop_duplicates().sort_values(by=['Experiment', secondary_metric])[
            'Experiment']


class HeuristicComparisonStatic(Experiment):
    def __init__(self):
        apps = ApplicationSet()
        for run_number in range(0, config.number_of_runs):
            topology = Topology(run_number)
            FileHandler.create_run_dir(run_number)
            users = UsersProcessor(topology, apps, config.num_of_offline_requests, run_number)
            self.run_number = run_number
            self.app = apps.get_app_names()[0]

            self.branch_name_to_node_link_capacity = {}
            self.full_app_total_node_capacity = 0
            self.full_app_total_link_capacity = 0
            self.exp_fm = {}
            self.collocated_demand = {}
            self.tiers_share = {}
            self.total_cost = {}
            self.allocated_by_type = {}
            self.num_of_runs = 1
            super().__init__(apps, topology, users)
            self.run_experiment()

    def _check_assumptions(self):
        self._check_size_defined_in_app()
        self._check_one_choice_node_in_app()

    def _create_data_for_experiment(self):
        self._create_data_for_experiment_branches()

    @timer("_run_experiment")
    def _run_experiment(self):
        self._run_model()
        self._run_comparison_algorithms()
        self._run_heuristic()

    def _experiment_post_processing(self):
        pass

    def _parse_utilization_file(self, key: ()):
        nrf, erf, branches = key
        exp_name = f"AlternativeComparisonWithMultipliers_{nrf}_{erf}_{branches}"
        filepath = FileHandler.get_run_dir(self.run_number, exp_name) / 'share_utilization.csv'
        df = pd.read_csv(filepath)
        total_share = df['norm share'].sum()
        node_share = df.loc[df['n'] == '-1', 'norm share'].sum() / total_share
        link_share = df.loc[df['n'] != '-1', 'norm share'].sum() / total_share
        self.allocated_by_type[key] = [node_share, link_share]

    def _check_one_app_is_used(self):
        if len(self.apps) > 1:
            raise ValueError("More than one app is used in experiment")


class Online(Experiment):
    def __init__(self):
        from online_handler import OnlineHandler
        self.hike = -1
        self.special_input = None
        self.intensity = -1
        topology, apps = self._run_topology_apps()
        self.requests_per_second = OnlineHandler.get_lambda(topology.graph, lam=config.lambda_per_node)
        print(f"Requests per second: {self.requests_per_second}")
        super().__init__(apps, topology)
        for set_id in range(config.num_of_experiments):
            self.set_id = set_id
            if config.online_exp_type == OnlineExpType.DIFFERENT_HOTSPOTS:
                topology.set_node_popularity()
                self._online_exp_runner(topology, apps, set_id)
            elif config.online_exp_type == OnlineExpType.NUMBER_OF_APPS:
                self._run_number_of_apps_exp(topology, set_id)
            elif config.online_exp_type == OnlineExpType.LENGTH_OF_APPS:
                self._run_length_of_apps_exp(topology, set_id)
            elif config.online_exp_type == OnlineExpType.LINK_CAPACITY:
                self._run_link_capacities_exp(topology, set_id)
            elif (config.online_exp_type == OnlineExpType.MMPP_PARAMS or
                  config.online_exp_type == OnlineExpType.REQUEST_DURATION):
                self._run_trace_input_exp(apps, topology, set_id)
            elif (config.online_exp_type == OnlineExpType.PERCENTILE or
                    config.online_exp_type == OnlineExpType.TRAIN_PERIOD_SHARE or
                  config.online_exp_type == OnlineExpType.LAMBDA):
                self._run_trace_input_with_diff_apps_exp(topology, set_id)
            elif config.online_exp_type == OnlineExpType.DIFFERENT_APPS:
                self._run_update_apps_exp(topology, set_id)
            elif config.online_exp_type == OnlineExpType.CLASSES:
                self._run_number_of_fic_layers_exp(topology, set_id)
            elif config.online_exp_type == OnlineExpType.SAME_HOTSPOTS:
                self._online_exp_runner(topology, apps, set_id)
            else:
                raise ValueError("Online experiment type not found")
                
                
    def _run_update_apps_exp(self, topology, set_id):
        apps = ApplicationSet(set_id=set_id)
        self.apps = apps.get_apps()
        self._online_exp_runner(topology, apps, set_id)

    def _run_trace_input_exp(self, apps, topology, set_id):
        if not config.export_requests:
            raise ValueError("Requests must be exported for this experiment")
        for param in config.special_exp_params:
            self.special_input = param
            self._online_exp_runner(topology, apps, set_id)

    def _run_trace_input_with_diff_apps_exp(self, topology, set_id):
        if not config.export_requests:
            raise ValueError("Requests must be exported for this experiment")
        apps = ApplicationSet(set_id=set_id)
        self.apps = apps.get_apps()
        for ratio in config.special_exp_params:
            self.special_input = ratio
            self._online_exp_runner(topology, apps, set_id)

    def _run_link_capacities_exp(self, topology, set_id):
        for base_capacity in config.special_exp_params:
            topology_cp = copy.deepcopy(topology)
            topology_cp.graph.update_link_capacities(base_capacity)
            self.special_input = base_capacity
            apps = ApplicationSet(set_id=set_id)
            self.apps = apps.get_apps()
            self._online_exp_runner(topology_cp, apps, set_id)

    def _run_number_of_apps_exp(self, topology, set_id):
        for num_apps in config.special_exp_params:
            apps = ApplicationSet(num_apps=num_apps, set_id=set_id)
            self.apps = apps.get_apps()
            self.special_input = num_apps
            self._online_exp_runner(topology, apps, set_id)

    def _run_number_of_fic_layers_exp(self, topology, set_id):
        for layers in config.special_exp_params:
            print(f"\n Running set {set_id} with layers {layers}\n")
            apps = ApplicationSet(fic_layers=layers)
            self.apps = apps.get_apps()
            self.special_input = layers
            self._online_exp_runner(topology, apps, set_id)

    def _run_length_of_apps_exp(self, topology, set_id):
        for length_apps in config.special_exp_params:
            apps = ApplicationSet(length_apps=length_apps, set_id=set_id)
            self.apps = apps.get_apps()
            self.special_input = length_apps
            self._online_exp_runner(topology, apps, set_id)

    def _run_topology_apps(self):
        if config.online_exp_type != OnlineExpType.NUMBER_OF_APPS and config.online_exp_type != OnlineExpType.LENGTH_OF_APPS:
            apps = ApplicationSet()
        else:
            apps = ApplicationSet(num_apps=config.special_exp_params[0], length_apps=config.special_exp_params[0]) #placeholder
        topology = Topology()
        return topology, apps

    @timer("_online_exp_runner")
    def _online_exp_runner(self, topology, apps, set_id):
        from user_processor import OnlineUserProcessor
        self.users = OnlineUserProcessor(topology, apps, set_id, self.special_input)
        if config.skip_to_exp_num is not None and set_id < config.skip_to_exp_num:
            print(f"WARNING: Skipping set {set_id}")
            return
        self.apps = apps.get_apps()
        self.physical_graph = topology.graph.copy()
        target_utilizations = config.target_utilization
        for target_util in target_utilizations:
            print(f"\n Running set {set_id} with factor {target_util}\n")
            self.target_util = target_util
            self.users.create_user_list(target_util)
            self._set_variables(set_id)
            self.run_experiment(self.users)
            self.allocation_stats_per_set[0] = copy.deepcopy(self.allocation_stats)
            AllocationStatExporter(self.allocation_stats_per_set, self.users.num_train_requests)

    @timer("create_test_utilization_file")
    def create_test_utilization_file(self, exp_settings, fm_summary):
        if 1 in self.users.test_requests:
            from online_handler import TestUtilizationAnalyzer
            test_req_util_1 = self.users.test_requests[1]
            TestUtilizationAnalyzer(fm_summary, self.apps, test_req_util_1, exp_settings.test_duration, self.run_number, exp_settings.exp_name)

    def _set_variables(self, set_id):
        self.run_number = set_id
        FileHandler.create_run_dir(self.run_number)
        self.branch_name_to_node_link_capacity = {}
        self.full_app_total_node_capacity = 0
        self.full_app_total_link_capacity = 0
        train_set_ratio = self.special_input if config.online_exp_type == OnlineExpType.TRAIN_PERIOD_SHARE else config.train_set_ratio
        self.train_duration = int(train_set_ratio * config.sim_time)
        self.exp_fm = {}
        self.collocated_demand = {}
        self.tiers_share = {}
        self.total_cost = {}
        self.allocated_by_type = {}
        self._get_total_demand()

    @timer("_run_experiment")
    def _run_experiment(self):
        self._run_model()
        for exp_settings in self.exp_fm.keys():
            if exp_settings.branch_id != self.FULL_APP:
                continue
        exp_settings = list(self.exp_fm.keys())[0]
        fm_summary = self.exp_fm[exp_settings].summary
        test_requests = self.users.test_requests[self.target_util]
        self.create_test_utilization_file(exp_settings, fm_summary)
        alt_combinations = self._remove_fic_alternatives()
        if config.multiprocess_algorithms:
            import multiprocessing
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = {}
                futures['comparison_algorithms'] = executor.submit(self._run_comparison_algorithms, exp_settings,
                                                                   fm_summary, test_requests, False)
                futures['heuristic'] = executor.submit(self._run_heuristic, exp_settings, fm_summary, test_requests, alt_combinations)
                futures['single'] = executor.submit(self._run_fullg, exp_settings, fm_summary, test_requests, alt_combinations)
                futures['oracle'] = executor.submit(self._run_oracle, exp_settings, fm_summary, test_requests,
                                                    alt_combinations)

                for future_identifier, future in futures.items():
                    try:
                        if future.result() is not None:
                            stats, new_exp_settings = future.result()
                            self.allocation_stats[new_exp_settings] = stats
                        else:
                            future.result()
                    except Exception as exc:
                        raise exc
        else:
            self._run_oracle(exp_settings, fm_summary, test_requests, alt_combinations)
            self._run_heuristic(exp_settings, fm_summary, test_requests, alt_combinations)
            self._run_comparison_algorithms(exp_settings, fm_summary, test_requests, False)
            self._run_fullg(exp_settings, fm_summary, test_requests, alt_combinations)

    def _remove_fic_alternatives(self):
        alt_combinations = {}
        for branch_id, apps in self.alt_combinations.items():
            branches = {}
            for app_name, alternatives in apps.items():
                apps = {}
                for alt_name, app_graph in alternatives.items():
                    if 'fic' in alt_name:
                        continue
                    apps[alt_name] = app_graph
                branches[app_name] = apps
            alt_combinations[branch_id] = branches
        return alt_combinations

    def _get_total_demand(self):
        self.total_demand = 0
        for req in self.users.grouped_user_list:
            self.total_demand += req.demand

    def _check_assumptions(self):
        from online_handler import OnlineHandler
        self._check_size_defined_in_app()
        OnlineHandler.check_fixed_capacities()

    def _extract_required_data(self):
        pass

    def _create_data_for_experiment(self):
        self.exp_fm = {}
        self.allocation_stats = {}
        self.exp_res = {}
        self.alt_combinations = self.get_alternative_combinations()
        for branch_id, branch_nodes in self.alt_combinations.items():
            branch_name = self.get_branch_name(branch_id, branch_nodes)
            for n in self.nrf:
                for e in self.erf:
                    special_input = '' if self.special_input is None else f'_{self.special_input}'
                    exp_name = f"{self.exp_name}_{self.target_util}_{self.set_id}{special_input}"
                    run_setting = ExperimentRunSettings(nrf=n, erf=e, branch_id=branch_id,
                                                        branch_name=branch_name, exp_name=exp_name,
                                                        run_number=self.set_id, total_demand=self.total_demand,
                                                        target_utilization=self.target_util, special_input=self.special_input,
                                                        test_duration=self.users.test_duration)
                    self.exp_fm[run_setting] = None
                    if not config.skip_individual_alternatives:
                        FileHandler.create_exp_rundir(self.run_number, exp_name)

    def _run_model(self):
        for exp_settings in self.exp_fm.keys():
            branch_id = exp_settings.branch_id
            if branch_id != self.FULL_APP and config.skip_individual_alternatives:
                continue
            exp_name = exp_settings.exp_name
            branch_name = exp_settings.branch_name
            alternatives = self.alt_combinations[branch_id]
            fic_layers = self.special_input if config.online_exp_type == OnlineExpType.CLASSES else config.fic_layers
            print(f"\nRunning experiment {exp_name}")
            fm = OfflineFluidModel(self.physical_graph, self.users.grouped_user_list, self.apps, alternatives,
                                   exp_settings, fic_layers=fic_layers)
            self.exp_fm[exp_settings] = fm
            mean_req_num = self.requests_per_second * self.train_duration
            exporter = ResultExporter(self.run_number, mean_req_num, fm, self.physical_graph, self.apps, branch_name)
            self.allocation_stats[exp_settings] = exporter.stats

            if branch_id == self.FULL_APP:
                self.full_app_total_node_capacity = self.physical_graph.get_total_node_capacity()
                self.full_app_total_link_capacity = self.physical_graph.get_total_link_capacity()
            else:
                total_node_capacity = self.physical_graph.get_total_node_capacity()
                total_link_capacity = self.physical_graph.get_total_link_capacity()
                self.branch_name_to_node_link_capacity[branch_name] = (total_node_capacity, total_link_capacity)

    @staticmethod
    @timer("_run_heuristic")
    def _run_heuristic(exp_settings, fm_summary, test_requests, alt_combinations=None):
        if config.skip_heuristic:
            return
        new_exp_settings = exp_settings.update_name('TANTO')
        branch_id = exp_settings.branch_id
        alternatives = alt_combinations[branch_id]
        heuristic = HeuristicRunner(fm_summary, test_requests, alt_combinations=alternatives, run_settings=new_exp_settings)
        return heuristic.stats, new_exp_settings

    @staticmethod
    @timer("_run_oracle")
    def _run_oracle(exp_settings, fm_summary, test_requests, alt_combinations):
        if config.skip_slot_off:
            return None
        from oracle_runner import OracleRunner
        new_exp_settings = exp_settings.update_oracle()
        branch_id = exp_settings.branch_id
        alternatives = alt_combinations[branch_id]
        OracleRunner(new_exp_settings, fm_summary, test_requests, alternatives)

    @staticmethod
    @timer("_run_comparison_algorithms")
    def _run_comparison_algorithms(exp_settings, fm_summary, test_users, function_wise=False):
        if config.skip_greedy:
            return None
        from comparison_algorithms import GreedyOnline
        new_exp_settings = exp_settings.update_greedy(function_wise)
        greedy = GreedyOnline(new_exp_settings, fm_summary, test_users, function_wise)
        return greedy.stats, new_exp_settings

    @staticmethod
    def _run_fullg(exp_settings, fm_summary, test_users, alt_combinations):
        if config.skip_fullg:
            return
        from comparison_algorithms import FullG
        new_exp_settings = exp_settings.update_fullg()
        branch_id = exp_settings.branch_id
        alternatives = alt_combinations[branch_id]
        greedy = FullG(new_exp_settings, fm_summary, test_users, alternatives)
        return greedy.stats, new_exp_settings

    def _experiment_post_processing(self):
        pass


class FicAlt(Online):
    def __init__(self):
        for fic_layers in config.fic_layers_num:
            self.fic_layers = fic_layers
            super().__init__()

    def _run_topology_apps(self):
        apps = ApplicationSet(fic_layers=self.fic_layers)
        topology = Topology()
        return topology, apps

    def _run_experiment(self):
        self._run_model()

    def _run_model(self):
        mean_req_num = self.requests_per_second * self.train_duration
        for exp_settings in self.exp_fm.keys():
            branch_id = exp_settings.branch_id
            if branch_id != self.FULL_APP and config.skip_individual_alternatives:
                continue
            exp_name = f"{exp_settings.exp_name}_LAY_{self.fic_layers}"
            branch_name = exp_settings.branch_name
            alternatives = self.alt_combinations[branch_id]
            print(f"\nRunning experiment {exp_name}")
            fm = OfflineFluidModel(self.physical_graph, self.users.grouped_user_list, self.apps, alternatives,
                                   self.run_number, exp_settings.nrf, exp_settings.erf, exp_name, fic_layers=self.fic_layers)
            self.exp_fm[exp_settings] = fm
            exporter = ResultExporter(self.run_number, mean_req_num, fm, self.physical_graph, self.apps, branch_name)
            self.allocation_stats[exp_settings] = exporter.stats


class MultiSet(Experiment):
    def __init__(self):
        apps = ApplicationSet()

        topology = Topology()
        sets_plus_train_set = config.num_of_experiments + 1
        num_requests_scaling_factor = [1]  #scale number of requests, used only in offline runtime experiment
        list_of_user_nums = [int(n * config.num_of_offline_requests) for n in num_requests_scaling_factor]
        for user_num in list_of_user_nums:
            num_of_users = user_num * sets_plus_train_set
            users = UsersProcessor(topology, apps, num_of_users)

            super().__init__(apps, topology)
            self.app = next(iter(apps.get_apps().values()))
            scale_demand_factor = 1
            _, requests_for_scaling = users.generate_partial_request_list(scale_demand_factor,
                                                                          set_id=config.num_of_experiments)

            self.physical_graph = self._create_scaled_physical_graph(
                requests=requests_for_scaling, nrf=config.nrf, erf=config.erf, apps=apps.get_apps_dict())
            for user_factor in config.target_utilization:
                self.exp_fm_opt = {}
                for set_id in range(0, config.num_of_experiments):
                    print(f"\n Running set {set_id} with factor {user_factor}, users: {user_num}, "
                          f"nodes: {self.physical_graph.get_num_of_nodes()}, "
                          f"links: {self.physical_graph.get_num_of_links()}\n")
                    FileHandler.create_run_dir(self.run_number)
                    self.full_user_list, self.requests = users.generate_partial_request_list(user_factor, set_id)
                    self._get_total_demand()
                    self.branch_name_to_node_link_capacity = {}
                    self.full_app_total_node_capacity = 0
                    self.full_app_total_link_capacity = 0
                    self.exp_fm = {}
                    self.collocated_demand = {}
                    self.tiers_share = {}
                    self.total_cost = {}
                    self.allocated_by_type = {}
                    self.run_experiment(users)
                    self.allocation_stats_per_set[set_id] = copy.deepcopy(self.allocation_stats)
                    self.run_number += 1
                AllocationStatExporter(self.allocation_stats_per_set, len(self.full_user_list))

    def _get_total_demand(self):
        self.total_demand = 0
        for req in self.requests:
            self.total_demand += req.demand

    def _check_assumptions(self):
        self._check_size_defined_in_app()

    def _extract_required_data(self):
        pass

    def _create_data_for_experiment(self):
        self._create_data_for_experiment_branches()

    def _run_model(self):
        for exp_settings in self.exp_fm.keys():
            branch_id = exp_settings.branch_id
            if branch_id != self.FULL_APP and config.skip_individual_alternatives:
                continue
            exp_name = exp_settings.exp_name
            branch_name = exp_settings.branch_name
            if len(self.apps) > 1:
                raise ValueError("More than one app is used in experiment")
            alternatives = self.alt_combinations[branch_id]

            print(f"\nRunning experiment {exp_name}")
            fm = OfflineFluidModel(self.physical_graph, self.requests, self.apps, alternatives, exp_settings)
            self.exp_fm[exp_settings] = fm
            exporter = ResultExporter(self.run_number, len(self.full_user_list), fm, self.physical_graph, self.apps,
                                      branch_name)
            self.allocation_stats[exp_settings] = exporter.stats

            if branch_id == self.FULL_APP:
                self.full_app_total_node_capacity = self.physical_graph.get_total_node_capacity()
                self.full_app_total_link_capacity = self.physical_graph.get_total_link_capacity()
            else:
                total_node_capacity = self.physical_graph.get_total_node_capacity()
                total_link_capacity = self.physical_graph.get_total_link_capacity()
                self.branch_name_to_node_link_capacity[branch_name] = (total_node_capacity, total_link_capacity)

    def _experiment_post_processing(self):
        pass
