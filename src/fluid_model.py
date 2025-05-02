import os
from line_profiler_pycharm import profile
from docplex.mp.dvar import Var
from include.enums import ExperimentType
from decimal import Decimal
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from physical_graph import EnhancedDiGraph
from docplex.mp.model import Model
from include import config
from users import User
from code_testing import timer
from req_timing_gen import RequestGenFileHandler
from file_handler import FileHandler
from multipliers import DefaultMultiplierDict


class FluidModel:
    def __init__(self, physical_graph: EnhancedDiGraph, requests: [], apps: {}, alternatives: {}, exp_setting,
                 multiplier=None, dirpath=None, oracle=False, fic_layers=config.fic_layers, online_opt=False):
        self.var = {}
        self.request_demand = {}
        self.allocation_graphs = {}
        self.requests = requests
        self.run_number = exp_setting.run_number
        self.solver = None
        self.online_opt = online_opt
        self.seed = config.seed + self.run_number
        self.apps = apps
        self.alternatives = alternatives
        self.exp_name = exp_setting.exp_name
        self.dirpath = FileHandler.get_run_dir(self.run_number, self.exp_name) if dirpath is None else dirpath
        self.dump_non_zero = FileHandler.get_run_dir().parent / f'non_zero_vars_{self.exp_name}.pkl'
        self.fic_layers = 0 if len(apps) == 1 else fic_layers
        self.constraints = []
        self.node_combinations = []
        self.summary = None
        self.cost = 0
        self.node_utilization = 0
        self.link_utilization = 0
        self.non_zero_weights = []
        self.non_zero_vars = {}
        self.nrf = exp_setting.nrf
        self.erf = exp_setting.erf
        self.oracle = oracle
        self.multiplier = multiplier
        self.physical_graph = physical_graph
        self.model = Model("placement")
        self.run()

    def run(self):
        if self.multiplier is None:
            self.multiplier = DefaultMultiplierDict(self.apps, self.requests, self.physical_graph, self.nrf, self.erf,
                                                    self.run_number, self.exp_name)
        self.create_linear_program()
        self.solve_model()
        self.solution_post_processing()

    @timer("_solve_model")
    def solve_model(self):
        self.set_model_parameters()
        if config.create_solver_file:
            solver_log_path = str(self.dirpath / "solver.log")
        else:
            solver_log_path = False
        self.solver = self.model.solve(log_output=solver_log_path)
        if not self.solver:
            from docplex.mp.conflict_refiner import ConflictRefiner
            print("No solution found. Starting conflict refinement...")
            conflict_refiner = ConflictRefiner()
            conflicts = conflict_refiner.refine_conflict(self.model)
            for conflict in conflicts:
                print("Constraint: ", conflict.element, " - Status: ", conflict.status)
            self.model.dump_as_lp(str(self.dirpath / "model_failed.lp"))
            print(self.requests)
            exit(1)

    @timer("create_linear_program")
    @profile
    def create_linear_program(self):
        self.create_variables()
        self.node_combinations = FluidModel._get_all_different_pairs_in_list(self.physical_graph.nodes)
        self._create_constraints()
        self._add_parallel_constraints_and_objective()

    @profile
    def solution_post_processing(self):
        self.save_model_to_file()
        self.collect_share_allocation()
        self.create_model_summary_object()

    def solution_post_processing_from_pickle(self):
        import pickle
        with open(self.dump_non_zero, 'rb') as f:
            self.non_zero_vars = pickle.load(f)
        self.collect_share_allocation()
        self.create_model_summary_object()

    @profile
    def _add_parallel_constraints_and_objective(self):
        if self.oracle:
            self._add_constraints_to_model()
        else:
            with ThreadPoolExecutor() as executor:
                executor.submit(self._add_constraints_to_model)
                executor.submit(self.add_objective)

    def collect_share_allocation(self):
        from allocation_logger import AllocationGraph
        for r, req in enumerate(self.requests):
            alternatives = self.get_non_fic_alternatives(req.app_name)
            app = self.apps[req.app_name]
            for alt_name, graph in alternatives.items():
                alloc_graph = AllocationGraph(req, alt_name, self.multiplier, app, self.non_zero_vars)
                self.allocation_graphs[req.app_name, alt_name, req.assoc_node] = alloc_graph
        if self.oracle:
            return
        self._collect_graph_elements()
        for key, alloc_graph in self.allocation_graphs.items():
            alloc_graph.post_process_graph()
            self.allocation_graphs[key] = alloc_graph

    def _collect_graph_elements(self):
        alloc_graph_vars = {k: [] for k in self.allocation_graphs.keys()}
        for key, demand in self.non_zero_vars.items():
            app_name, id, assoc_node, i, j, m, n = key
            alloc_graph_vars[app_name, id, assoc_node].append((i, j, m, n, demand))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_graph, key, graph_vars) for key, graph_vars in
                       alloc_graph_vars.items()]
            for future in futures:
                future.result()

    def process_graph(self, key, graph_vars):
        alloc_graph = self.allocation_graphs[key]
        self._collect_graph_elements_per_graph(graph_vars, alloc_graph)

    def _collect_graph_elements_per_graph(self, graph_vars, alloc_graph):
        for key in graph_vars:  #i, j, m, n, demand = key
            alloc_graph.add_element(*key)

    @timer("save_model_to_file")
    def save_model_to_file(self):
        if config.create_lp_file:
            print("NOTICE: Writing LP file")
            self.model.dump_as_lp(str(self.dirpath / "model.lp"))
        self._export_solution()

    @profile
    def _export_solution(self):
        self._get_request_demand()
        with open(self.dirpath / "branch_share.txt", "w") as f:
            f.write(f"app_name,user_node,dest_node,choice_node,branch_name,branch_num,share\n")
        self.non_zero_weights = self._get_non_zero_weights()
        self._parse_solution()

    def _get_non_zero_weights(self, no_fic=True):
        if type(self.solver) == dict:
            if no_fic:
                return [(k, Decimal(str(v))) for k, v in self.solver.items() if v > 0 and 'fic_' not in k]
            else:
                return [(k, Decimal(str(v))) for k, v in self.solver.items() if v > 0]
        if no_fic:
            return [(v, Decimal(str(v.solution_value))) for v in self.model.iter_variables() if
                    v.solution_value > 0 and 'fic_' not in v.name]
        else:
            return [(v, Decimal(str(v.solution_value))) for v in self.model.iter_variables() if v.solution_value > 0]

    @profile
    def _parse_solution(self):
        total_virtual_node_allocation = 0
        total_virtual_link_allocation = 0
        total_node_capacity = self.physical_graph.get_total_node_capacity(avoid_gpu=False)
        total_link_capacity = self.physical_graph.get_total_link_capacity(avoid_gpu=False)

        solution_share_data = []
        node_app_allocation = []
        solution_cost_data = []
        branch_share_data = []
        total_allocated_demand = 0
        if len(self.non_zero_weights) == 0:
            return

        for v, value in self.non_zero_weights:
            app_name, alt_name, assoc_node, i, j, m, n = FileHandler.parse_variable(str(v))
            key = (app_name, alt_name, assoc_node, i, j, m, n)
            if self.oracle and (app_name, assoc_node) not in self.request_demand:
                continue
            allocated_demand = value * self.request_demand[app_name, assoc_node]
            allocated_demand_str = f'{value} * {self.request_demand[app_name, assoc_node]} = {allocated_demand}'
            src_node_indicator = ''
            if i == j and i == config.user_func:
                node_app_allocation.append([assoc_node, app_name, allocated_demand])
                src_node_indicator = '>>'
            self.non_zero_vars[key] = round(allocated_demand, 8)
            total_allocated_demand += allocated_demand
            alloc_type = 'N: ' if m == n else 'L: '

            solution_share_data.append(f"{src_node_indicator}{alloc_type}{v} = {allocated_demand_str}\n")

            shareCost = self.physical_graph.get_node_cost(m) if m == n else self.physical_graph.get_link_cost(m, n)
            multiplier = self.multiplier.get_node_multiplier(app_name, j, m) if m == n else self.multiplier[
                app_name, i, j, m, n]
            cost = shareCost * multiplier * allocated_demand

            self.cost += cost
            if m == n:
                total_virtual_node_allocation += allocated_demand * multiplier
            else:
                total_virtual_link_allocation += allocated_demand * multiplier

            solution_cost_data.append(f"{v} = {shareCost} * {multiplier} * {allocated_demand} = {cost}\n")

        self.link_utilization = round(total_virtual_link_allocation / total_link_capacity, 2)
        self.node_utilization = round(total_virtual_node_allocation / total_node_capacity, 2)

        if config.online_mode and self.oracle:
            return

        self._export_node_app_allocation(node_app_allocation)

        # Write to files after the loop
        with open(self.dirpath / f"solution_share.txt", "w") as f1:
            f1.writelines(solution_share_data)

        with open(self.dirpath / f"solution_cost.txt", "w") as f2:
            f2.writelines(solution_cost_data)

        if config.online_mode:
            return
        with open(self.dirpath / "branch_share.txt", "a") as f3:
            f3.writelines(branch_share_data)

            users, demand = 0, 0
            for req in self.requests:
                users += req.grouped_users
                demand += req.demand

            f3.write(f"METADATA\n")
            f3.write(f"OPT,{users},{demand},{self.cost}\n")

    def _export_node_app_allocation(self, node_app_allocation):
        if config.experiment != ExperimentType.FIC_ALTERNATIVES:
            return
        import pandas as pd
        print(f'Layers: {self.fic_layers}')
        df = pd.DataFrame(node_app_allocation, columns=['Node', 'App', 'Demand'])
        edge_nodes = [node for node in self.physical_graph.nodes if self.physical_graph.get_node_tier(node) == 1]
        apps = list(self.apps.keys())
        for node in edge_nodes:
            for app in apps:
                if df[(df['Node'] == node) & (df['App'] == app)].empty:
                    df.loc[len(df)] = [node, app, 0]
        df['Runtime'] = self.solver.solve_details.time
        df.to_csv(self.dirpath / 'node_app_allocation.csv', index=False)

    def _get_request_demand(self):
        raise NotImplementedError("This method should be implemented in the child class")

    def set_model_parameters(self, singleLP=False):
        max_cores = cpu_count()
        self.model.parameters.randomseed(config.seed + self.run_number)
        self.model.parameters.threads.set(min(16, max_cores))
        self.model.parameters.barrier.colnonzeros(150)
        self.model.parameters.barrier.startalg(4)
        self.model.parameters.preprocessing.dual(-1)
        self.model.parameters.advance(2)
        if singleLP:
            self.model.parameters.barrier.convergetol(1e-9)
            self.model.parameters.mip.tolerances.mipgap(1e-9)
            self.model.parameters.simplex.tolerances.feasibility(1e-9)
            self.model.parameters.emphasis.mip(2)
            self.model.parameters.mip.strategy.probe(3)
            self.model.parameters.mip.strategy.rinsheur = 10
        self.model.apply_parameters()

    def create_variables(self) -> None:
        if self.oracle or not config.online_mode:
            MIN_ALLOC_SHARE = 0
        else:
            MIN_ALLOC_SHARE = 0.01
        cpus = cpu_count()
        processes = min(cpus, len(self.requests))
        with ThreadPoolExecutor(processes) as pool:
            var_tuples = list(pool.map(self._create_variables_per_req, self.requests))
        var_list, var_fic_list, var_u_not_at_src, var_u_at_src = list(zip(*var_tuples))
        model_vars = {k: v for dic in var_list for k, v in dic.items()}
        self.var = self.model.continuous_var_dict(model_vars.keys(), lb=0, ub=1, name=model_vars.values())
        model_vars = {k: v for dic in var_u_not_at_src for k, v in dic.items()}
        self.var.update(self.model.continuous_var_dict(model_vars.keys(), lb=0, ub=0, name=model_vars.values()))
        model_vars = {k: v for dic in var_u_at_src for k, v in dic.items()}
        self.var.update(
            self.model.continuous_var_dict(model_vars.keys(), lb=MIN_ALLOC_SHARE, ub=1, name=model_vars.values()))
        if config.online_mode and self.fic_layers > 0:  #limit layers of fic by their ub
            fic_share = 1 / self.fic_layers
            model_vars_fic = {k: v for dic in var_fic_list for k, v in dic.items()}
            self.var.update(
                self.model.continuous_var_dict(model_vars_fic.keys(), lb=0, ub=fic_share, name=model_vars_fic.values()))

    def _create_variables_per_req(self, req: User) -> {}:
        if req.demand and not self.oracle <= 0:
            raise ValueError(f"Request {req.id} demand is {req.demand}")
        variables_u_not_at_src = {}
        variables_u_at_src = {}
        variables = {}
        variables_fic = {}
        alternatives = self.alternatives[req.app_name]
        physical_edges = self.physical_graph.get_edge_list()
        physical_nodes = self.physical_graph.get_node_list()
        app = self.apps[req.app_name]
        for alt_name, app_graph in alternatives.items():
            fic = True if 'fic' in alt_name else False
            app_edges = app.get_alternative_edge_list(alt_name)
            app_nodes = app.get_alternative_node_list(alt_name)
            name = f'{req.app_name},{alt_name},{req.assoc_node}'
            var_dict = {
                **{(req.id, alt_name, i, j, m, n): f'link[({name}),{i},{j},{m},{n}]'
                   for i, j in app_edges for m, n in physical_edges if not fic},
                **{(req.id, alt_name, i, i, m, m): f'node[({name}),{i},{m}]'
                   for i in app_nodes for m in physical_nodes if
                   i != config.user_func and (not fic or (fic and m == req.assoc_node))}
            }
            var_u_not_at_assoc_node = {
                **{(req.id, alt_name, i, i, m, m): f'node[({name}),{i},{m}]'
                   for i in app_nodes for m in physical_nodes if i == config.user_func and m != req.assoc_node and
                   (not fic or (fic and m == req.assoc_node))}
            }
            var_u_at_assoc_node = {
                **{(req.id, alt_name, i, i, m, m): f'node[({name}),{i},{m}]'
                   for i in app_nodes for m in physical_nodes if i == config.user_func and m == req.assoc_node and
                   (not fic or (fic and m == req.assoc_node))}
            }
            if 'fic' in alt_name:
                variables_fic.update(var_dict)
                variables_fic = {**variables_fic, **var_dict, **var_u_not_at_assoc_node, **var_u_at_assoc_node}
            else:
                variables.update(var_dict)
                variables_u_not_at_src.update(var_u_not_at_assoc_node)
                variables_u_at_src.update(var_u_at_assoc_node)

        return variables, variables_fic, variables_u_not_at_src, variables_u_at_src

    def add_objective(self):
        with ThreadPool(cpu_count()) as pool:
            dot_expr = pool.map(self._add_objective_per_request, self.requests)
        vars, coefficients = list(zip(*dot_expr))
        vars = FluidModel._join_list_of_lists(vars)
        coefficients = FluidModel._join_list_of_lists(coefficients)
        self.model.minimize(self.model.dot(vars, coefficients))

    def _add_objective_per_request(self, req: User) -> []:
        raise NotImplementedError("This method should be implemented in the child class")

    def _create_constraints(self):
        raise NotImplementedError("This method should be implemented in the child class")

    @timer("flow_preservation_constraint")
    def flow_preservation_constraint(self):
        node_to_incoming_edges = self.physical_graph.get_incoming_edges_dict()
        node_to_outgoing_edges = self.physical_graph.get_outgoing_edges_dict()
        func_input = [(request, node_to_incoming_edges, node_to_outgoing_edges) for request in self.requests]

        with ThreadPool(cpu_count()) as pool:
            constraints = pool.starmap(self._flow_preservation_constraint_per_request, func_input)
        self._add_constraints(constraints)

    def _flow_preservation_constraint_per_request(self, req: User, node_to_incoming_edges,
                                                  node_to_outgoing_edges) -> []:
        physical_graph_nodes = list(self.physical_graph.nodes())
        constraints = []
        constraint_names = []
        alternatives = self.alternatives[req.app_name]

        for alt_name, graph in alternatives.items():
            if 'fic' in alt_name:
                continue
            for i, j in graph.edges:
                for n in physical_graph_nodes:
                    incoming_edges = node_to_incoming_edges[n]
                    outgoing_edges = node_to_outgoing_edges[n]
                    sum_incoming_vars = self.model.sum_vars_all_different(
                        self.var[req.id, alt_name, i, j, m, n] for m, n in incoming_edges)
                    sum_outgoing_vars = self.model.sum_vars_all_different(
                        self.var[req.id, alt_name, i, j, n, l] for n, l in outgoing_edges)

                    constraints.append(
                        self.var[req.id, alt_name, i, i, n, n] + sum_incoming_vars ==
                        self.var[req.id, alt_name, j, j, n, n] + sum_outgoing_vars
                    )
                    constraint_names.append(f"FLOW_PRESERVATION_{req.id}_{i}_{j}_{n}")
        return constraints, constraint_names

    @timer("link_capacity_constraint")
    def link_capacity_constraint(self):
        raise NotImplementedError("This method should be implemented in the child class")

    @timer("link_zero_capacity_constraint")
    def link_zero_capacity_constraint(self):
        raise NotImplementedError("This method should be implemented in the child class")

    def _link_capacity_constraint_per_link(self, m, n):
        raise NotImplementedError("This method should be implemented in the child class")

    def _link_zero_capacity_constraint_per_link(self, m, n):
        raise NotImplementedError("This method should be implemented in the child class")

    @timer("node_capacity_constraint")
    def node_capacity_constraint(self):
        raise NotImplementedError("This method should be implemented in the child class")

    def _node_capacity_constraint_per_node(self, m: str) -> []:
        raise NotImplementedError("This method should be implemented in the child class")

    def _add_constraints_to_model(self, all_constraints=None):
        if all_constraints is None:
            all_constraints = self.constraints
        constraints, constraint_names = list(zip(*all_constraints))
        list_of_constraints = list(chain.from_iterable(constraints))
        list_of_constraint_names = list(chain.from_iterable(constraint_names))

        self.model.add_constraints(list_of_constraints, names=list_of_constraint_names)
        self.constraints = []

    def _add_constraints(self, constraints):
        self.constraints += constraints

    @staticmethod
    def _get_all_different_pairs_in_list(_list) -> []:
        _list = list(_list)
        a_to_b = [(a, b) for idx, a in enumerate(_list) for b in _list[idx + 1:]]
        b_to_a = [(b, a) for idx, a in enumerate(_list) for b in _list[idx + 1:]]
        joined_set = set(a_to_b + b_to_a)
        sorted_list = sorted(list(joined_set))
        return list(sorted_list)

    @staticmethod
    def _join_list_of_lists(_list):
        return list(chain.from_iterable(_list))

    def create_model_summary_object(self):
        self.summary = FluidModelSummary(self)

    def get_non_fic_alternatives(self, app_name) -> {}:
        alternatives = {}
        for alt_name, graph in self.alternatives[app_name].items():
            if 'fic_factor' not in graph.nodes[alt_name]:
                alternatives[alt_name] = graph
        return alternatives


class OfflineFluidModel(FluidModel):
    def __init__(self, physical_graph: EnhancedDiGraph, requests: [], apps: {}, alternatives: {}, exp_setting,
                 multiplier=None, dirpath=None, oracle=False, fic_layers=config.fic_layers):
        super().__init__(physical_graph, requests, apps, alternatives, exp_setting, multiplier, dirpath,
                         oracle, fic_layers)

    def _create_constraints(self):
        self.flow_preservation_constraint()
        self.sum_alt_share_constraint()
        self.link_capacity_constraint()
        self.node_capacity_constraint()

    def sum_alt_share_constraint(self):
        constraints = []
        constraint_names = []
        for req in self.requests:
            vars = self.model.sum(
                self.var[req.id, alt_name, config.user_func, config.user_func, req.assoc_node, req.assoc_node]
                for alt_name in self.alternatives[req.app_name])
            if config.online_mode and self.fic_layers > 0:
                constraint = (vars == 1)
            elif not config.online_mode:
                constraint = (vars <= 1)
            else:
                return
            constraints.append(constraint)
            constraint_names.append(f"SUM_ALT_SHARE_{req.app_name}_{req.assoc_node}")
        constraints = [(constraints, constraint_names)]
        self._add_constraints(constraints)

    def link_capacity_constraint(self):
        with ThreadPool() as pool:
            constraints = pool.starmap(self._link_capacity_constraint_per_link, self.physical_graph.edges)
        constraints = [c for c in constraints if c[1] is not None]
        constraints, constraint_names = list(zip(*constraints))
        constraints = [(constraints, constraint_names)]
        if self.oracle:
            constraints, constraint_names = list(zip(*constraints))
            list_of_constraints = list(chain.from_iterable(constraints))
            list_of_constraint_names = list(chain.from_iterable(constraint_names))
            self.model.add_constraints(list_of_constraints, names=list_of_constraint_names)
        else:
            self._add_constraints(constraints)

    def _link_capacity_constraint_per_link(self, m, n):
        vars = []
        coefficients = []
        for req in self.requests:
            alternatives = self.get_non_fic_alternatives(req.app_name)
            for alt_name, graph in alternatives.items():
                for i, j in graph.edges:
                    vars.append(self.var[req.id, alt_name, i, j, m, n])
                    coefficients.append(float(self.multiplier[(req.app_name, i, j, m, n)] * req.demand))
        try:
            lhs = self.model.dot(vars, coefficients)
        except Exception as e:
            print(e)
            raise e
        constraint = (lhs <= float(self.physical_graph.get_link_capacity(m, n)))
        constraint_name = f"LINK_CAPACITY_{m}_{n}"
        return constraint, constraint_name

    def node_capacity_constraint(self):
        with ThreadPool() as pool:
            constraints = pool.map(self._node_capacity_constraint_per_node, self.physical_graph.nodes())
        constraints, constraint_names = list(zip(*constraints))
        constraints = [(constraints, constraint_names)]
        if self.oracle:
            constraints, constraint_names = list(zip(*constraints))
            list_of_constraints = list(chain.from_iterable(constraints))
            list_of_constraint_names = list(chain.from_iterable(constraint_names))
            self.model.add_constraints(list_of_constraints, names=list_of_constraint_names)
        else:
            self._add_constraints(constraints)

    def _node_capacity_constraint_per_node(self, m: str) -> []:
        vars = []
        coefficients = []
        for req in self.requests:
            alternatives = self.get_non_fic_alternatives(req.app_name)
            for alt_name, graph in alternatives.items():
                for i in graph.nodes:
                    vars.append(self.var[req.id, alt_name, i, i, m, m])
                    coefficients.append(float(self.multiplier.get_node_multiplier(req.app_name, i, m) * req.demand))
        lhs = self.model.dot(vars, coefficients)
        constraint = (lhs <= float(self.physical_graph.get_node_capacity(m)))
        constraint_name = f"NODE_CAPACITY_{m}"
        return constraint, constraint_name

    def add_objective(self):
        with ThreadPool(cpu_count()) as pool:
            dot_expr = pool.map(self._add_objective_per_request, self.requests)
        vars, coefficients, pen_cost = list(zip(*dot_expr))
        vars = FluidModel._join_list_of_lists(vars)
        coefficients = FluidModel._join_list_of_lists(coefficients)
        load_cost = self.model.dot(vars, coefficients)
        cost_penalty = self.model.sum(FluidModel._join_list_of_lists(pen_cost))
        self.model.minimize(load_cost + cost_penalty)

    def _add_objective_per_request(self, req: User) -> []:
        try:
            var = self.var
            multiplier = self.multiplier
            vars = []
            coefficients = []
            pen_cost = []

            # Cached methods
            append_var = vars.append
            append_coeff = coefficients.append
            physical_graph = self.physical_graph
            physical_edges = physical_graph.edges
            physical_nodes = physical_graph.nodes

            edges_dict = {(m, n): physical_graph.get_link_cost(m, n) for (m, n) in physical_edges}
            nodes_dict = {m: physical_graph.get_node_cost(m) for m in physical_nodes}

            alternatives = self.alternatives[req.app_name]
            for alt_name, graph in alternatives.items():
                for i in graph.nodes():
                    for m in nodes_dict:
                        if 'fic' in alt_name and m != req.assoc_node:
                            continue
                        element = (req.app_name, i, i, m, m)
                        append_var(var[req.id, alt_name, i, i, m, m])
                        append_coeff(float(nodes_dict[m] * multiplier[element] * req.demand))
                if 'fic' in alt_name:
                    continue
                for i, j in graph.edges():
                    for (m, n) in edges_dict:
                        element = (req.app_name, i, j, m, n)
                        append_var(var[req.id, alt_name, i, j, m, n])
                        append_coeff(float(edges_dict[(m, n)] * multiplier[element] * req.demand))
            reject_penalty = sum(coefficients) * 2  # sum of costs to allocate request * demand * factor of 2
            if config.online_mode and len(alternatives) > 1:
                penalties = []
                for alt_name, graph in alternatives.items():
                    if 'fic_factor' in graph.nodes[alt_name]:
                        fic_factor = graph.nodes[alt_name]['fic_factor']
                        penalty = (self.var[
                            req.id, alt_name, config.user_func, config.user_func, req.assoc_node, req.assoc_node]) * reject_penalty * fic_factor
                        penalties.append(penalty)
                pen_cost.append(self.model.sum(penalties))
            else:
                pen_cost.append((1 - self.model.sum(
                    [self.var[req.id, alt_name, config.user_func, config.user_func, req.assoc_node, req.assoc_node]
                     for alt_name in alternatives.keys()])) * reject_penalty)
        except Exception as e:
            print(f'Objective exception: {e}')
            raise e
        return vars, coefficients, pen_cost

    def _get_request_demand(self):
        for req in self.requests:
            self.request_demand[req.app_name, req.assoc_node] = req.demand


class OracleBasicModel(OfflineFluidModel):
    def __init__(self, physical_graph: EnhancedDiGraph, requests: [], apps: {}, alternatives: {}, exp_settings,
                 multiplier=None, dirpath=None, oracle=False):
        super().__init__(physical_graph, requests, apps, alternatives, exp_settings, multiplier, dirpath,
                         oracle, fic_layers=0)

        self.run_oracle()
        self.node_app_set = self._create_node_app_set()

    def run(self):
        pass

    def run_oracle(self):
        if self.multiplier is None:
            self.multiplier = DefaultMultiplierDict(self.apps, self.requests, self.physical_graph, self.nrf,
                                                    self.erf, self.run_number, self.exp_name)
        self.create_linear_program()

    @timer("create_multiple_copies")
    def create_multiple_copies(self, model_copies, n=1):
        if len(model_copies) == 0:
            if n == 1:
                return [self.create_deepcopy()]
            else:
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.create_deepcopy) for _ in range(n)]
                    copies = [future.result() for future in futures]
                return copies
        else:
            return model_copies

    def create_deepcopy(self):
        copied_obj = self.__class__.__new__(self.__class__)

        copied_obj.physical_graph = self.physical_graph.copy()
        copied_obj.requests = self.requests.copy()
        copied_obj.seed = self.seed
        copied_obj.apps = self.apps
        copied_obj.alternatives = self.alternatives
        copied_obj.run_number = self.run_number
        copied_obj.nrf = self.nrf
        copied_obj.erf = self.erf
        copied_obj.exp_name = self.exp_name
        copied_obj.multiplier = self.multiplier
        copied_obj.dirpath = self.dirpath
        copied_obj.oracle = self.oracle
        copied_obj.fic_layers = self.fic_layers
        copied_obj.var = self.var.copy()
        copied_obj.model = self.copy_model(self.model)
        copied_obj.request_demand = {}
        copied_obj.allocation_graphs = {}
        copied_obj.solver = None
        copied_obj.constraints = self.constraints.copy()
        copied_obj.node_combinations = self.node_combinations
        copied_obj.summary = None
        copied_obj.cost = 0
        copied_obj.node_utilization = 0
        copied_obj.link_utilization = 0
        copied_obj.non_zero_weights = []
        copied_obj.non_zero_vars = {}
        copied_obj.node_app_set = self.node_app_set

        return copied_obj

    def _create_constraints(self):
        self.flow_preservation_constraint()
        # self.request_at_assoc_node_constraint()

    @profile
    def run_model(self, requests):
        self.add_request_dependant_constraints(requests)
        self.add_objective()
        self.solve_model()
        self.solution_post_processing()

    @profile
    def _export_solution(self):
        self._get_request_demand()
        self.non_zero_weights = self._get_non_zero_weights(no_fic=True)
        self._parse_solution()

    def nullify_unused_vars(self):
        from user_processor import UsersProcessor
        current_node_app_set = self._create_node_app_set()
        unused_pairs = self.node_app_set - current_node_app_set
        var_list = []
        for var_name, var in self.var.items():
            app_name, node = UsersProcessor.create_or_get_request_id(create=False, existing_id=var_name[0])
            if (node, app_name) in unused_pairs:
                var_list.append(var)
        self.model.add_constraints([self.model.sum(var_list) == 0], names=["NULLIFY_UNUSED_VARS"])

    def _create_node_app_set(self):
        node_app_set = set()
        for req in self.requests:
            node_app_set.add((req.assoc_node, req.app_name))
        return node_app_set

    def add_request_dependant_constraints(self, requests):
        self.requests = requests
        self.link_capacity_constraint()
        self.node_capacity_constraint()
        self.nullify_unused_vars()

    def solution_post_processing(self):
        self.save_model_to_file()

    @staticmethod
    def copy_model(model):
        actual_copy_name = "Copy of %s" % model.name
        copy_kwargs = model._get_kwargs()
        copy_context = model.context
        copy_model = Model(name=actual_copy_name, context=copy_context, **copy_kwargs)
        ctn_map = {}
        for ctn in model.iter_var_containers():
            copied_ctn = ctn.copy(copy_model)
            ctn_map[ctn] = copied_ctn
            copy_model._add_var_container(copied_ctn)

        memo = OracleBasicModel.make_memo(copy_model, model)

        # copy constraints
        linear_cts = []
        for ct in model.iter_constraints():
            linear_cts.append(OracleBasicModel.copy_ct(ct, copy_model))
        OracleBasicModel._new_constraint_block1(copy_model, linear_cts)

        # clone objective
        copy_model.set_objective_sense(model.objective_sense)
        copy_model.set_objective(model.objective_sense, model.objective_expr.copy(copy_model, memo))
        return copy_model

    @staticmethod
    def copy_ct(ct, target_model):
        return ct.__class__(target_model, ct.left_expr, ct.sense, ct.right_expr, ct.name)

    @staticmethod
    def _new_constraint_block1(model, cts):
        posted_cts = []
        ctseq = list(cts)
        for ct in ctseq:
            posted_cts.append(ct)
        # copy_model._post_constraint_block(posted_cts)
        OracleBasicModel._post_constraint_block(model, posted_cts)

    @staticmethod
    def _post_constraint_block(model, posted_cts):
        copy_model = model._lfactory
        ct_indices = OracleBasicModel.create_block_linear_constraints(model, posted_cts)
        copy_model._model._register_block_cts(copy_model._model._linct_scope, posted_cts, ct_indices)

    @staticmethod
    def create_block_linear_constraints(model, linct_seq):
        engine = model._lfactory._engine
        cpx_adapter = engine.cpx_adapter
        engine._resync_if_needed()
        block_size = len(linct_seq)
        linct_seq_list = [(ct.cplex_num_rhs(), ct.cplex_code, ct.safe_name, OracleBasicModel.linear_ct_to_cplex(ct)) for
                          ct in linct_seq]
        cpx_rhss, cpx_sense_string, cpx_names, cpx_linexprs = zip(*linct_seq_list)
        cpx_rhss = list(cpx_rhss)
        cpx_names = list(cpx_names)
        cpx_linexprs = list(cpx_linexprs)
        cpx_sense_string = "".join(cpx_sense_string)
        # cpx_linexprs = [OracleBasicModel.linear_ct_to_cplex(ct) for ct in linct_seq]
        ret_add = cpx_adapter.fast_add_linear(engine._cplex, cpx_linexprs, cpx_sense_string, cpx_rhss, cpx_names)
        return engine._allocate_range_index(size=block_size, ret_value=ret_add)

    @staticmethod
    def linear_ct_to_cplex(linear_ct):
        return OracleBasicModel.make_cpx_linear_from_exprs(linear_ct.get_left_expr(), linear_ct.get_right_expr())

    @staticmethod
    # @lru_cache(maxsize=None)
    def make_cpx_linear_from_exprs(left_expr, right_expr):
        indices = []
        coefs = []
        if right_expr is None or right_expr.is_constant():
            nb_terms = left_expr.number_of_terms()
            if nb_terms:
                indices = [-1] * nb_terms
                coefs = [0.0] * nb_terms
                for i, (dv, k) in enumerate(left_expr.iter_terms()):
                    indices[i] = dv._index
                    coefs[i] = float(k)

        elif left_expr.is_constant():
            nb_terms = right_expr.number_of_terms()
            if nb_terms:
                indices = [-1] * nb_terms
                coefs = [0] * nb_terms
                for i, (dv, k) in enumerate(right_expr.iter_terms()):
                    indices[i] = dv._index
                    coefs[i] = -float(k)

        else:
            from docplex.mp.constr import BinaryConstraint
            # hard to guess array size here:
            # we could allocate size(left) + size(right) and truncate, but??
            for dv, k in BinaryConstraint._generate_net_linear_coefs2_unsorted(left_expr, right_expr):
                indices.append(dv._index)
                coefs.append(float(k))
        return [indices, coefs]

    @staticmethod
    def make_memo(copy_model, model):
        memo = {}
        make_new_var = OracleBasicModel._make_new_var  # Local variable for faster access
        iter_variables = model.iter_variables
        for i, v in enumerate(iter_variables()):
            copied_var = make_new_var(i, copy_model, v.vartype, v.lb, v.ub, v.name)
            memo[v] = copied_var
        return memo

    @staticmethod
    def _make_new_var(var_num, model, vartype, lb, ub, varname):
        self_model = model._lfactory._model
        idx = OracleBasicModel.create_one_variable(var_num, model, vartype, lb, ub, varname)
        var = Var(self_model, vartype, varname, lb, ub, _safe_lb=True, _safe_ub=True)
        OracleBasicModel.__notify_new_model_object(var, idx, varname, self_model._vars_by_name, self_model._var_scope)
        return var

    @staticmethod
    def __notify_new_model_object(mobj, mindex, mobj_name, name_dir, idx_scope):
        mobj._set_index(mindex)
        name_dir[mobj_name] = mobj
        if idx_scope:
            idx_scope.notify_obj_index(mobj, mindex)

    @staticmethod
    def create_one_variable(var_num, model, vartype, lb, ub, name):
        lb = float(lb)
        ub = float(ub)
        indices = OracleBasicModel._create_variables(var_num, model, vartype, lb, ub, name)
        return indices[0]

    @staticmethod
    def _create_variables(var_num, model, vartype, lbs, ubs, name):
        model._lfactory._engine._resync_if_needed()
        cpx_types = model._lfactory._engine.compute_cpx_vartype(vartype.cplex_typecode, 1)
        return OracleBasicModel._create_cpx_variables(var_num, model, cpx_types, lbs, ubs, name)

    @staticmethod
    def _create_cpx_variables(var_num, model, cpx_vartypes, lbs, ubs, name):
        ret_add = OracleBasicModel.fast_add_cols(var_num, model, cpx_vartypes, lbs, ubs, [name])
        return model._lfactory._engine._allocate_range_index(size=1, ret_value=ret_add)

    @staticmethod
    def fast_add_cols(var_num, model, cpx_vartype, lb, ub, names):
        # assume lbs, ubs, names are list that are either [] or have len size
        engine = model._lfactory._engine
        cpx = engine._cplex
        cpx_e = cpx._env._e
        cpx_lp = cpx._lp
        size = 1
        engine.cpx_adapter.newcols(cpx_e, cpx_lp, obj=[], lb=[lb], ub=[ub], xctype=cpx_vartype, colname=names)
        return range(var_num, var_num + size)


class SingleLPModel(OfflineFluidModel):
    def __init__(self, physical_graph: EnhancedDiGraph, requests: [], apps: {}, alternatives: {}, exp_settings,
                 multiplier=None, dirpath=None, oracle=False,
                 fic_layers=0):
        super().__init__(physical_graph, requests, apps, alternatives, exp_settings, multiplier, dirpath,
                         oracle, fic_layers)

    @profile
    def run(self):
        if self.multiplier is None:
            self.multiplier = DefaultMultiplierDict(self.apps, self.requests, self.physical_graph, self.nrf, self.erf,
                                                    self.run_number, self.exp_name)
        self.create_linear_program()
        self.solve_model()
        if self.solver is None:
            return
        self.solution_post_processing()

    def solve_model(self):
        self.set_model_parameters()
        if config.create_solver_file:
            solver_log_path = str(self.dirpath / "solver.log")
        else:
            solver_log_path = False
        self.solver = self.model.solve(log_output=solver_log_path)

    @profile
    def set_model_parameters(self):
        self.model.parameters.randomseed(config.seed + self.run_number)
        self.model.parameters.preprocessing.presolve(0)
        self.model.apply_parameters()

    def solution_post_processing(self):
        self.save_model_to_file()
        self.collect_share_allocation()
        # self.create_model_summary_object()

    @profile
    def _create_constraints(self):
        node_to_incoming_edges = self.physical_graph.get_incoming_edges_dict()
        node_to_outgoing_edges = self.physical_graph.get_outgoing_edges_dict()
        func_input = [(request, node_to_incoming_edges, node_to_outgoing_edges) for request in self.requests]

        with ThreadPool(cpu_count()) as pool:
            flow_constraints = pool.starmap(self._flow_preservation_constraint_per_request, func_input)
            link_constraints = pool.starmap(self._link_capacity_constraint_per_link, self.physical_graph.edges)
            node_constraints = pool.map(self._node_capacity_constraint_per_node, self.physical_graph.nodes())

        self._add_constraints(flow_constraints)

        link_constraints = [c for c in link_constraints if c[1] is not None]
        link_constraints, link_constraint_names = list(zip(*link_constraints))
        link_constraints = [(link_constraints, link_constraint_names)]
        self._add_constraints(link_constraints)

        node_constraints, node_constraint_names = list(zip(*node_constraints))
        node_constraints = [(node_constraints, node_constraint_names)]
        self._add_constraints(node_constraints)

    def _parse_solution(self):
        if len(self.non_zero_weights) == 0:
            return

        for v, value in self.non_zero_weights:
            assert value <= 1 + config.allowed_error
            app_name, alt, assoc_node, i, j, m, n = FileHandler.parse_variable(str(v))
            key = (app_name, alt, assoc_node, i, j, m, n)
            allocated_demand = value * self.request_demand[app_name, assoc_node]
            self.non_zero_vars[key] = round(allocated_demand, 8)

    def create_variables(self) -> {}:
        var_dict = {}
        var_u_not_at_assoc_node = {}
        req = self.requests[0]
        alternatives = self.alternatives[req.app_name]
        physical_edges = self.physical_graph.get_edge_list()
        physical_nodes = self.physical_graph.get_node_list()
        app = self.apps[req.app_name]
        for alt_name, app_graph in alternatives.items():
            app_edges = app.get_alternative_edge_list(alt_name)
            app_nodes = app.get_alternative_node_list(alt_name)
            name = f'{req.app_name},{alt_name},{req.assoc_node}'
            var_dict = {
                **{(req.id, alt_name, i, j, m, n): f'link[({name}),{i},{j},{m},{n}]'
                   for i, j in app_edges for m, n in physical_edges},
                **{(req.id, alt_name, i, i, m, m): f'node[({name}),{i},{m}]'
                   for i in app_nodes for m in physical_nodes if not (i == config.user_func and m != req.assoc_node)}
            }
            var_u_not_at_assoc_node = {
                **{(req.id, alt_name, i, i, m, m): f'node[({name}),{i},{m}]'
                   for i in app_nodes for m in physical_nodes if i == config.user_func and m != req.assoc_node}
            }
        self.var = self.model.binary_var_dict(var_dict.keys(), lb=0, ub=1, name=var_dict.values())
        self.var.update(self.model.binary_var_dict(var_u_not_at_assoc_node.keys(), lb=0, ub=0,
                                                   name=var_u_not_at_assoc_node.values()))


class OnlineFluidModel(FluidModel):
    def __init__(self, physical_graph: EnhancedDiGraph, requests: [], apps: {}, alternatives, exp_settings,
                 multiplier=None, dirpath=None, duration=0, ilp=False):
        self.duration = duration
        self.is_ilp = ilp
        self.active_requests = self.get_active_requests_by_t(requests)
        super().__init__(physical_graph, requests, apps, alternatives, exp_settings, multiplier, dirpath,
                         fic_layers=0, online_opt=True)

    def get_active_requests_by_t(self, requests):
        active_requests = [[] for _ in range(self.duration)]
        for request in requests:
            for t in range(request.start_time, request.end_time):
                if t >= self.duration:
                    continue
                active_requests[t].append(request)
        return active_requests

    @profile
    def _create_constraints(self):
        self.flow_preservation_constraint()
        self.link_capacity_constraint()
        self.node_capacity_constraint()

    @timer("link_capacity_constraint")
    def link_capacity_constraint(self):
        with ThreadPool(cpu_count()) as pool:
            constraints = list(pool.starmap(self._link_capacity_constraint_per_link, self.physical_graph.edges))
        self._add_constraints(constraints)

    def _link_capacity_constraint_per_link(self, m, n):
        constraints = []
        constraint_names = []
        for t in range(self.duration):
            requests = self.active_requests[t]
            vars = [self.var[req.id, alt_name, i, j, m, n] for req in requests for alt_name, graph in
                    self.get_non_fic_alternatives(req.app_name).items() for i, j in graph.edges]
            coefficients = [float(self.multiplier[(req.app_name, i, j, m, n)] * req.demand) for req in requests
                            for alt_name, graph in self.get_non_fic_alternatives(req.app_name).items() for i, j in
                            graph.edges]
            try:
                lhs = self.model.dot(vars, coefficients)
            except Exception as e:
                print(e)
                raise e
            constraints.append(lhs <= float(self.physical_graph.get_link_capacity(m, n)))
            constraint_names.append(f"LINK_CAPACITY_{m}_{n}_t{t}")
        return constraints, constraint_names

    @timer("node_capacity_constraint")
    def node_capacity_constraint(self):
        with ThreadPool(cpu_count()) as pool:
            constraints = list(pool.map(self._node_capacity_constraint_per_node, self.physical_graph.nodes()))
        self._add_constraints(constraints)

    def _node_capacity_constraint_per_node(self, m: str) -> []:
        constraints = []
        constraint_names = []
        for t in range(self.duration):
            requests = self.active_requests[t]
            vars = [self.var[req.id, alt_name, i, i, m, m] for req in requests for alt_name, graph in
                    self.get_non_fic_alternatives(req.app_name).items() for i in graph.nodes]
            coefficients = [float(self.multiplier.get_node_multiplier(req.app_name, i, m) * req.demand) for req in
                            requests for alt_name, graph in
                            self.get_non_fic_alternatives(req.app_name).items() for i in graph.nodes]
            lhs = self.model.dot(vars, coefficients)
            constraints.append(lhs <= float(self.physical_graph.get_node_capacity(m)))
            constraint_names.append(f"NODE_CAPACITY_{m}_t{t}")
        return constraints, constraint_names

    def add_objective(self):
        cost_per_t = []
        for t in range(self.duration):
            requests = self.active_requests[t]
            with ThreadPool(cpu_count()) as pool:
                dot_expr = pool.map(self._add_objective_per_request, requests)
            vars, coefficients, pen_cost = list(zip(*dot_expr))
            vars = FluidModel._join_list_of_lists(vars)
            coefficients = FluidModel._join_list_of_lists(coefficients)
            load_cost = self.model.dot(vars, coefficients)
            cost_penalty = self.model.sum(FluidModel._join_list_of_lists(pen_cost))
            cost_per_t.append(load_cost + cost_penalty)
        self.model.minimize(self.model.sum(cost_per_t))

    def _add_objective_per_request(self, req: User) -> []:
        var = self.var
        multiplier = self.multiplier
        vars = []
        coefficients = []
        pen_cost = []

        # Cached methods
        append_var = vars.append
        append_coeff = coefficients.append
        physical_graph = self.physical_graph
        physical_edges = physical_graph.edges
        physical_nodes = physical_graph.nodes

        edges_dict = {(m, n): physical_graph.get_link_cost(m, n) for (m, n) in physical_edges}
        nodes_dict = {m: physical_graph.get_node_cost(m) for m in physical_nodes}
        alt_name = list(self.get_non_fic_alternatives(req.app_name).keys())[0]
        graph = self.get_non_fic_alternatives(req.app_name)[alt_name]
        for i, j in graph.edges():
            for (m, n) in edges_dict:
                element = (req.app_name, i, j, m, n)
                append_var(var[req.id, alt_name, i, j, m, n])
                append_coeff(float(edges_dict[(m, n)] * multiplier[element] * req.demand))
        for i in graph.nodes():
            for m in nodes_dict:
                element = (req.app_name, i, i, m, m)
                append_var(var[req.id, alt_name, i, i, m, m])
                append_coeff(float(nodes_dict[m] * multiplier[element] * req.demand))
        reject_penalty = sum(coefficients) * 5
        pen_cost.append((1 - self.model.sum(
            [self.var[
                 req.id, alt_name, config.user_func, config.user_func, req.assoc_node, req.assoc_node]])) * reject_penalty)
        return vars, coefficients, pen_cost

    def _get_request_demand(self):
        for req in self.requests:
            self.request_demand[req.app_name, req.assoc_node, req.req_demand] = req.demand

    def create_variables(self) -> None:
        cpus = cpu_count()
        processes = min(cpus, len(self.requests))
        with ThreadPoolExecutor(processes) as pool:
            var_tuples = list(pool.map(self._create_variables_per_req, self.requests))
        var_list, var_u_not_at_src, var_u_at_src = list(zip(*var_tuples))
        model_vars = {k: v for dic in var_list for k, v in dic.items()}
        if self.is_ilp:
            self.var = self.model.binary_var_dict(model_vars.keys(), lb=0, ub=1, name=model_vars.values())
        else:
            self.var = self.model.continuous_var_dict(model_vars.keys(), lb=0, ub=1, name=model_vars.values())
        model_vars = {k: v for dic in var_u_not_at_src for k, v in dic.items()}
        if self.is_ilp:
            self.var.update(self.model.binary_var_dict(model_vars.keys(), lb=0, ub=0, name=model_vars.values()))
        else:
            self.var.update(self.model.continuous_var_dict(model_vars.keys(), lb=0, ub=0, name=model_vars.values()))
        model_vars = {k: v for dic in var_u_at_src for k, v in dic.items()}
        if self.is_ilp:
            self.var.update(self.model.binary_var_dict(model_vars.keys(), lb=0, ub=1, name=model_vars.values()))
        else:
            self.var.update(self.model.continuous_var_dict(model_vars.keys(), lb=0, ub=1, name=model_vars.values()))

    def _create_variables_per_req(self, req: User) -> {}:
        variables_u_not_at_src = {}
        variables_u_at_src = {}
        variables = {}
        alternatives = self.alternatives[req.app_name]
        physical_edges = self.physical_graph.get_edge_list()
        physical_nodes = self.physical_graph.get_node_list()
        app = self.apps[req.app_name]
        for alt_name, app_graph in alternatives.items():
            app_edges = app.get_alternative_edge_list(alt_name)
            app_nodes = app.get_alternative_node_list(alt_name)
            name = f'{req.id},{req.app_name},{req.assoc_node}'
            var_dict = {
                **{(req.id, alt_name, i, j, m, n): f'link[({name}),{i},{j},{m},{n}]'
                   for i, j in app_edges for m, n in physical_edges},
                **{(req.id, alt_name, i, i, m, m): f'node[({name}),{i},{m}]'
                   for i in app_nodes for m in physical_nodes if i != config.user_func}
            }
            var_u_not_at_assoc_node = {
                **{(req.id, alt_name, i, i, m, m): f'node[({name}),{i},{m}]'
                   for i in app_nodes for m in physical_nodes if i == config.user_func and m != req.assoc_node}
            }
            var_u_at_assoc_node = {
                **{(req.id, alt_name, i, i, m, m): f'node[({name}),{i},{m}]'
                   for i in app_nodes for m in physical_nodes if i == config.user_func and m == req.assoc_node}
            }
            variables.update(var_dict)
            variables_u_not_at_src.update(var_u_not_at_assoc_node)
            variables_u_at_src.update(var_u_at_assoc_node)

        return variables, variables_u_not_at_src, variables_u_at_src

    def _parse_solution(self):
        if len(self.non_zero_weights) == 0:
            return
        for v, value in self.non_zero_weights:
            id, app_name, assoc_node, i, j, m, n = FileHandler.parse_variable(str(v))
            key = (app_name, int(id), assoc_node, i, j, m, n)
            allocated_demand = value
            self.non_zero_vars[key] = round(allocated_demand, 8)

    def collect_share_allocation(self):
        from allocation_logger import AllocationGraph
        for r, req in enumerate(self.requests):
            alternatives = self.get_non_fic_alternatives(req.app_name)
            app = self.apps[req.app_name]
            for alt_name, graph in alternatives.items():
                alloc_graph = AllocationGraph(req, alt_name, self.multiplier, app, self.non_zero_vars, is_online=True)
                self.allocation_graphs[req.app_name, req.id, req.assoc_node] = alloc_graph
        if self.oracle:
            return
        self._collect_graph_elements()
        for key, alloc_graph in self.allocation_graphs.items():
            alloc_graph.post_process_graph()
            self.allocation_graphs[key] = alloc_graph

    def solve_model(self):
        self.set_model_parameters(is_ilp=self.is_ilp)
        solver_log_path = str(self.dirpath / "solver.log")
        self.solver = self.model.solve(log_output=solver_log_path)

    def set_model_parameters(self, singleLP=False, is_ilp=False):
        max_cores = cpu_count()
        self.model.parameters.randomseed(config.seed + self.run_number)
        self.model.parameters.threads.set(min(16, max_cores))
        self.model.parameters.workmem(240000)
        if is_ilp:
            self.model.parameters.mip.tolerances.mipgap(0.05)
            self.model.parameters.emphasis.mip(1)
            self.model.parameters.mip.strategy.probe(3)
            self.model.parameters.mip.strategy.dive(2)
            self.model.parameters.mip.strategy.rinsheur = 75
        self.model.apply_parameters()


class FluidModelSummary:
    def __init__(self, fm: FluidModel):
        self.allocation = fm.allocation_graphs
        self.physical_graph = fm.physical_graph
        self.apps = fm.apps
        self.requests = fm.requests
        self.cost = fm.cost
        self.actual_erf = fm.link_utilization
        self.actual_nrf = fm.node_utilization
        self.multiplier = fm.multiplier
        if fm.model.solution is None:
            self.solution_cost = 0
            self.runtime = 0
        else:
            self.solution_cost = fm.model.solution.get_objective_value()
            self.runtime = fm.model.get_solve_details().time
            self._remove_redundant_allocation_attributes()

    def _remove_redundant_allocation_attributes(self):
        for key_pair in self.allocation.keys():
            alloc = self.allocation[key_pair]
            delattr(alloc, 'vars')