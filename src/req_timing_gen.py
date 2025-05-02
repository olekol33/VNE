from online_handler import OnlineHandler
from decimal import Decimal
import hashlib
import json
import polars as pl
from code_testing import timer
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path
from users import RequestDemandGenerator
from include import config
from application import ApplicationSet
import numpy as np
from dataclasses import dataclass
from include.enums import ArrivalProcess, AppDist, UserDist, OnlineExpType, RequestDuration


@dataclass
class Events:
    time: int
    parameter: str
    value: int
    node: int = None


class RequestTimingGenerator:
    from physical_graph import Topology
    def __init__(self, apps: ApplicationSet, topology, seed, run_number, trace_special_input=None):
        np.random.seed(seed)
        self.warm_up_time = 2 * config.mmpp_state_duration
        self.sim_time = config.sim_time + self.warm_up_time
        self.trace_special_input = trace_special_input
        self.train_set_ratio = trace_special_input if config.online_exp_type == OnlineExpType.TRAIN_PERIOD_SHARE else config.train_set_ratio
        self.apps = apps
        self.run_number = run_number
        self.ordered_edge_nodes = topology.ordered_edge_nodes
        self.seed = seed
        self.requests_dir = Path(os.path.dirname(__file__)) / 'requests'
        if not self.requests_dir.exists():
            os.makedirs(self.requests_dir)
        self.params_dict, self.trace_hash = RequestGenFileHandler.get_params_and_trace_name(seed)
        self.requests_file, self.node_prob_file = RequestGenFileHandler.get_file_names(self.trace_hash)
        self.requests = None
        self.requests_per_second = OnlineHandler.get_lambda(topology.graph, lam=config.lambda_per_node)
        self.node_probs = self.get_node_probs(topology, seed) if config.online_exp_type == OnlineExpType.DIFFERENT_HOTSPOTS \
            else self.get_node_probs(topology)
        hot_cold_demand_ratio = config.mmpp_hot_cold_demand_ratio
        cold_hot_duration_ratio = config.mmpp_cold_hot_duration_ratio
        self.request_duration_mean = config.request_duration_mean
        if config.online_exp_type == OnlineExpType.MMPP_PARAMS:
            hot_cold_demand_ratio, cold_hot_duration_ratio = (trace_special_input, trace_special_input)
        elif config.online_exp_type == OnlineExpType.REQUEST_DURATION:
            self.request_duration_mean = trace_special_input
            scaled_lambda = (config.request_duration_mean / self.request_duration_mean) * config.lambda_per_node
            self.requests_per_second = OnlineHandler.get_lambda(topology.graph, lam=scaled_lambda)
        elif config.online_exp_type == OnlineExpType.LAMBDA:
            self.requests_per_second = OnlineHandler.get_lambda(topology.graph, lam=trace_special_input)
        if config.online_mode and config.request_arrival_process == ArrivalProcess.MMPP:
            self.original_mmpp_param_values = {'duration': self.request_duration_mean,
                                               'hot_cold_demand_ratio': hot_cold_demand_ratio,
                                               'cold_hot_duration_ratio': cold_hot_duration_ratio}
            self.hot_arrival_rate = 0
            self.cold_arrival_rate = 0
            self.get_arrival_rates(cold_hot_duration_ratio, hot_cold_demand_ratio)
            self.mmpp_states = [0, 1]  # low rate, high rate
            # Transition rates: state0, state1

            keys = self.ordered_edge_nodes
            current_state = np.random.choice(self.mmpp_states, size=len(keys))
            self.current_state = {key: current_state[i] for i, key in enumerate(keys)}
            self.time_in_state = {key: 0 for key in keys}
            self.arrival_rates = {key: 0 for key in keys}
            self.mean_times_in_state = {key: 0 for key in keys}
            for key in keys:
                self.set_arrival_rates_per_node(key)
                self.set_mean_time_in_state(cold_hot_duration_ratio, key)

    def get_arrival_rates(self, cold_hot_duration_ratio, hot_cold_demand_ratio):
        """lambda_c * Tc + lambda_h * Th = lambda   =>
        lambda_c * Tc + hot_cold_demand_ratio * lambda_c * (Tc / cold_hot_duration_ratio) = lambda =>
        lambda_c = lambda / (Tc + hot_cold_demand_ratio * (Tc / cold_hot_duration_ratio))"""
        cold_share, hot_share = self.get_cold_hot_duration_share(cold_hot_duration_ratio)
        self.cold_arrival_rate = self.requests_per_second / (
                cold_share + hot_cold_demand_ratio * (cold_share / cold_hot_duration_ratio))
        self.hot_arrival_rate = self.cold_arrival_rate * hot_cold_demand_ratio

    def set_arrival_rates_per_node(self, key):
        self.arrival_rates[key] = [self.cold_arrival_rate, self.hot_arrival_rate]  # Poisson arrival rates for each state

    def set_mean_time_in_state(self, cold_hot_duration_ratio, key):
        cold_share, hot_share = self.get_cold_hot_duration_share(cold_hot_duration_ratio)
        hot_duration = config.mmpp_state_duration * hot_share
        cold_duration = config.mmpp_state_duration * cold_share
        self.mean_times_in_state[key] = [cold_duration, hot_duration]

    def get_cold_hot_duration_share(self, cold_hot_duration_ratio):
        cold_share = cold_hot_duration_ratio / (1 + cold_hot_duration_ratio)
        hot_share = 1 - cold_share
        return cold_share, hot_share

    @timer('generate_req_timing')
    def generate_req_timing(self) -> []:
        if config.export_requests:
            if config.request_arrival_process == ArrivalProcess.EXTERNAL:
                pcap = PcapParser(self.apps, self.ordered_edge_nodes, self.run_number)
                self.requests = pcap.requests
                self.node_probs = pcap.probs
            else:
                self._generate_requests()
            self.node_probs = self.export_requests_and_node_probs()
        else:
            self.requests, self.node_probs = self.import_requests()
        self._update_params_if_external()
        self._ensure_arrival_rate()
        return self.requests, self.node_probs

    def _ensure_arrival_rate(self):
        arrival_rate = len(self.requests) / config.sim_time
        print(f"Arrival rate: {arrival_rate}")
        max_arrival_rate = 510
        if config.online_exp_type == OnlineExpType.MMPP_PARAMS:
            if arrival_rate < max_arrival_rate:
                raise ValueError(f"Arrival rate is too low: {arrival_rate}. Increase the duration or the number of requests")
            else:
                self.requests = self.requests.sample(n=max_arrival_rate * config.sim_time)
                pass


    def _update_params_if_external(self):
        if config.request_arrival_process != ArrivalProcess.EXTERNAL:
            return
        self.sim_time = self.requests[self.requests.height - 1, 'start_time']
        end_train_period = int(self.train_set_ratio * self.sim_time)
        num_train_requests = self.requests.filter(pl.col('start_time') <= end_train_period).height
        self.requests_per_second = num_train_requests / end_train_period
        self.warm_up_time = 0

    def _parse_external_file(self):
        pass

    def _generate_requests(self):
        node_data_list = [(node_index, node, self.node_probs[node_index])
                          for node_index, node in enumerate(self.ordered_edge_nodes)]
        if os.cpu_count() < 16:
            executor_class = ThreadPoolExecutor
        else:
            executor_class = ProcessPoolExecutor
        with executor_class() as executor:
            results = list(executor.map(self._generate_req_timing_per_node, node_data_list))
        requests_list = [item for sublist in results for item in sublist]
        self.requests = pl.DataFrame(requests_list,
                                     schema={"id": pl.Int64, "app_name": pl.Utf8, "demand": pl.Int64,
                                             "assoc_node": pl.Utf8,
                                             "start_time": pl.Int64, "end_time": pl.Int64})
        request_demand = RequestDemandGenerator(len(self.requests), self.seed, self.trace_special_input).calculate_request_demand()
        self.requests = self.requests.with_columns(demand=pl.Series(request_demand))

    def _generate_req_timing_per_node(self, node_data):
        additional_req_factor = 1
        node_index, node, node_prob = node_data
        state_key = node
        if config.online_exp_type == OnlineExpType.DEMAND_HIKE:
            additional_req_factor = 1
        requests_list = []
        id = 0  # Each process will have its own id counter, which will be corrected later.

        t = -self.warm_up_time
        np.random.seed(self.seed + node_index)
        rng = np.random.RandomState(self.seed + node_index)
        apps = list(self.apps.get_apps_dict().keys())
        app_index = 0
        prob_count = 0
        while t < self.sim_time:
            start_time = round(t)
            end_time = round(start_time + self.draw_duration(rng, request_duration_mean=self.request_duration_mean))
            to_allocate = rng.uniform() < node_prob
            if end_time > 0 and to_allocate:
                app = apps[app_index]
                app_index = (app_index + 1) % len(apps)
                requests_list.append((id, app, 1, node, start_time, end_time))
                id += 1
            t += self._draw_interarrival(state_key, rng) / additional_req_factor
        return requests_list

    def export_requests_and_node_probs(self):
        self.requests.write_csv(self.requests_file)
        self._append_params_dict()

        probs_list = []
        for node_index, node in enumerate(self.ordered_edge_nodes):
            probs_list.append((node_index, node, self.node_probs[node_index]))
        probs_df = pl.DataFrame(probs_list,
                                schema={'id': pl.Int64, 'node': pl.Utf8, 'prob': pl.Decimal(precision=None, scale=6)})
        probs_df = probs_df.with_columns(
            probs_df['prob'].cast(pl.Utf8)
        )
        probs_df.write_csv(self.node_prob_file)
        return probs_df

    def _append_params_dict(self):
        with open(self.requests_file, 'a') as f:
            f.write('#')
            json.dump(self.params_dict, f)
        with open(self.requests_dir / 'hashmap.txt', 'a') as f:
            f.write(f'{self.trace_hash} : {self.params_dict}\n')

    @timer('import_requests')
    def import_requests(self):
        if not os.path.exists(self.requests_file):
            raise FileNotFoundError(f'{self.requests_file} does not exist')
        probs = RequestGenFileHandler.get_node_probs_from_file(self.node_prob_file)
        probs = probs.drop('id')
        requests = pl.read_csv(self.requests_file, comment_prefix='#', schema={'id': pl.Int64, 'app_name': pl.Utf8,
                                                                               'demand': pl.Int64,
                                                                               'assoc_node': pl.Utf8,
                                                                               'start_time': pl.Int64,
                                                                               'end_time': pl.Int64})
        return requests, probs

    @staticmethod
    def draw_duration(rng, request_duration_mean=None) -> int:
        if config.request_duration_distribution == RequestDuration.GEOMETRIC:
            return rng.geometric(p=1 / request_duration_mean)
        elif config.request_duration_distribution == RequestDuration.CONSTANT:
            return request_duration_mean
        else:
            raise NotImplementedError(f'{config.request_duration_distribution} is not implemented')

    def _draw_interarrival(self, key, rng) -> float:
        if config.request_arrival_process == ArrivalProcess.MMPP:
            return self._generate_mmpp_interarrival(key, rng)
        elif config.request_arrival_process == ArrivalProcess.CONSTANT:
            return 1 / self.requests_per_second
        elif config.request_arrival_process == ArrivalProcess.POISSON:
            return rng.exponential(1 / self.requests_per_second)
        else:
            raise NotImplementedError(f'{config.request_arrival_process} is not implemented')

    def _generate_mmpp_interarrival(self, key, rng) -> float:
        if self.time_in_state[key] <= 0:
            self.current_state[key] = 1 - self.current_state[key]
            mean_time_in_state = self.mean_times_in_state[key][self.current_state[key]]
            self.time_in_state[key] = rng.exponential(mean_time_in_state)

        arrival_rate = self.arrival_rates[key][self.current_state[key]]
        arrival_time = rng.exponential(1 / arrival_rate)
        self.time_in_state[key] -= arrival_time
        return arrival_time


    def get_node_probs(self, topology: Topology, seed=None, get_df=False):
        if config.export_requests and not get_df:
            return topology.get_node_probs(seed)
        else:
            return RequestGenFileHandler.get_node_probs_from_file(self.node_prob_file)


class RequestGenFileHandler:
    @staticmethod
    def get_params_and_trace_name(seed):
        params = RequestGenFileHandler.get_trace_params(seed)
        serialized_params = json.dumps(params, sort_keys=True)
        hash_object = hashlib.md5(serialized_params.encode())
        hash_hex = hash_object.hexdigest()
        filename = f"{hash_hex}"
        return params, filename

    @staticmethod
    def get_trace_params(seed):
        exp_type = {
            OnlineExpType.SAME_HOTSPOTS: 'same_hotspots',
            OnlineExpType.HOTSPOT_INTENSITY: 'hotspot_intensity',
            OnlineExpType.DIFFERENT_HOTSPOTS: 'different_hotspots',
            OnlineExpType.NUMBER_OF_APPS: 'number_of_apps',
            OnlineExpType.LENGTH_OF_APPS: 'length_of_apps',
            OnlineExpType.DEMAND_HIKE: 'demand_hike',
            OnlineExpType.LINK_CAPACITY: 'link_capacity',
            OnlineExpType.MMPP_PARAMS: 'mmpp_params',
            OnlineExpType.REQUEST_DURATION: 'request_duration',
            OnlineExpType.TRAIN_PERIOD_SHARE: 'train_period_share',
            OnlineExpType.LAMBDA: 'lambda',
            OnlineExpType.DIFFERENT_APPS: 'diff_apps',
            OnlineExpType.PERCENTILE: 'percentile',
            OnlineExpType.CLASSES: 'layers'
        }
        duration = config.sim_time if config.request_arrival_process != ArrivalProcess.EXTERNAL else 0
        lambda_per_node = config.lambda_per_node if config.request_arrival_process != ArrivalProcess.EXTERNAL else 0
        edge_popularity_zipf_alpha = config.edge_popularity_zipf_alpha if (config.edge_node_popularity == UserDist.ZIPF
                                                                           and not config.request_arrival_process == ArrivalProcess.EXTERNAL) else 0
        agen_chain_apps = config.agen_chain_apps if config.use_app_generator else 0
        agen_app_min_size = config.agen_app_min_length if config.use_app_generator else 0
        agen_app_max_size = config.agen_app_max_length if config.use_app_generator else 0
        agen_func_mean = config.agen_func_mean if config.use_app_generator else 0
        agen_func_std = config.agen_func_std if config.use_app_generator else 0
        agen_link_mean = config.agen_link_mean if config.use_app_generator else 0
        agen_link_std = config.agen_link_std if config.use_app_generator else 0
        agen_acc_apps = config.agen_acc_apps if config.use_app_generator else 0
        agen_tree_apps = config.agen_tree_apps if config.use_app_generator else 0
        agen_gpu_apps = config.agen_gpu_apps if config.agen_gpu_apps else 0
        return {
            'topology': config.topology_name,
            'exp_type': exp_type[config.online_exp_type],
            'duration': duration,
            'train_ratio': config.train_set_ratio,
            'interarrival': config.request_arrival_process.value,
            'lambda_per_node': lambda_per_node,
            'duration_dist': config.request_duration_distribution.value,
            'edge_node_popularity': config.edge_node_popularity.value,
            'edge_popularity_zipf_alpha': edge_popularity_zipf_alpha,
            'state_duration': int(config.mmpp_state_duration),
            'request_duration_mean': int(config.request_duration_mean),
            'apps': config.applications_to_use,
            'seed': seed,
            'nodes_to_core_ratio': config.nodes_to_core_ratio,
            'transport_to_core_ratio': config.transport_to_core_ratio,
            'online_exp_type': config.online_exp_type.value,
            'use_app_generator': config.use_app_generator,
            'agen_chain_apps': agen_chain_apps,
            'agen_app_min_size': agen_app_min_size,
            'agen_app_max_size': agen_app_max_size,
            'agen_func_mean': agen_func_mean,
            'agen_func_std': agen_func_std,
            'agen_link_mean': agen_link_mean,
            'agen_link_std': agen_link_std,
            'agen_acc_apps': agen_acc_apps,
            'agen_tree_apps': agen_tree_apps,
            'agen_gpu_apps': agen_gpu_apps

        }

    @staticmethod
    def get_file_names(trace_hash):
        requests_dir = RequestGenFileHandler.get_requests_dir()
        requests_file = requests_dir / f'{trace_hash}.csv'
        node_prob_file = requests_dir / f'{trace_hash}_node_probs.csv'
        return requests_file, node_prob_file

    @staticmethod
    def get_requests_dir():
        return Path(os.path.dirname(__file__)) / 'requests'

    @staticmethod
    def get_node_probs_from_file(node_prob_file):
        if not os.path.exists(node_prob_file):
            raise FileNotFoundError(f'{node_prob_file} does not exist')
        return pl.read_csv(node_prob_file, schema={'id': pl.Int64, 'node': pl.Utf8, 'prob': pl.Utf8})


class PcapParser:
    def __init__(self, apps, ordered_edge_nodes, run_number):
        self.apps = apps
        self.ordered_edge_nodes = ordered_edge_nodes
        self.requests_dir = Path(os.path.dirname(__file__)) / 'requests' / 'caida'
        self.probs = None
        filename, pcap_file = self.get_pcap_file(run_number)
        self.syn_file = self.requests_dir.parent / f'{filename}.syn'
        if self.syn_file.exists():
            syn_requests_df = pl.read_csv(self.syn_file)
        else:
            syn_requests_df = self._extract_syn_requests_from_pcap(pcap_file)
            syn_requests_df.write_csv(self.syn_file)
        print(f"Using external pcap file: {filename}")
        self.requests = self._parse_syn(syn_requests_df)

    def get_pcap_file(self, run_number):
        pcap_files = self.requests_dir / 'traces.txt'
        url, filename = self.get_pcap_web_path_and_filename(pcap_files, run_number)
        pcap_file = self.download_file(url)
        return filename, pcap_file


    @staticmethod
    def download_file(url):
        import requests
        from requests.auth import HTTPBasicAuth
        import gzip
        import io
        # LOGIN to CAIDA
        username = 'TODO'
        password = 'TODO'
        response = requests.get(url.strip(), auth=HTTPBasicAuth(username, password))
        if response.status_code == 200:
            file_bytes = io.BytesIO(response.content)
            with gzip.GzipFile(fileobj=file_bytes, mode='rb') as gz_file:
                decompressed_data = gz_file.read()
                return decompressed_data
        else:
            print(f"Failed to download file. Status code: {response.status_code}")


    @staticmethod
    def get_pcap_web_path_and_filename(pcap_files, run_number):
        if not pcap_files.exists():
            raise FileNotFoundError(f'{pcap_files} does not exist')
        with open(pcap_files, 'r') as f:
            lines = f.readlines()
            line = lines[run_number]
            trace_name = line.split('/')[-1]
            return line, trace_name.split('gz')[0]



    def _parse_syn(self, df):
        rng = np.random.RandomState(config.seed)
        df = df.with_columns(pl.col('src').apply(self.first_16_bits, return_dtype=pl.Utf8).alias('src'))
        df = df.with_columns(df["src"].apply(self.hash_ip_to_node, return_dtype=pl.Utf8).alias("assoc_node"))
        df = self._add_apps_to_df(df)
        self.create_node_probs(df)
        durations = [RequestTimingGenerator.draw_duration(rng=rng, request_duration_mean=config.request_duration_mean) for _ in range(len(df))]
        df = df.with_columns(pl.Series("duration", durations))
        df = df.with_columns(pl.col("start_time").mul(config.external_file_time_scaling_factor).cast(pl.Int64))
        df = df.with_columns(pl.Series("end_time", df["start_time"] + df["duration"]))
        df = df.sort('start_time').with_row_count().rename({'row_nr': 'id'})
        df = df.drop(['duration', 'src'])
        request_demand = RequestDemandGenerator(len(df), config.seed).calculate_request_demand()
        df = df.with_columns(pl.Series("demand", request_demand))
        df = df.select(['id', 'app_name', 'demand', 'assoc_node', 'start_time', 'end_time'])
        request_rate = len(df) / df['end_time'].max()
        print(f"External file request rate: {request_rate}, duration: {df['end_time'].max()}")
        return df

    def create_node_probs(self, df):
        grouped_df = df.group_by('assoc_node').count()
        total = grouped_df['count'].sum()
        grouped_df = grouped_df.with_columns(pl.Series('prob', grouped_df['count'] / total))
        nodes_and_probs = zip(grouped_df['assoc_node'].to_list(), grouped_df['prob'].to_list())
        nodes_and_probs = {node: Decimal(str(prob)) for node, prob in nodes_and_probs}
        self.probs = [0] * len(self.ordered_edge_nodes)
        for node_index, node in enumerate(self.ordered_edge_nodes):
            self.probs[node_index] = nodes_and_probs[node]

    @staticmethod
    def first_16_bits(ip):
        parts = ip.split('.')
        return f"{parts[0]}.{parts[1]}"

    def hash_ip_to_node(self, ip_address):
        hash_object = hashlib.sha256(ip_address.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        node_index = hash_int % len(self.ordered_edge_nodes)
        return self.ordered_edge_nodes[node_index]

    def _add_apps_to_df(self, df):
        if config.app_popularity == AppDist.UNIFORM:
            apps = np.random.choice(list(self.apps.get_apps_dict().keys()), len(df))
            df = df.with_columns(pl.Series("app_name", apps))
        else:
            raise NotImplementedError(f'{config.app_popularity} is not implemented')
        return df

    def _add_identical_requests(self, df, app_name, node, n=1):
        """Takes first n rows and adds the same rows with node and app_name to the start of df"""
        first_n_rows = df.head(n)
        new_rows = first_n_rows.with_columns(pl.Series('app_name', [app_name] * n))
        new_rows = new_rows.with_columns(pl.Series('assoc_node', [node] * n))
        return pl.concat([new_rows, df])


    @timer("_extract_syn_requests_from_pcap")
    def _extract_syn_requests_from_pcap(self, pcap_data=None):
        import dpkt
        import socket
        import io
        syn_requests_list = []
        with io.BytesIO(pcap_data) as f:
            pcap = dpkt.pcap.Reader(f)
            start_time = None
            for ts, buf in pcap:
                try:
                    ip = dpkt.ip.IP(buf)
                    if not isinstance(ip.data, dpkt.tcp.TCP):
                        continue
                    tcp = ip.data
                    if (tcp.flags & dpkt.tcp.TH_SYN) and not (tcp.flags & dpkt.tcp.TH_ACK):
                        if start_time is None:
                            start_time = ts
                        src_ip = socket.inet_ntoa(ip.src)
                        dst_ip = socket.inet_ntoa(ip.dst)
                        time = ts - start_time
                        syn_requests_list.append({'start_time': time, 'src': src_ip, 'dst': dst_ip})
                except dpkt.dpkt.UnpackError:
                    continue


        # Convert the list to a Polars DataFrame
        syn_requests_df = pl.DataFrame(syn_requests_list, schema={'start_time': pl.Float64, 'src': pl.Utf8, 'dst': pl.Utf8})
        print(f"Extracted {len(syn_requests_df)} SYN requests from file")

        return syn_requests_df
