import numpy as np
from . import enums


online_mode = False # True for "Plan-Based Scalable Online Virtual Network Embedding", False for "The Power of Alternatives in Network Embedding"

# ================================== Debug Settings ==================================
check_feasible_node_popularities = False #check if demand at node doesn't exceed capacities
measure_function_runtime = False #use with @timer decorator
advanced_logging_mode = False  # Additional logs
create_lp_file = False  # CPLEX LP file
create_solver_file = False  # CPLEX solver file
create_node_popularity_file = False # Node popularity file
log_alternatives = False  #alternative-related logs


# ================================== Experiment Setup ==================================
skip_heuristic = False  #Don't run TANTO (offline) / OLIVE (online)
skip_greedy = False  #Greedy (offline) / QuickG (online)

#Offline algorithms
skip_individual_alternatives = True
#Online algorithms
skip_slot_off = False #Oracle mode
skip_fullg = False  #Full greedy

num_of_experiments = 30  #usually 30
skip_to_exp_num = None  # for debug, skip to experiment number


target_utilization = [1]  # utilization scales demand (offline) / number of requests (online)

# ================================== Topology Settings ==================================
topology_name = 'Ilan'
topology_dir = 'Topologies/topologyZoo'  # topology files, priority is gml files, then edgelist
# If True use strict ratios (nodes_to_core_ratio, transport_to_core_ratio), else use Jenks Natural Breaks classification
use_ratios_to_assign_nodes_to_tiers = True  # Used True in Online, False in Offline
nodes_to_core_ratio = 40
transport_to_core_ratio = 4

#Utilization scaling factor (mostly used in offline)
nrf = 1
erf = 1

#Base values, scaled up (by ~3) for each tier
base_link_capacity = 100000
base_node_capacity = 200000
link_cost = 1
node_cost = 0.01
tier_scale_ratio = 3

bidirectional_links = True
num_of_tiers = 3


# ================================== Application and User Settings ==================================
num_of_offline_requests = 10000
applications_to_use = ['cctv'] # edgelist file under 'include/applications', mostly in offline
app_popularity = enums.AppDist.UNIFORM
#distribution of users by geolocation. LOGNORMAL for offline, ZIPF for online. Rest is uniform (edge_node_uniform_probability)
edge_node_popularity = enums.UserDist.ZIPF
lognormal_node_popularity_hotspots = 2 # number of hotspots in lognormal edge node popularity

edge_popularity_zipf_alpha = 1 #if edge_node_popularity = enums.UserDist.ZIPF
edge_node_uniform_probability = 0.1 # uniform share of node popularity

# Request size statistics
request_demand_size_mean = 10
request_demand_size_std = 2

# ================================== Seed Configuration ==================================
use_arbitrary_seed = False
if use_arbitrary_seed:
    print("WARNING: using arbitrary seed")
    seed = np.random.randint(100)
else:
    seed = 43

# ================================== Application Specific Settings ==================================
acc_scaling_factor = 0.3  # scaling factor for accelerator VNF

#GPU (online mode)
gpu_nodes = False
edge_nodes_with_gpu_share = 0.1
gpu_multiplier = 1000

# app generator settings (online mode)
# distributions: app size - uniform, func - Binomial, link - Binomial
use_app_generator = True
agen_chain_apps = 2
agen_app_min_length = 3
agen_app_max_length = 5
agen_func_mean = 50
agen_func_std = 30
agen_link_mean = 50
agen_link_std = 30
agen_acc_apps = 1
agen_tree_apps = 1
agen_gpu_apps = 0




# ================================== Online Arrival Settings ==================================

multiprocess_algorithms = True #for parallel algorithm processing
export_requests = True  # export requests to file to avoid regeneration for the same setting
sim_time = 6000
lambda_per_node = 10 # requests per second per node
enable_preempt = True #preemt non-guaranteed requests

## MMPP (bursty demand) settings
request_arrival_process = enums.ArrivalProcess.MMPP #Markov Modulated Poisson Process (bursty demand)
mmpp_hot_cold_demand_ratio = 10
mmpp_cold_hot_duration_ratio = 10
mmpp_state_duration = 5

# Online requests
request_duration_distribution = enums.RequestDuration.GEOMETRIC
request_duration_mean = 10

train_set_ratio = 0.9

online_exp_type = enums.OnlineExpType.DIFFERENT_APPS # Online experiments
special_exp_params = [15] # If parameters are required for online experiments

# ================================== Recommended Settings ==================================
if online_mode:
    sim_time = 6000
    topology_name = 'Iris'
    nrf = 1
    erf = 1
    for app in applications_to_use:
        if 'gpu' in app:
            gpu_nodes = True
            break
    if use_app_generator and agen_gpu_apps > 0:
        gpu_nodes = True
    experiment = enums.ExperimentType.ONLINE
    evaluation_scenario_name = 'Online'
    fic_layers = 10  # number of classes
    if request_arrival_process == enums.ArrivalProcess.EXTERNAL: # requests from external file, pcap files under requests / caida
        external_file_time_scaling_factor = 100  # e.g., external file 10 sec, sim time 1000 sec
        adjust_node_popularities = False
    shift_train_set_hotspots = False  # Different node hotspots between train and test (experiment to test poor planning)
else:
    edge_node_popularity = enums.UserDist.LOGNORMAL
    applications_to_use = ['cctv']
    use_ratios_to_assign_nodes_to_tiers = False
    experiment = enums.ExperimentType.MULTI_SET
    evaluation_scenario_name = 'Offline'
    fic_layers = 0

# Analyze classes
if experiment == enums.ExperimentType.FIC_ALTERNATIVES:
    edge_node_popularity = enums.UserDist.UNIFORM
    online_exp_type = enums.OnlineExpType.SAME_HOTSPOTS
    fic_layers_num = [0, 2, 5, 10]
    target_utilization = [1]
    num_req_factor_for_scaling = 1
special_exp_params = [15]

# ================================== Misc ==================================
precision = 6 # floating point precision issues might arise during computation, ignore if less than 10e-6
allowed_error = 10 ** -precision

general_cost_division_factor = 10000  #scale costs for readability

# Global naming variables
agg_node = 'AGG'
full_app = 'Full'
user_func = 'U'