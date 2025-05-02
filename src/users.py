from dataclasses import dataclass, replace
import math
from decimal import Decimal
from include.enums import OnlineExpType
from include import config
import numpy as np


@dataclass(frozen=True)
class User:
    id: int = -1
    app_name: str = ''
    demand: Decimal = 0
    req_demand: Decimal = 0
    assoc_node: str = ''
    grouped_users: int = 1
    branch_id: int = None
    start_time: int = 0
    end_time: int = 1

    def update_demand(self, new_demand):
        return replace(self, demand=new_demand)

    def update_branch_id(self, new_branch_id):
        return replace(self, branch_id=new_branch_id)


class RequestDemandGenerator:
    def __init__(self, number_of_requests, seed, node_lambda=None):
        self.number_of_requests = number_of_requests
        self.seed = seed
        self.node_lambda = node_lambda
        np.random.seed(seed)

    def calculate_request_demand(self):
        self.calculate_users_demand_truncated_normal()
        if config.online_exp_type == OnlineExpType.LAMBDA:
            self.scale_demand_vector()
        return self.request_demand

    def scale_demand_vector(self):
        if self.node_lambda is None:
            raise ValueError("Node lambda is not provided")
        factor = config.lambda_per_node / self.node_lambda
        self.request_demand = [round(val * factor, 2) for val in self.request_demand]

    def calculate_users_demand_truncated_normal(self):
        """creates a list of demand units with predefined mean and std"""
        from scipy.stats import truncnorm
        mu, sigma = config.request_demand_size_mean, config.request_demand_size_std
        big_number = 1000
        myclip_a = 0
        myclip_b = mu * big_number
        my_mean = mu
        my_std = sigma

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        self.request_demand = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=self.number_of_requests, random_state=self.seed)
        self.request_demand = [math.ceil(val) for val in self.request_demand]
        if min(self.request_demand) <= 0:
            raise ValueError(f"Unexpected demand value {min(self.request_demand)}")
