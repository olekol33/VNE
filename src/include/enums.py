from enum import Enum


class UserDist(Enum):
    UNIFORM = 0
    LOGNORMAL = 1
    NORMAL = 2
    INPUT = 3
    ZIPF = 4


class GraphAttrSource(Enum):
    EXTERNAL = 0
    DISTANCE = 1

class GraphLinkCost(Enum):
    FIXED = 0
    DISTANCE = 1


class SpecialAtrtribute(Enum):
    ZIP = 0
    OPT = 1

class AppDist(Enum):
    UNIFORM = 0
    ZIPF = 1


class ExperimentType(Enum):
    DONT_RUN = 0
    STATIC_ALTERNATIVE_SELECTION_SWEEP_EXP = 1
    ALTERNATIVE_COMPARISON_WITH_MULTIPLIERS = 2
    HEURISTIC_COMPARISON_STATIC = 3
    MULTI_SET = 4
    ONLINE = 5
    FIC_ALTERNATIVES = 6


class ToyScenario(Enum):
    NO_CAP_LIM = 0
    LINK_CAP_LIM = 1
    EDGE_CAP_LIM = 2
    EDGE_LINK_CAP_LIM = 3
    CORE_CAP_LIM = 4


class SpecialNodeAttributes(Enum):
    ZIP: str = 'zip'
    ACCELERATOR: str = 'acc'
    GPU: str = 'gpu'


class ZipNodeTypes(Enum):
    ZIP: str = 'z'
    UNZIP: str = 'uz'

class ArrivalProcess(Enum):
    CONSTANT: str = 'const'
    GEOMETRIC: str = 'geom'
    POISSON: str = 'poisson'
    MMPP: str = 'bursty'
    EXTERNAL: str = 'external'  #external file

class RequestDuration(Enum):
    CONSTANT: str = 'const'
    GEOMETRIC: str = 'geom'


class GreedyInvalid(Enum):
    NODE_REMOVED = 0
    DST_INVALID = 1
    SRC_INVALID = 2
    LINK_REMOVED = 3


class FlowTypes(Enum):
    LOCAL = 0
    TRANSIENT = 1
    LOCAL_PARTIAL = 2


class OnlineExpType(Enum):
    SAME_HOTSPOTS = 0
    HOTSPOT_INTENSITY = 1
    DIFFERENT_HOTSPOTS = 2
    DEMAND_HIKE = 3
    NUMBER_OF_APPS = 4
    LENGTH_OF_APPS = 5
    LINK_CAPACITY = 6
    MMPP_PARAMS = 7
    REQUEST_DURATION = 8
    TRAIN_PERIOD_SHARE = 9
    LAMBDA = 10
    DIFFERENT_APPS = 11  #basic experiment,  requests can use different apps
    PERCENTILE = 12
    CLASSES = 13