# External package imports
from numpy import arange
from numpy.random import default_rng
from scipy.stats import zipfian
import networkx as nx

# Builtin imports
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple
from hashlib import sha256

# Internal imports
from .policies import *
from .vip import *
from .utils import convertListFieldsToTuples, namedProduct


@dataclass(eq=True, frozen=True)
class SimulationParameters:
    fwd_pol: str = "none"
    cache_pol: str = "none"
    num_objects: int = 1000
    source_read_rate: int = 1000
    source_map_seed: int = 1
    request_generator_seed: int = 1
    stop_time: int = 100
    request_rate: int = 10
    request_dist_param: float = 0.75
    request_dist_type: str = "zipf"
    pen_weight: int = 0
    vip_inc: int = 1
    vip_slot_len: int = 1
    vip_win_size: int = 100
    cache_capacities: Tuple[int] = (5, 100)
    cache_read_rates: Tuple[int] = (20, 10)
    cache_write_rates: Tuple[int] = (20, 10)
    cache_read_pens: Tuple[int] = (2, 1)
    cache_write_pens: Tuple[int] = (4, 2)

    def __iter__(self):
        return iter(self.__dict__.items())

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return ",".join([f"{key}={value}" for key, value in self.items()])

    def __hash__(self):
        return int(sha256(self.__repr__().encode()).hexdigest(), 16)


def simConfigToParamSets(config):
    param_sets = [params for params in namedProduct(**config)]
    param_sets = [params for params in filter(ignoreDudFilter, param_sets)]
    param_sets = [convertListFieldsToTuples(params) for params in param_sets]
    param_sets = [SimulationParameters(**params) for params in param_sets]
    return param_sets


def getNode(env, node_id, fwd_pol, cache_pol, **kwargs):
    match fwd_pol, cache_pol:
        case "none", "none":
            return Node(env, node_id)
        case "none", "lru":
            return LRUNode(env, node_id, kwargs["num_objects"])
        case "none", "lfu":
            return LFUNode(env, node_id, kwargs["num_objects"])
        case "none", "wlfu":
            return WLFUNode(env, node_id, kwargs["num_objects"], kwargs["vip_args"]["vip_win_size"], kwargs["vip_args"]["vip_slot_len"])
        case "none", "fifo":
            return FIFONode(env, node_id)
        case "none", "unif":
            return UNIFNode(env, node_id)
        case "none", "palfu":
            return PALFUNode(env, node_id, kwargs["num_objects"], kwargs["pen_weight"])
        case "none", "pawlfu":
            return PAWLFUNode(env, node_id, kwargs["num_objects"], kwargs["vip_args"]["vip_win_size"], kwargs["vip_args"]["vip_slot_len"], kwargs["pen_weight"])
        case "rr", "none":
            return RoundRobinNode(env, node_id)
        case "rr", "lru":
            return RRLRUNode(env, node_id, kwargs["num_objects"])
        case "rr", "lfu":
            return RRLFUNode(env, node_id, kwargs["num_objects"])
        case "rr", "wlfu":
            return RRWLFUNode(env, node_id, kwargs["num_objects"])
        case "rr", "fifo":
            return RRFIFONode(env, node_id)
        case "rr", "unif":
            return RRUNIFNode(env, node_id)
        case "rr", "palfu":
            return RRPALFUNode(
                env, node_id, kwargs["num_objects"], kwargs["pen_weight"]
            )
        case "lrt", "none":
            return LeastResponseTimeNode(env, node_id)
        case "lrt", "lru":
            return LRTLRUNode(env, node_id, kwargs["num_objects"])
        case "lrt", "lfu":
            return LRTLFUNode(env, node_id, kwargs["num_objects"])
        case "lrt", "wlfu":
            return LRTWLFUNode(env, node_id, kwargs["num_objects"], kwargs["vip_args"]["vip_win_size"], kwargs["vip_args"]["vip_slot_len"])
        case "lrt", "fifo":
            return LRTFIFONode(env, node_id)
        case "lrt", "unif":
            return LRTUNIFNode(env, node_id)
        case "lrt", "palfu":
            return LRTPALFUNode(
                env, node_id, kwargs["num_objects"], kwargs["pen_weight"]
            )
        case "lrt", "pawlfu":
            return LRTPAWLFUNode(
                env, node_id, kwargs["num_objects"], kwargs["vip_args"]["vip_win_size"], kwargs["vip_args"]["vip_slot_len"], kwargs["pen_weight"]
            )
        case "vip", cache_pol if cache_pol in [
            "none",
            "vip",
            "vip2",
            "vipsbw",
            "vipsbw2",
            "mvip",
            "mvipsbw",
        ]:
            match cache_pol:
                case cache_pol if cache_pol in ["none", "vip"]:
                    return VIPNode(
                        env,
                        node_id,
                        kwargs["num_objects"],
                        kwargs["pen_weight"],
                        **kwargs["vip_args"],
                    )
                case "vip2":
                    return VIP2Node(
                        env,
                        node_id,
                        kwargs["num_objects"],
                        kwargs["pen_weight"],
                        **kwargs["vip_args"],
                    )
                case "vipsbw":
                    return VIPSBWNode(
                        env,
                        node_id,
                        kwargs["num_objects"],
                        kwargs["pen_weight"],
                        **kwargs["vip_args"],
                    )
                case "vipsbw2":
                    return VIPSBW2Node(
                        env,
                        node_id,
                        kwargs["num_objects"],
                        kwargs["pen_weight"],
                        **kwargs["vip_args"],
                    )
                case "mvip":
                    return MVIPNode(
                        env,
                        node_id,
                        kwargs["num_objects"],
                        kwargs["pen_weight"],
                        **kwargs["vip_args"],
                    )
                case "mvipsbw":
                    return MVIPSBWNode(
                        env,
                        node_id,
                        kwargs["num_objects"],
                        kwargs["pen_weight"],
                        **kwargs["vip_args"],
                    )
        case _, _:
            print("fwd_pol: {}, cache_pol: {}".format(fwd_pol, cache_pol))
            raise ValueError("No node with the specified policies exist.")


# Filter to eliminate certain combination of parameters that are not meaningful
def ignoreDudFilter(params):
    # Rule 1: VIP type fwd/cache policies should only match with the same type
    # of fwd/cache policies, except for the "none" cache policy which can be matched
    # with any fwd policy.
    if (
        params["cache_pol"] in ["vip", "vip2", "vipsbw", "vipsbw2", "mvip", "mvipsbw"]
        and params["fwd_pol"] != "vip"
    ) or (
        params["cache_pol"] in ["lru", "lfu", "wlfu", "unif", "fifo", "palfu", "pawlfu"]
        and params["fwd_pol"] not in ["sp", "rr", "lrt"]
    ):
        return False

    # Rule 2: Cache parameter arrays should have the same shapes
    cache_params = [
        params["cache_capacities"],
        params["cache_read_rates"],
        params["cache_write_rates"],
        params["cache_read_pens"],
        params["cache_write_pens"],
    ]
    cache_param_shapes = list(map(np.shape, cache_params))
    if len(set(cache_param_shapes)) != 1:
        return False

    # Rule 3: The "none" cache policy should only be matched with empty cache
    # parameters (and vice versa)
    if (
        np.shape(params["cache_capacities"]) == (0,) and params["cache_pol"] != "none"
    ) or (
        np.shape(params["cache_capacities"]) != (0,) and params["cache_pol"] == "none"
    ):
        return False

    # Rule 4: Single tier VIP type cache policies should only match with
    # cache parameters that express a single cache tier.
    if params["cache_pol"] in ["vip", "vip2", "vipsbw", "vipsbw2"] and np.shape(
        params["cache_capacities"]
    ) != (1,):
        return False

    # Rule 5: Cache write rates should not exceed cache read rates
    if any(
        [
            params["cache_write_rates"][i] > params["cache_read_rates"][i]
            for i in range(len(params["cache_write_rates"]))
        ]
    ):
        return False

    # If no rules are violated, return True
    return True


def probDistGenerator(num_objects, dist_param=0.75, dist_type="zipf"):
    if "zipf" in dist_type:
        tot = sum([1/(k**dist_param) for k in range(1, num_objects+1)])
        dist = [1/(k**dist_param)/tot for k in range(1, num_objects+1)]
        return dist

def offlineRequestGenerator(
    nodes, num_objects, seed, stop_time, rate, dist_param, dist_type
):
    rng = default_rng(seed)
    prob_dist = probDistGenerator(num_objects, dist_param, dist_type)
    reqs = {}
    for node_id in nodes:
        intervals = []
        objects = []
        shuffle_interval = 1
        while sum(intervals) < stop_time:
            intervals.append(rng.exponential(1 / rate))
            objects.append(rng.choice(arange(num_objects), p=prob_dist))
            if "shuffle" in dist_type and sum(intervals) > shuffle_interval * (
                stop_time / 4
            ):
                rng.shuffle(prob_dist)
                shuffle_interval += 1
        reqs[node_id] = {"intervals": intervals, "objects": objects}
    return reqs


def assignRouting(G, node_ids, source_map):
    fibs = {}
    for node_id in node_ids:
        fibs[node_id] = defaultdict(list)
        for source_node_id, objects in source_map.items():
            if objects and (node_id != source_node_id):
                paths = [p for p in nx.all_shortest_paths(G, node_id, source_node_id)]
                next_hops = [p[1] for p in paths]
                unique_next_hops = list(set(next_hops))
                for object_id in objects:
                    fibs[node_id][object_id] = [hop_id for hop_id in unique_next_hops]
    return fibs


def assignSources(seed, nodes, num_objects):
    rng = default_rng(seed)
    source_map_inv = rng.choice(nodes, size=num_objects, replace=True)
    source_map = {}
    for node_id in nodes:
        source_map[node_id] = [
            object_id for object_id in range(num_objects) if source_map_inv[object_id] == node_id
        ]

    return source_map
