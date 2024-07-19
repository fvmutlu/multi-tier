# External package imports
import numpy as np

# Builtin imports
import json
from typing import List, Tuple, Callable, Any, Union
from copy import copy, deepcopy
from itertools import product
from os.path import isfile
from urllib.request import urlopen
from dataclasses import dataclass
from hashlib import sha256

# Internal imports
from simulator.helpers import SimulationParameters, simConfigToParamSets
from simulator.topologies import topologies, getRandomTopology

@dataclass(eq=True, frozen=True)
class LegacySimulationParameters:
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

def dictsToParamSets(param_dicts: List[dict], legacy: bool = False) -> List[SimulationParameters] | List[LegacySimulationParameters]:
    for param_set in param_dicts:
        param_set["cache_capacities"] = tuple(param_set["cache_capacities"])
        param_set["cache_read_rates"] = tuple(param_set["cache_read_rates"])
        param_set["cache_write_rates"] = tuple(param_set["cache_write_rates"])
        param_set["cache_read_pens"] = tuple(param_set["cache_read_pens"])
        param_set["cache_write_pens"] = tuple(param_set["cache_write_pens"])
    if legacy:
        return [LegacySimulationParameters(**param_set) for param_set in param_dicts]
    else:
        return [SimulationParameters(**param_set) for param_set in param_dicts]


def getTestConfig(config_path: str):
    if isfile(config_path):
        with open(config_path, "r") as f:
            test_config = json.loads(f.read())
    else:
        response = urlopen(config_path)
        test_config = json.loads(response.read())
    return test_config


def getJsonDb(db_filepath: str) -> dict:
    db_file = open(db_filepath, "r")
    db = json.loads(db_file.read())
    return db


def filterParamList(
    input_param_list: List[SimulationParameters] | List[LegacySimulationParameters],
    filters: List[Tuple[str, Any, Callable]],
):
    param_list = deepcopy(input_param_list)
    for filter_key, filter_value, filter_func in filters:
        if filter_func:
            if isinstance(filter_value, list):
                param_list = [
                    params
                    for params in param_list
                    if filter_func(params[filter_key]) in filter_value
                ]
            else:
                param_list = [
                    params
                    for params in param_list
                    if filter_func(params[filter_key]) == filter_value
                ]
        else:
            if isinstance(filter_value, list):
                param_list = [
                    params for params in param_list if params[filter_key] in filter_value
                ]
            else:
                param_list = [
                    params for params in param_list if params[filter_key] == filter_value
                ]
    return param_list


def getParamHashList(param_list: List[SimulationParameters] | List[LegacySimulationParameters]):
    return [str(hash(sp)) for sp in param_list]


def singleEntrySumDataFieldAcrossNodes(top_name: str, db_entry: dict, field: str):
    if "num_nodes" in topologies[top_name].keys():
        num_nodes = topologies[top_name]["num_nodes"]
    elif "num_nodes" in topologies[top_name]["top_args"].keys():
        num_nodes = topologies[top_name]["top_args"]["num_nodes"]
    else:
        tmp_G = getRandomTopology(top_name, **topologies[top_name]["top_args"])
        num_nodes = len(tmp_G.nodes)
    if isinstance(db_entry["data"]["0"][field], (int, float)):
        return sum([db_entry["data"][str(node)][field] for node in range(num_nodes)])
    elif isinstance(db_entry["data"]["0"][field], (list, tuple)):
        return np.sum(
            [db_entry["data"][str(node)][field] for node in range(num_nodes)],
            axis=0,
        )


def getDataFieldSumsAcrossEntries(
    top_name: str, db: dict, entry_hashes: List[str], field: str
):
    return np.array(
        [
            singleEntrySumDataFieldAcrossNodes(top_name, db[hash], field)
            for hash in entry_hashes
        ]
    )


def getDataFieldSumAvgsAcrossSeeds(
    top_name: str,
    db: dict,
    param_list: List[SimulationParameters] | List[LegacySimulationParameters],
    source_map_seeds: List[int],
    request_generator_seeds: List[int],
    field: str,
):
    num_runs = len(source_map_seeds) * len(request_generator_seeds)
    accumulator = None
    for source_map_seed in source_map_seeds:
        for request_generator_seed in request_generator_seeds:
            run_filters = [
                ("source_map_seed", source_map_seed, None),
                ("request_generator_seed", request_generator_seed, None),
            ]
            run_param_list = filterParamList(param_list, run_filters)
            run_param_hashes = getParamHashList(run_param_list)
            if accumulator is None:
                accumulator = getDataFieldSumsAcrossEntries(
                    top_name, db, run_param_hashes, field
                )
            else:
                accumulator = np.add(
                    accumulator,
                    getDataFieldSumsAcrossEntries(
                        top_name, db, run_param_hashes, field
                    ),
                )
        
    return accumulator / num_runs


def getCustomParamList(
    base: SimulationParameters, variant_key: str, variant_list: List
):
    sim_param_list = []
    for var in variant_list:
        sim_param_list.append(
            SimulationParameters(
                fwd_pol=var if variant_key == "fwd_pol" else base.fwd_pol,
                cache_pol=var if variant_key == "cache_pol" else base.cache_pol,
                num_objects=var if variant_key == "num_objects" else base.num_objects,
                source_read_rate=(
                    var if variant_key == "source_read_rate" else base.source_read_rate
                ),
                source_map_seed=(
                    var if variant_key == "source_map_seed" else base.source_map_seed
                ),
                request_generator_seed=(
                    var
                    if variant_key == "request_generator_seed"
                    else base.request_generator_seed
                ),
                stop_time=var if variant_key == "stop_time" else base.stop_time,
                request_rate=(
                    var if variant_key == "request_rate" else base.request_rate
                ),
                request_dist_param=(
                    var
                    if variant_key == "request_dist_param"
                    else base.request_dist_param
                ),
                request_dist_type=(
                    var
                    if variant_key == "request_dist_type"
                    else base.request_dist_type
                ),
                pen_weight=var if variant_key == "pen_weight" else base.pen_weight,
                vip_inc=var if variant_key == "vip_inc" else base.vip_inc,
                vip_slot_len=(
                    var if variant_key == "vip_slot_len" else base.vip_slot_len
                ),
                vip_win_size=(
                    var if variant_key == "vip_win_size" else base.vip_win_size
                ),
                cache_capacities=(
                    var
                    if variant_key == "cache_capacities"
                    else copy(base.cache_capacities)
                ),
                cache_read_rates=(
                    var
                    if variant_key == "cache_read_rates"
                    else copy(base.cache_read_rates)
                ),
                cache_write_rates=(
                    var
                    if variant_key == "cache_write_rates"
                    else copy(base.cache_write_rates)
                ),
                cache_read_pens=(
                    var
                    if variant_key == "cache_read_pens"
                    else copy(base.cache_read_pens)
                ),
                cache_write_pens=(
                    var
                    if variant_key == "cache_write_pens"
                    else copy(base.cache_write_pens)
                ),
            )
        )
    return sim_param_list
