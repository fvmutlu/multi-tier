# External package imports

# Builtin imports
from typing import List
from copy import copy

# Internal imports
from simulator.helpers import SimulationParameters


def getParamList(base: SimulationParameters, variant_key: str, variant_list: List):
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


def getParamHashList(param_list: List[SimulationParameters]):
    return [hash(sp) for sp in param_list]
