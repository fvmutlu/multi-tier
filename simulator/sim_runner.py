# External package imports
import numpy as np
import simpy as sp
import joblib as jb
import networkx as nx

# Builtin imports
import argparse
import json
import logging
import logging.config
from itertools import product
from datetime import datetime
from time import process_time
from urllib.request import urlopen
from os.path import isfile

# Internal imports
from .topologies import topologies, getRandomTopology
from .network import getNetwork
from .helpers import (
    assignSources,
    assignRouting,
    offlineRequestGenerator,
    simConfigToParamSets
)
from .utils import NpEncoder, namedZip, timeDiffPrinter
from .logutils import LOGGING_CONFIG, rootlogger

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment_name", type=str, default="sample")
parser.add_argument("-t", "--topology", type=str, default="abilene")
parser.add_argument("-c", "--num_cpus", type=int, default=-2)
parser.add_argument("--logging", action=argparse.BooleanOptionalAction)
parser.add_argument("--profiling", action=argparse.BooleanOptionalAction)
config_mutex_group = parser.add_mutually_exclusive_group()
config_mutex_group.add_argument("--config_local", type=str, dest="config_path")
config_mutex_group.add_argument("--config_url", type=str, dest="config_path")
parser.set_defaults(config_path="./sim_configs/sample_config.json")
args = parser.parse_args()

logger = rootlogger
if isfile(args.config_path):
    with open(args.config_path, "r") as f:
        test_config = json.loads(f.read())
else:
    response = urlopen(args.config_path)
    test_config = json.loads(response.read())

logger.info("Config read, setting up simulation.")

top_params = topologies[args.topology]
if top_params["fixed"]:
    num_nodes = top_params["num_nodes"]
    adj_mat = top_params["adjacency_matrix"]
    top_graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
else:
    top_graph = getRandomTopology(args.topology, **top_params["top_args"])
    adj_mat = nx.to_numpy_array(top_graph, dtype=int)
    num_nodes = adj_mat.shape[0]
    top_graph = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)

nodes = list(range(num_nodes))

# Set source nodes
if top_params["source_nodes"]:
    sources = top_params["source_nodes"]
else:
    sources = nodes

# Set caching nodes
if top_params["cache_nodes"]:
    cache_nodes = top_params["cache_nodes"]
else:
    cache_nodes = nodes

# Set requester nodes
if top_params["requester_nodes"]:
    requester_nodes = top_params["requester_nodes"]
else:
    requester_nodes = nodes

# Set link capacities
if isinstance(top_params["link_caps"], (int, float)):
    link_caps = np.empty_like(adj_mat)
    link_caps[np.where(adj_mat != 0)] = top_params["link_caps"]
else:
    link_caps = top_params["link_caps"]

# Build links
links = []
for v, u in product(nodes, nodes):
    if adj_mat[v][u] != 0:
        link = {
            "edge": (v, u),
            "cap": link_caps[v][u],
            "prop_delay": 0,
            "ctrl_args": {},
        }
        links.append(link)

param_set = simConfigToParamSets(test_config)

logger.info("Setup complete, generating requests.")
reqgen_begin_time = datetime.now()

source_map_dict = {}
for params in param_set:
    source_map_key = (params.num_objects, params.source_map_seed)
    if source_map_key in source_map_dict:
        continue
    source_map_dict[source_map_key] = assignSources(
        params.source_map_seed, sources, params.num_objects
    )

requests_dict = {}
for params in param_set:
    requests_key = (
        params.num_objects,
        params.request_generator_seed,
        params.stop_time,
        params.request_rate,
        params.request_dist_param,
        params.request_dist_type,
    )
    if requests_key in requests_dict:
        continue
    requests_dict[requests_key] = offlineRequestGenerator(
        requester_nodes, *requests_key
    )

reqgen_end_time = datetime.now()
logger.info(
    "Request generation complete, elapsed time: {:s}".format(
        timeDiffPrinter(reqgen_end_time - reqgen_begin_time)
    )
)


def simRun(
    fwd_pol="none",
    cache_pol="none",
    num_objects=1000,
    source_read_rate=1000,
    source_map_seed=1,
    request_generator_seed=1,
    stop_time=100,
    request_rate=10,
    request_dist_param=0.75,
    request_dist_type="zipf",
    pen_weight=0,
    vip_inc=1,
    vip_slot_len=1,
    vip_win_size=100,
    cache_capacities=[5, 100],
    cache_read_rates=[25, 10],
    cache_write_rates=[25, 5],
    cache_read_pens=[2, 1],
    cache_write_pens=[4, 2],
):
    start_time = process_time()

    env = sp.Environment()

    network = getNetwork(env)

    additional_arguments = {
        "num_objects": num_objects,
        "pen_weight": pen_weight,
        "vip_args": {
            "vip_inc": vip_inc,
            "vip_slot_len": vip_slot_len,
            "vip_win_size": vip_win_size,
        },
    }
    network.installNodes(num_nodes, fwd_pol, cache_pol, **additional_arguments)

    network.installLinks(links, fwd_pol, cache_pol)

    source_map = source_map_dict[(num_objects, source_map_seed)]
    network.installSources(sources, source_read_rate, source_map)

    cache_dict = {
        "cap": cache_capacities,
        "read_rate": cache_read_rates,
        "write_rate": cache_write_rates,
        "read_pen": cache_read_pens,
        "write_pen": cache_write_pens,
    }
    caches = [i for i in namedZip(**cache_dict)]
    network.installCaches(cache_nodes, caches)

    fibs = assignRouting(top_graph, nodes, source_map)
    network.installFIBs(fibs)

    requests = requests_dict[
        (
            num_objects,
            request_generator_seed,
            stop_time,
            request_rate,
            request_dist_param,
            request_dist_type,
        )
    ]
    network.initRequests(requester_nodes, requests)

    termination_watcher = env.process(network.terminateSim(requester_nodes))
    env.run(until=termination_watcher)

    elapsed_time = process_time() - start_time

    return network.getStats(), elapsed_time


logger.info("Starting simulation run.")
test_begin_time = datetime.now()

if args.profiling:
    results = []
    for params in param_set:
        results.append(simRun(**params))
else:
    results = jb.Parallel(n_jobs=args.num_cpus)(
        jb.delayed(simRun)(**params) for params in param_set
    )

test_end_time = datetime.now()
logger.info(
    "Simulation run complete, elapsed time: {:s}".format(
        timeDiffPrinter(test_end_time - test_begin_time)
    )
)

data_collection = {}
for i, params in enumerate(param_set):  # Per param set
    param_hash = hash(params)
    data_collection[param_hash] = {
        "parameters": {**params},
        "cpu_time": results[i][1],
        "data": {**results[i][0]},
    }

json_file_path = (
    "./sim_outputs/" + args.experiment_name + "_" + args.topology + "_db.json"
)
with open(json_file_path, "w") as json_file:
    json.dump(data_collection, json_file, cls=NpEncoder)
logger.info("Data dumped to JSON file: {}".format(json_file_path))
