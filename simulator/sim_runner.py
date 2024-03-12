import argparse
import numpy as np
import simpy as sp
import joblib as jb
import networkx as nx

from urllib.request import urlopen
import json

from tinydb import TinyDB
from itertools import product
from datetime import datetime

from topologies import topologies, getRandomTopology
from network import getNetwork
from helpers import assignSources, assignRouting, offlineRequestGenerator, ignoreDudFilter
from utils import namedProduct, namedZip

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--test_name', type = str)
parser.add_argument('-u', '--test_url', type = str)
parser.add_argument('-t', '--topology', type = str)
parser.add_argument('-d', '--database_name', type = str)
parser.add_argument('-c', '--num_cpus', type = int)
args = parser.parse_args()

response = urlopen(args.test_url)
test_config = json.loads(response.read())
print("Config read, setting up for simulation...")

top_params = topologies[args.topology]
if top_params['fixed']:
    num_nodes = top_params['num_nodes']
    adj_mat = top_params['adjacency_matrix']
    top_graph = nx.from_numpy_array(adj_mat, create_using = nx.DiGraph)
else:
    top_graph = getRandomTopology(args.topology, **top_params['top_args'])
    adj_mat = nx.to_numpy_array(top_graph, dtype=int)
    num_nodes = adj_mat.shape[0]
    top_graph = nx.from_numpy_array(adj_mat, create_using = nx.DiGraph)
    
nodes = list(range(num_nodes))

if top_params['source_nodes']:
    sources = top_params['source_nodes']
else:
    sources = nodes

if top_params['cache_nodes']:
    cache_nodes = top_params['cache_nodes']
else:
    cache_nodes = nodes

if top_params['requester_nodes']:
    requester_nodes = top_params['requester_nodes']
else:
    requester_nodes = nodes

if isinstance(top_params['link_caps'], (int,float)):
    link_caps = np.empty_like(adj_mat)
    link_caps[np.where(adj_mat != 0)] = top_params['link_caps']
else:
    link_caps = top_params['link_caps']

links = []
for v, u in product(nodes,nodes):
    if adj_mat[v][u] != 0:
        link = {'edge': (v, u), 'cap': link_caps[v][u], 'prop_delay': 0, 'ctrl_args': {}}
        links.append(link)

logging_interval = test_config['logging_interval']
sim_params = test_config['sim_params']
param_set = [params for params in namedProduct(**sim_params)]
param_set = [params for params in filter(ignoreDudFilter, param_set)]

requests_dict = {}
print("Generating requests...")
reqgen_begin_time = datetime.now()
for params in param_set:
    requests_key = (params['num_objects'], params['request_generator_seed'], params['stop_time'], params['request_rate'], params['request_dist_param'], params['request_dist_type'])
    if requests_key in requests_dict:
        continue
    requests_dict[requests_key] = offlineRequestGenerator(requester_nodes, *requests_key)
reqgen_end_time = datetime.now()
print("Request generation complete, elapsed time: {:s}".format(str(reqgen_end_time - reqgen_begin_time)))

def simRun(
    fwd_pol = 'none',
    cache_pol = 'none',
    num_objects = 1000,
    source_read_rate = 1000,
    source_map_seed = 1,
    request_generator_seed = 1,
    stop_time = 100,
    request_rate = 10,
    request_dist_param = 0.75,
    request_dist_type = 'zipf',
    pen_weight = 0,
    vip_inc = 1,
    vip_slot_len = 1,
    vip_win_size = 30,
    cache_capacities  = [10, 100],
    cache_read_rates  = [20, 10],
    cache_write_rates = [20, 10],
    cache_read_pens   = [2, 1],
    cache_write_pens  = [4, 2]
    ):
    env = sp.Environment()
    objects = list(range(num_objects))

    network = getNetwork(env)

    additional_arguments = {'num_objects': num_objects, 'pen_weight': pen_weight, 'vip_args': {'vip_inc': vip_inc, 'vip_slot_len': vip_slot_len, 'vip_win_size': vip_win_size}}
    network.installNodes(num_nodes, fwd_pol, cache_pol, **additional_arguments)

    network.installLinks(links, fwd_pol, cache_pol)

    source_map = assignSources(source_map_seed, sources, objects)
    network.installSources(sources, source_read_rate, source_map)

    cache_dict = {'cap': cache_capacities, 'read_rate': cache_read_rates, 'write_rate': cache_write_rates, 'read_pen': cache_read_pens, 'write_pen': cache_write_pens}
    caches = [i for i in namedZip(**cache_dict)]
    network.installCaches(cache_nodes, caches)

    fibs = assignRouting(top_graph, nodes, source_map)
    network.installFIBs(fibs)

    requests = requests_dict[(num_objects, request_generator_seed, stop_time, request_rate, request_dist_param, request_dist_type)]
    network.initRequests(requester_nodes, requests)

    env.process(network.statLogger(logging_interval))

    termination_watcher = env.process(network.terminateSim(requester_nodes))
    env.run(until = termination_watcher)
    return network.getStats()


print("Starting simulation...")
test_begin_time = datetime.now()
## USE THIS NON-PARALLELIZED LOOP FOR PROFILING PURPOSES (COMMENT OUT BELOW Parallel JOB FIRST)
#results = []
#for params in param_set:
#    results.append(simRun(**params))
## USE THIS Parallel JOB FOR FASTER SIMULATIONS (COMMENT OUT ABOVE LOOP FIRST)
results = jb.Parallel(n_jobs = args.num_cpus)(jb.delayed(simRun)(**params) for params in param_set)
test_end_time = datetime.now()
print("Simulation ended, elapsed time: {:s}".format(str(test_end_time - test_begin_time)))

print("Inserting data into DB...")
tic = datetime.now()

test_db = TinyDB('./test_outputs/test_configs.json')
test_db.insert({'test': args.test_name, 'time': str(test_begin_time), 'topology': args.topology, 'elapsed': str(test_end_time - test_begin_time), **test_config})

data_rows = []
for i, params in enumerate(param_set): # Per param set
    for res in results[i]: # Per log interval
        data_rows.append({**params, **res})
        
data_db = TinyDB('./test_outputs/' + args.database_name + '.json')
data_db.insert({'test': args.test_name, 'time': str(test_begin_time), 'topology': args.topology, 'elapsed': str(test_end_time - test_begin_time), 'data': data_rows})

toc = datetime.now()
print("Insertion done in {:s}, exiting.".format(str(toc-tic)))