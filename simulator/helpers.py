# External package imports
from numpy import arange
from numpy.random import default_rng
from scipy.stats import zipfian
import networkx as nx

# Builtin imports
from collections import defaultdict

# Internal imports
from .policies import *

def getNode(env, node_id, fwd_pol, cache_pol, **kwargs):
    if fwd_pol == 'none':
        if cache_pol == 'none':
            return Node(env, node_id)
        elif cache_pol == 'lru':
            return LRUNode(env, node_id, kwargs['num_objects'])
        elif cache_pol == 'lfu':
            return LFUNode(env, node_id, kwargs['num_objects'])
        elif cache_pol == 'wlfu':
            return WLFUNode(env, node_id, kwargs['num_objects'])
        elif cache_pol == 'fifo':
            return FIFONode(env, node_id)
        elif cache_pol == 'unif':
            return UNIFNode(env, node_id)
        elif cache_pol == 'palfu':
            return PALFUNode(env, node_id, kwargs['num_objects'], kwargs['pen_weight'])
    elif fwd_pol == 'rr':
        if cache_pol == 'none':
            return RoundRobinNode(env, node_id)
        elif cache_pol == 'lru':
            return RRLRUNode(env, node_id, kwargs['num_objects'])
        elif cache_pol == 'lfu':
            return RRLFUNode(env, node_id, kwargs['num_objects'])
        elif cache_pol == 'wlfu':
            return RRWLFUNode(env, node_id, kwargs['num_objects'])
        elif cache_pol == 'fifo':
            return RRFIFONode(env, node_id)
        elif cache_pol == 'unif':
            return RRUNIFNode(env, node_id)
        elif cache_pol == 'palfu':
            return RRPALFUNode(env, node_id, kwargs['num_objects'], kwargs['pen_weight'])
    elif fwd_pol == 'lrt':
        if cache_pol == 'none':
            return LeastResponseTimeNode(env, node_id)
        elif cache_pol == 'lru':
            return LRTLRUNode(env, node_id, kwargs['num_objects'])
        elif cache_pol == 'lfu':
            return LRTLFUNode(env, node_id, kwargs['num_objects'])
        elif cache_pol == 'wlfu':
            return LRTWLFUNode(env, node_id, kwargs['num_objects'])
        elif cache_pol == 'fifo':
            return LRTFIFONode(env, node_id)
        elif cache_pol == 'unif':
            return LRTUNIFNode(env, node_id)
        elif cache_pol == 'palfu':
            return LRTPALFUNode(env, node_id, kwargs['num_objects'], kwargs['pen_weight'])
    elif fwd_pol == 'svip' and cache_pol in ['svip','none']:
        return VIPNode(env, node_id, kwargs['num_objects'], kwargs['pen_weight'], **kwargs['vip_args'])
    elif fwd_pol == 'mvip' and cache_pol in ['mvip','none']:
        return MVIPNode(env, node_id, kwargs['num_objects'], kwargs['pen_weight'], **kwargs['vip_args'])
    else:
        print("fwd_pol: {}, cache_pol: {}".format(fwd_pol,cache_pol))
        raise ValueError("No node with the specified policies exist.")

# Filter to eliminate certain combination of parameters that are not meaningful
def ignoreDudFilter(params):
    # Rule 1: VIP type fwd/cache policies should only match with the same type of fwd/cache policies; except the "none" cache policy can be matched with any fwd policy
    if (params['cache_pol'] in ['svip','mvip'] and params['fwd_pol'] != params['cache_pol']) or (params['cache_pol'] in ['lru','lfu','unif','fifo','palfu'] and params['fwd_pol'] not in ['sp','rr','lrt']):
        return False

    # Rule 2: Cache parameter arrays should have the same shapes
    cache_params = [params['cache_capacities'], params['cache_read_rates'], params['cache_write_rates'], params['cache_read_pens'], params['cache_write_pens']]
    cache_param_shapes = list(map(np.shape, cache_params))
    if len(set(cache_param_shapes)) != 1:
        return False
    
    # Rule 3: The "none" cache policy should only be matched with empty cache parameters (and vice versa)
    if (np.shape(params['cache_capacities']) == (0,) and params['cache_pol'] != 'none') or (np.shape(params['cache_capacities']) != (0,) and params['cache_pol'] == 'none'):
        return False
    
    # If no rules are violated, return True
    return True

def probDistGenerator(num_objects, dist_param = 0.75, dist_type = 'zipf'):
    if dist_type == 'zipf':
        return [zipfian.pmf(k, dist_param, num_objects) for k in range(1, num_objects + 1)]

def offlineRequestGenerator(nodes, num_objects, seed, stop_time, rate, dist_param, dist_type):
    rng = default_rng(seed)
    prob_dist = probDistGenerator(num_objects, dist_param, dist_type)
    reqs = {}
    for node_id in nodes:
        intervals = []
        objects = []
        while sum(intervals) < stop_time:
            intervals.append(rng.exponential(1/rate))
            objects.append(rng.choice(arange(num_objects), p = prob_dist))
        reqs[node_id] = {'intervals': intervals, 'objects': objects}
    return reqs

def assignRouting(G, node_ids, source_map):
    fibs = {}
    for node_id in node_ids:
        fibs[node_id] = defaultdict(list)
        for source_node_id, objects in enumerate(source_map):
            if objects and (node_id != source_node_id):
                paths = [p for p in nx.all_shortest_paths(G,node_id,source_node_id)]
                next_hops = [p[1] for p in paths]
                unique_next_hops = list(set(next_hops))
                for object_id in objects:
                    fibs[node_id][object_id] = [hop_id for hop_id in unique_next_hops]
    return fibs

def assignSources(seed, nodes, num_objects):
    rng = default_rng(seed)
    source_map_inv = rng.choice(nodes, size = num_objects, replace = True)
    source_map = [[object_id for object_id in range(num_objects) if source_map_inv[object_id] == node_id] for node_id in nodes]

    return source_map