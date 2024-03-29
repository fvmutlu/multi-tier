# External package imports
import numpy as np
import networkx as nx

# Builtin imports

# Internal imports

topologies = {
    "service": {
        "fixed": True,
        "num_nodes": 8,
        "adjacency_matrix": np.array([[0, 1, 0, 0, 0, 0, 0, 0], 
                                    [1, 0, 1, 0, 0, 0, 0, 0], 
                                    [0, 1, 0, 1, 0, 0, 0, 0], 
                                    [0, 0, 1, 0, 1, 1, 1, 1], 
                                    [0, 0, 0, 1, 0, 0, 0, 0], 
                                    [0, 0, 0, 1, 0, 0, 0, 0], 
                                    [0, 0, 0, 1, 0, 0, 0, 0], 
                                    [0, 0, 0, 1, 0, 0, 0, 0]]),
        "link_caps": np.array([ [0, 40, 0, 0, 0, 0, 0, 0], 
                                [40, 0, 40, 0, 0, 0, 0, 0], 
                                [0, 40, 0, 40, 0, 0, 0, 0], 
                                [0, 0, 40, 0, 10, 10, 10, 10], 
                                [0, 0, 0, 10, 0, 0, 0, 0], 
                                [0, 0, 0, 10, 0, 0, 0, 0], 
                                [0, 0, 0, 10, 0, 0, 0, 0], 
                                [0, 0, 0, 10, 0, 0, 0, 0]]),
        "source_nodes": [0],
        "cache_nodes": [1,2,3],
        "requester_nodes": [4,5,6,7]
    },
    "abilene": {
        "fixed": True,
        "num_nodes": 11,
        "adjacency_matrix": np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], 
                                    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0], 
                                    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0], 
                                    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                                    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], 
                                    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], 
                                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]]),
        "link_caps": 10,
        "source_nodes": [],
        "cache_nodes": [],
        "requester_nodes": []
    },
    "geant": {
        "fixed": True,
        "num_nodes": 34,
        "adjacency_matrix": np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]]),
        "link_caps": 10,
        "source_nodes": [],
        "cache_nodes": [],
        "requester_nodes": []
    },
    "grid": {
        "fixed": False,
        "link_caps": 10,
        "source_nodes": [],
        "cache_nodes": [],
        "requester_nodes": [],
        "top_args": {
            "rows": 4,
            "cols": 4
        }
    },
    "regular": {
        "fixed": False,
        "link_caps": 10,
        "source_nodes": [],
        "cache_nodes": [],
        "requester_nodes": [],
        "top_args": {
            "num_nodes": 50,
            "degree": 3,
            "seed": 1
        }
    },
    "erdos": {
        "fixed": False,
        "link_caps": 10,
        "source_nodes": [],
        "cache_nodes": [],
        "requester_nodes": [],
        "top_args": {
            "num_nodes": 50,
            "p": 0.1,
            "seed": 1
        }
    },
    "watts": {
        "fixed": False,
        "link_caps": 10,
        "source_nodes": [],
        "cache_nodes": [],
        "requester_nodes": [],
        "top_args": {
            "num_nodes": 50,
            "k": 3,
            "p": 0.1,
            "seed": 1
        }
    
    },
    "barabasi": {
        "fixed": False,
        "link_caps": 10,
        "source_nodes": [],
        "cache_nodes": [],
        "requester_nodes": [],
        "top_args": {
            "num_nodes": 50,
            "m": 3,
            "seed": 1
        }
    }
}

def getRandomTopology(top_name, **args):
    if top_name == 'grid':
        return nx.grid_2d_graph(args['rows'], args['cols'])
    if top_name == 'regular':
        return nx.random_regular_graph(args['degree'], args['num_nodes'], args['seed'])
    if top_name == 'erdos':
        return nx.erdos_renyi_graph(args['num_nodes'], args['p'], args['seed'])
    if top_name == 'watts':
        return nx.watts_strogatz_graph(args['num_nodes'], args['k'], args['p'], args['seed'])
    if top_name == 'barabasi':
        return nx.barabasi_albert_graph(args['num_nodes'], args['m'], args['seed'])
