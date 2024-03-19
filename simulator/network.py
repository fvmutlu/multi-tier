# External package imports
import simpy as sp

# Builtin imports
from copy import deepcopy

# Internal imports
from .link import getLink, getCtrlLink
from .cache import Permastore, Cache
from .helpers import getNode

class Network(object):
    def __init__(self, env):
        self.env = env
        self.nodes = []

        # Statistics
        self.stats = []
    
    def installNodes(self, num_nodes, fwd_pol, cache_pol, **kwargs):
        for node_id in range(num_nodes):
            node = getNode(self.env, node_id, fwd_pol, cache_pol, **kwargs)
            self.nodes.append(node)
            self.env.process(node.packetProcessor())

    def installLinks(self, links, fwd_pol, cache_pol):
        for link in links:
            v, u = link['edge']
            link_vu = getLink(self.env, link['cap'], link['prop_delay'])
            ctrl_link_vu = getCtrlLink(self.env, fwd_pol, cache_pol, **link['ctrl_args'])
            self.nodes[v].addOutputLink(u, link_vu, ctrl_link_vu)
            self.nodes[u].addInputLink(v, link_vu, ctrl_link_vu)

    def installSources(self, source_nodes, source_read_rate, source_map):
        for node_id in source_nodes:
            self.nodes[node_id].addPermastore(Permastore(self.env, source_read_rate, source_map[node_id]))
    
    def installCaches(self, cache_nodes, caches):
        for node_id in cache_nodes:
            for cache in caches:
                self.nodes[node_id].addCache(Cache(self.env, **cache))

    def installFIBs(self, fibs, dist_diffs=None):
        for node_id, node in enumerate(self.nodes):
            node.addFIB(fibs[node_id],None)
    
    def initRequests(self, requester_nodes, requests):
        for node_id in requester_nodes:
            self.env.process(self.nodes[node_id].requestGenerator(requests[node_id]['intervals'], requests[node_id]['objects']))

    def terminateSim(self, requester_node_ids):
        node_terminate_events = [self.nodes[node_id].can_terminate for node_id in requester_node_ids]
        yield sp.events.AllOf(self.env, node_terminate_events)
        self.recordStats()
        return

    def recordStats(self):
        time = self.env.now
        for node in self.nodes:
            self.stats.append({'sim_time': time, 'node_id': node.id, **node.getStats()})
    
    def getStats(self):
        return self.stats

def getNetwork(env):
    return Network(env)