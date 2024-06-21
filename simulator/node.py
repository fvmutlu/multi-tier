# External package imports
import simpy as sp

# Builtin imports
from collections import namedtuple, defaultdict
from copy import deepcopy

# Internal imports
from .components import Request, InterestPacket, DataPacket
from .cache import Cache, Permastore
from .link import Link

Packet = namedtuple("Packet", ["type", "object_id"])

class Node(object):
    """
    Template for a barebones node object.
    This class defines commons functions that should be used by all nodes.
    However, it makes no assumptions for forwarding or caching strategies.
    Those functions must be implemented by inheriting classes that define custom forwarding and caching.
    """

    def __init__(self, env: sp.Environment, node_id: int):
        # SimPy environment
        self.env = env

        # Node properties
        self.id = node_id
        self.is_source = False
        self.is_requester = False
        self.has_caches = False

        # Packet rx/tx
        self.out_links = {}
        self.in_links = {}
        self.pkt_buffer = sp.Store(self.env)

        # Out-of-band control
        self.ctrl_out_links = {}
        self.ctrl_in_links = {}

        # Forwarding Information Base
        self.fib = {}

        # Pending Interest Table
        self.pit = defaultdict(dict)

        # Termination events (connected to exogenous request generation)
        self.exo_gen_done = False
        self.can_terminate = env.event()

        # Statistics tracking

        # Logging
        self.stats = {
            "delay": 0,
            "rx_interests": 0,
            "gen_reqs": 0,
            "done_reqs": 0,
            "source_hits": 0,
            "source_hit_delay": 0,
            "cache_hits": [],
            "cache_writes": [],
            "cache_replacements": [],
            "cache_hit_delays": [],
            "cache_read_penalties": [],
            "cache_write_penalties": [],
            "tx_interests": {},
        }

    def addOutputLink(self, remote_id: int, link: Link, ctrl_link: Link):
        """
        Add an output link to the node.
        """
        self.out_links[remote_id] = link
        self.ctrl_out_links[remote_id] = ctrl_link
        self.stats["tx_interests"][remote_id] = 0

    def addInputLink(self, remote_id, link, ctrl_link):
        """
        Add an input link to the node.
        Initialize the packetReceiver process for the added input link.
        """
        self.in_links[remote_id] = link
        self.ctrl_in_links[remote_id] = ctrl_link
        self.env.process(self.packetReceiver(remote_id))

    def addPermastore(self, permastore):
        """
        Add a Permastore instance to the node (a node should have at most 1 Permastore instance added to it).
        Set the is_source attribute to True (a node with a Permastore is a content source).
        Initialize the source_hits instance attribute with 0.
        Also initialize the permastoreController process for the adde Permastore instance.
        """
        self.permastore = permastore
        self.is_source = True
        self.env.process(permastore.permastoreController())

    def addCache(self, cache):
        """
        Add a Cache instance to the node.
        If no Cache instance was present previously, create the instance attribute caches (List[Cache]).
        If one or more Cache instances were present previously, insert the new instance in the caches list at a specific index based on its readout rate.
        Initialize the cacheController process for the Cache instance added.
        """
        if self.has_caches:
            for j, existing_cache in enumerate(self.caches):
                if cache.read_rate >= existing_cache.read_rate:
                    self.caches.insert(j, cache)
                    break
            if j == len(self.caches) - 1:
                self.caches.append(cache)
        else:
            self.caches = [cache]
            self.has_caches = True
        self.env.process(cache.cacheController())

    def addFIB(self, fib):
        self.fib = fib

    def packetReceiver(self, remote_id):
        """
        SimPy process that receives packets on a single link and directs them to be processed.
        Blocks until a packet is put into the Link instance connecting current node to the specified remote node.
        When a packet is received, if it is a data packet, transmission delay is enforced on the link.
        Transmission delays are ignored for interest packet since their sizes are negligible.
        """
        while True:
            pkt = yield self.in_links[remote_id].get()
            if pkt.isData():
                yield self.env.timeout(1 / self.in_links[remote_id].link_cap)
            if pkt.isData() or pkt.isInterest():
                self.pkt_buffer.put((remote_id, pkt))

    def packetProcessor(self):
        """
        SimPy process that handles processing of packets received by all input links.
        Retrieves packets placed into the processing buffer by receiver processes and initiates processes that invoke forwarding/caching strategies
        based on the type of packets.
        """
        while True:
            remote_id, pkt = yield self.pkt_buffer.get()
            if pkt.isInterest():
                self.receiveInterest(remote_id, pkt.request)
            elif pkt.isData():
                self.receiveData(pkt.request)

    def receiveInterest(self, remote_id, request):
        request.logEvent(self.id)
        self.stats["rx_interests"] += 1
        object_id = request.object_id
        object_location = self.locateObject(object_id)

        self.pit[object_id][request.origin_id, request.seq_id] = remote_id
        if object_location == -2:
            self.forwardInterest(request)
        else:
            self.env.process(self.respondWithLocal(request, object_location))

    def respondWithLocal(self, request, object_location):
        print(f"Node {self.id} responding to request for src:{request.origin_id},obj:{request.object_id},seq:{request.seq_id}")
        object_id = request.object_id
        if object_location == -1:
            yield self.env.process(self.permastore.readObject(object_id))
        else:
            yield self.env.process(self.caches[object_location].readObject(object_id))

        self.receiveData(request)

    def receiveData(self, request):
        """
        Satisfy a locally generated exogenous interest with arriving data, or forward data to remote node with matching PIT entry.
        """
        object_id = request.object_id
        request.logEvent(self.id)
        remote_id = self.pit[object_id].pop((request.origin_id, request.seq_id))
        if remote_id != self.id:
            self.sendDataPacket(remote_id, request)
        else:
            self.requests.remove(request.seq_id)
            self.stats["done_reqs"] += 1
            self.stats["delay"] += request.getDelay()
            del request

        if self.is_requester:
            if self.req_gen_done and not self.requests:
                if not (self.can_terminate.triggered or self.can_terminate.processed):
                    self.can_terminate.succeed()

        if self.has_caches and self.locateObject(object_id) == -2:
            self.env.process(self.decideCaching(object_id))

    def sendInterestPacket(self, remote_id, request):
        """
        Send an interest packet to a remote node.
        """
        pkt = InterestPacket(request)
        self.out_links[remote_id].put(pkt)
        self.stats["tx_interests"][remote_id] += 1

    def sendDataPacket(self, remote_id, request):
        """
        Send a data packet to a remote node.
        """
        pkt = DataPacket(request)
        self.out_links[remote_id].put(pkt)

    def requestGenerator(self, intervals, object_ids):
        """
        SimPy process that generates exogenous requests for data objects.
        """
        self.is_requester = True
        self.req_gen_done = False
        self.requests = []
        self.request_counter = 0
        for interval, object_id in zip(intervals, object_ids):
            yield self.env.timeout(interval)
            req_seq = self.request_counter + 1
            self.request_counter += 1
            self.requests.append(req_seq)
            req = Request(self.env, self.id, req_seq, object_id)
            self.stats["gen_reqs"] += 1
            self.receiveInterest(self.id, req)
        self.req_gen_done = True

    def locateObject(self, object_id):
        """
        Find whether an object is being cached/sourced at the node, and where if so.
        """
        if self.is_source and self.permastore.isSourced(object_id):
            return -1
        elif self.has_caches:
            for j, cache in enumerate(self.caches):
                if cache.isCached(object_id):
                    return j
        return -2

    def forwardInterest(self, request):
        """
        Forward an interest based on the given object ID and forwarding strategy.
        This function must be implemented by an inheriting class.
        """
        object_id = request.object_id
        self.sendInterestPacket(self.fib[object_id][0], request)
        return

    def decideCaching(self, object_id):
        """
        Make a caching decision based on the given object ID and caching strategy.
        This function must be implemented by an inheriting class to implement custom caching policy.
        """
        yield self.env.timeout(0)
        return

    def getStats(self):
        if self.is_source:
            source_stats = self.permastore.getStats()
            self.stats["source_hits"] = source_stats["reads"]
            self.stats["source_hit_delay"] = source_stats["read_delay"]
        if self.has_caches:
            self.stats["cache_hits"] = []
            self.stats["cache_writes"] = []
            self.stats["cache_replacements"] = []
            self.stats["cache_hit_delays"] = []
            self.stats["cache_read_penalties"] = []
            self.stats["cache_write_penalties"] = []
            for cache in self.caches:
                cache_stats = cache.getStats()
                read_pen = cache.read_penalty * cache_stats["replacements"]
                write_pen = cache.write_penalty * (
                    cache_stats["writes"] + cache_stats["replacements"]
                )
                self.stats["cache_hits"].append(cache_stats["reads"])
                self.stats["cache_writes"].append(cache_stats["writes"])
                self.stats["cache_replacements"].append(cache_stats["replacements"])
                self.stats["cache_hit_delays"].append(cache_stats["read_delay"])
                self.stats["cache_read_penalties"].append(read_pen)
                self.stats["cache_write_penalties"].append(write_pen)
        return self.stats


class IANode(Node):
    def __init__(self, env, node_id):
        super().__init__(env, node_id)
        self.pit = defaultdict(list)

    def receiveInterest(self, remote_id, request):
        request.logEvent(self.id)
        self.stats["rx_interests"] += 1
        object_id = request.object_id
        object_location = self.locateObject(object_id)

        if self.pit[object_id]:
            self.pit[object_id].append((remote_id, request))
        else:
            if object_location == -2:
                self.forwardInterest(request)
            else:
                self.env.process(self.respondWithLocal(request, object_location))
            self.pit[object_id].append((remote_id, request))

    def receiveData(self, request):
        """
        Satisfy a locally generated exogenous interest with arriving data, or forward data to remote node with matching PIT entry.
        """
        object_id = request.object_id

        if object_id in self.pit.keys() and self.pit[object_id]:
            while self.pit[object_id]:
                remote_id, pending_req = self.pit[object_id].pop(0)
                pending_req.logEvent(self.id)
                if remote_id != self.id:
                    self.sendDataPacket(remote_id, pending_req)
                else:
                    self.requests.remove(pending_req.seq_id)
                    self.stats["done_reqs"] += 1
                    self.stats["delay"] += pending_req.getDelay()
                    del pending_req

        if self.is_requester:
            if self.req_gen_done and not self.requests:
                if not (self.can_terminate.triggered or self.can_terminate.processed):
                    self.can_terminate.succeed()

        if self.has_caches and self.locateObject(object_id) == -2:
            self.env.process(self.decideCaching(object_id))
