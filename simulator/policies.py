# External package imports

# Builtin imports

# Internal imports

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from .caching_policies import *
from .forwarding_policies import *

from .utils import wique, invertDict, resetDict

class RRLRUNode(RoundRobinNode, LRUNode):
    def __init__(self, env, node_id, num_objects):
        super().__init__(env, node_id)
        self.lruInit(num_objects)

class RRLFUNode(RoundRobinNode, LFUNode):
    def __init__(self, env, node_id, num_objects):
        super().__init__(env, node_id)
        self.lfuInit(num_objects)

class RRWLFUNode(RoundRobinNode, WLFUNode):
    def __init__(self, env, node_id, num_objects):
        super().__init__(env, node_id)
        self.wlfuInit(num_objects)

class RRFIFONode(RoundRobinNode, FIFONode):
    pass

class RRUNIFNode(RoundRobinNode, UNIFNode):
    pass

class RRPALFUNode(RoundRobinNode, PALFUNode):
    def __init__(self, env, node_id, num_objects, pen_weight):
        super().__init__(env, node_id)
        self.lfuInit(num_objects, pen_weight)

class LRTLRUNode(LeastResponseTimeNode, LRUNode):
    def __init__(self, env, node_id, num_objects):
        super().__init__(env, node_id)
        self.lruInit(num_objects)

class LRTLFUNode(LeastResponseTimeNode, LFUNode):
    def __init__(self, env, node_id, num_objects):
        super().__init__(env, node_id)
        self.lfuInit(num_objects)

class LRTWLFUNode(LeastResponseTimeNode, WLFUNode):
    def __init__(self, env, node_id, num_objects):
        super().__init__(env, node_id)
        self.wlfuInit(num_objects)

class LRTFIFONode(LeastResponseTimeNode, FIFONode):
    def __init__(self, env, node_id):
        super().__init__(env, node_id)

class LRTUNIFNode(LeastResponseTimeNode, UNIFNode):
    def __init__(self, env, node_id):
        super().__init__(env, node_id)

class LRTPALFUNode(LeastResponseTimeNode, PALFUNode):
    def __init__(self, env, node_id, num_objects, pen_weight):
        super().__init__(env, node_id)
        self.lfuInit(num_objects, pen_weight)

class VIPNode(Node):
    def __init__(self, env, node_id, num_objects, pen_weight, **vip_args):
        super().__init__(env, node_id)
        self.num_objects = num_objects
        self.pw = pen_weight

        # VIP Parameters
        self.vip_inc = vip_args['vip_inc']
        self.slot_len = vip_args['vip_slot_len']
        self.win_size = vip_args['vip_win_size']        

        # VIP State Variables
        self.vip_counts = [0] * num_objects
        self.vip_rx = [0] * num_objects
        self.vip_rx_windows = [wique(maxlen = self.win_size) for _ in range(self.num_objects)]
        self.cache_scores = [0] * num_objects
        self.neighbor_vip_counts = {}
        self.neighbor_vip_tx = {}
        self.virtual_caches = []
        self.vip_allocs = [[] for _ in range(self.num_objects)]

        # LRT in VIP to break ties
        self.link_delays = {}
        self.req_timestamps = {}

        # Init VIP process
        self.env.process(self.vipProcess())

    def addInputLink(self, remote_id, link, ctrl_link):
        super().addInputLink(remote_id, link, ctrl_link)
        self.neighbor_vip_counts[remote_id] = [0] * self.num_objects
    
    def addOutputLink(self, remote_id, link, ctrl_link):
        super().addOutputLink(remote_id, link, ctrl_link)
        self.link_delays[remote_id] = wique(maxlen = 1)
        self.req_timestamps[remote_id] = {}
    
    def packetProcessor(self):
        while True:
            remote_id, pkt = yield self.pkt_buffer.get()
            if pkt.isInterest():
                self.receiveInterest(remote_id, pkt.request)
            elif pkt.isData():
                delay = self.env.now - self.req_timestamps[remote_id].pop((pkt.request.origin_id, pkt.request.seq_id))
                self.link_delays[remote_id].append(delay)
                self.receiveData(pkt.request)

    def addCache(self, cache):
        super().addCache(cache)
        self.virtual_caches.append([])
    
    def addFIB(self, fib, dist_diff=None):
        super().addFIB(fib, dist_diff)
        self.fib_inv = invertDict(fib)
        for object_id in fib:
            for remote_id in fib[object_id]:
                self.neighbor_vip_tx[remote_id, object_id] = wique(maxlen = self.win_size)

    def receiveInterest(self, remote_id, request):
        if remote_id == self.id:
            self.vip_counts[request.object_id] += self.vip_inc
            self.vip_rx[request.object_id] += self.vip_inc
        super().receiveInterest(remote_id, request)
    
    def forwardInterest(self, request):
        object_id = request.object_id
        neighbors = self.fib[object_id]
        link_profits = np.array([ self.neighbor_vip_tx[remote_id, object_id].mean for remote_id in neighbors ])

        #LRT to break ties
        link_delays = np.array([ self.link_delays[remote_id].mean for remote_id in neighbors ])
        max_profit_link_idx = np.where(link_profits == link_profits.max())[0]
        max_profit_link_delays = np.array([link_delays[idx] for idx in max_profit_link_idx])
        idx = max_profit_link_idx[np.argmin(max_profit_link_delays)]

        remote_id = neighbors[idx]

        ##LRT tracking
        self.req_timestamps[remote_id][(request.origin_id, request.seq_id)] = self.env.now
        ##LRT tracking

        self.sendInterestPacket(remote_id, request)

    def decideCaching(self, object_id):
        benefits = []
        victims = []
        for j, cache in enumerate(self.caches):
            if cache.isFull():
                victim_id = min(cache.contents, key=lambda k: self.cache_scores[k])
                benefits.append(cache.read_rate*(self.cache_scores[object_id] - self.cache_scores[victim_id]) - self.pw * (cache.read_penalty + cache.write_penalty))
                victims.append(victim_id)
            else:
                benefits.append(cache.read_rate*self.cache_scores[object_id] - self.pw * cache.write_penalty)
                victims.append(None)
        j = np.argmax(benefits)
        if benefits[j] > 0:
            if victims[j] is not None:
                yield self.env.process(self.caches[j].replaceObject(victims[j], object_id))
                self.env.process(self.decideCaching(victims[j]))
            else:
                self.caches[j].cacheObject(object_id)

    def decideCaching_OG(self, object_id):
        for cache in self.caches:
            if cache.isFull():
                victim_id = min(cache.contents, key=lambda k: self.cache_scores[k])
                if self.cache_scores[object_id] > self.cache_scores[victim_id]:
                    yield self.env.process(cache.replaceObject(victim_id,object_id))
                    object_id = victim_id
            else:
                cache.cacheObject(object_id)
                return

    def vipForwarding(self):
        for k in range(self.num_objects):
            self.vip_allocs[k].clear()

        fwd_vips = {}

        for b in self.fib_inv:
            fwd_vips[b] = defaultdict(int)
            vip_diff = [self.vip_counts[k] - self.neighbor_vip_counts[b][k] for k in self.fib_inv[b]]
            k_star_index = np.argmax(vip_diff)
            k_star = self.fib_inv[b][k_star_index]
            self.vip_allocs[k_star].append((vip_diff[k_star_index], b))
        
        for k in self.fib:
            self.vip_allocs[k].sort(reverse = True)
            while (self.vip_counts[k] > 0) and (self.vip_allocs[k]):
                _, b = self.vip_allocs[k].pop(0)
                vip_amount = min(self.slot_len * self.out_links[b].link_cap, self.vip_counts[k])
                fwd_vips[b][k] = vip_amount

        return fwd_vips

    def vipCaching(self):
        for k in range(self.num_objects):
            self.cache_scores[k] = self.vip_rx_windows[k].mean
        
        cache_score_ranks = [i for i in np.argsort(self.cache_scores)][::-1]
        temp_idx = 0
        for j, cache in enumerate(self.caches):
            self.virtual_caches[j] = cache_score_ranks[temp_idx : temp_idx + cache.capacity].copy()
            temp_idx += cache.capacity

    def vipProcess(self):
        while True:
            # Timeout for a time equal to specified slot length
            yield self.env.timeout(self.slot_len)
            
            # Set VIP counts to 0 for all sourced objects
            if self.is_source:
                for k in self.permastore.contents:
                    self.vip_counts[k] = 0

            # Determine virtual plane forwarding
            fwd_vips = self.vipForwarding()

            # Update local VIP counts with sent VIPs
            for b in self.fib_inv:
                for k in self.fib:
                    self.vip_counts[k] = max(0, self.vip_counts[k] - fwd_vips[b][k])

            # Send VIP count updates
            for remote_id in self.ctrl_out_links:
                self.ctrl_out_links[remote_id].pushUpdate(self.vip_counts)

            # Receive VIP count updates
            for remote_id in self.ctrl_in_links:
                self.neighbor_vip_counts[remote_id] = yield self.ctrl_in_links[remote_id].getUpdate() 

            # Send VIPs
            for remote_id in self.ctrl_out_links:
                if remote_id in fwd_vips:
                    self.ctrl_out_links[remote_id].pushVips(fwd_vips[remote_id])
                    for object_id in self.fib_inv[remote_id]:
                        # Record TX VIP stats
                        self.neighbor_vip_tx[remote_id, object_id].append(fwd_vips[remote_id][object_id])
                else:
                    self.ctrl_out_links[remote_id].pushVips({})
            
            # Receive VIPs
            for remote_id in self.ctrl_in_links:
                rx_vips = yield self.ctrl_in_links[remote_id].getVips()
                for object_id in rx_vips:
                    self.vip_counts[object_id] += rx_vips[object_id]
                    # Record RX VIP stats
                    self.vip_rx[object_id] += rx_vips[object_id]                    
            
            # Update VIP RX windows
            for k in range(self.num_objects):
                self.vip_rx_windows[k].append(self.vip_rx[k])
                self.vip_rx[k] = 0

            # Update local vip counts with cache-drained VIPs
            if self.has_caches:
                self.vipCaching()
                for j, cache in enumerate(self.caches):
                    for k in self.virtual_caches[j]:
                        self.vip_counts[k] = max(0, self.vip_counts[k] - self.slot_len * cache.read_rate)

class MVIPNode(VIPNode):
    def __init__(self, env, node_id, num_objects, pen_weight, **vip_args):
        super().__init__(env, node_id, num_objects, pen_weight, **vip_args)
        self.virtual_object_locs = {key : -2 for key in range(num_objects)}
        self.tier_mapping, self.tier_slices = [], []
    
    def addCache(self, cache):
        super().addCache(cache)
        self.tier_mapping += [len(self.caches)-1] * cache.capacity
        if self.tier_slices:
            end_of_last_slice = self.tier_slices[-1].stop
            new_slice = slice(end_of_last_slice, end_of_last_slice + cache.capacity)
        else:
            new_slice = slice(0, cache.capacity)
        self.tier_slices.append(new_slice)

    def decideCaching(self, object_id):
        benefits = []
        victims = []
        for j, cache in enumerate(self.caches):
            if cache.isFull():
                victim_id = min(cache.contents, key=lambda k: self.cache_scores[k])
                benefits.append(cache.read_rate*(self.cache_scores[object_id] - self.cache_scores[victim_id]) - self.pw * (cache.read_penalty + cache.write_penalty))
                victims.append(victim_id)
            else:
                benefits.append(cache.read_rate*self.cache_scores[object_id] - self.pw * cache.write_penalty)
                victims.append(None)
        j = np.argmax(benefits)
        if benefits[j] > 0:
            if victims[j] is not None:
                yield self.env.process(self.caches[j].replaceObject(victims[j], object_id))
                self.env.process(self.decideCaching(victims[j]))
            else:
                self.caches[j].cacheObject(object_id)

    def vipForwarding(self):
        b_list = list(self.fib_inv.keys())
        k_list = list(self.fib.keys())
        vip_diff = np.zeros((len(b_list), len(k_list)))
        vip_alloc = {}
        
        for b_id, b in enumerate(b_list):
            for k_id, k in enumerate(k_list):
                vip_diff[b_id,k_id] = self.vip_counts[k] - self.neighbor_vip_counts[b][k]
        
        k_ids = np.argmax(vip_diff, axis=1)
        for b_id, b in enumerate(b_list):
            k = k_list[k_ids[b_id]]
            vip_alloc[b] = k
        
        return vip_alloc

    def vipCaching(self):
        r_nj, p_rj, p_wj = [], [], []
        total_cache_size = 0
        for cache in self.caches:
            total_cache_size += cache.capacity
            r_nj.append(cache.read_rate)
            p_rj.append(cache.read_penalty)
            p_wj.append(cache.write_penalty)
        
        # Initialize cost matrix, with shape J x k
        cost_matrix = np.zeros((total_cache_size, self.num_objects))

        for k in range(self.num_objects):
            # Locate object
            object_loc = self.virtual_object_locs[k]
            # Update cache scores
            if object_loc == -1:
                self.cache_scores[k] = 0
            else:
                self.cache_scores[k] = self.vip_rx_windows[k].mean
            
            cs_k = self.cache_scores[k]
            for j in range(len(self.caches)):
                tier_slice = self.tier_slices[j]
                # TODO: penalty weights were missing here, investigate why
                if object_loc == j:
                    cost_matrix[tier_slice, k] = r_nj[j] * cs_k + self.pw * p_rj[j]
                else:
                    cost_matrix[tier_slice, k] = r_nj[j] * cs_k - self.pw * p_wj[j]

        # Round small values to zero
        cost_matrix[np.abs(cost_matrix) < 1e-6] = 0
        
        # Run LSAP. Rows are cache spaces, columns are object ids.
        row_ind, col_ind  = linear_sum_assignment(cost_matrix, maximize = True)
        
        # Reset virtual object locs
        self.virtual_object_locs = resetDict(self.virtual_object_locs, -2)
        
        # Clear virtual caches
        for v_cache in self.virtual_caches:
            v_cache.clear()
        
        # Assign objects to virtual cache
        for i, k in zip(row_ind, col_ind):
            if cost_matrix[i][k] >= 0:
                v_cache = self.virtual_caches[self.tier_mapping[i]]
                v_cache.append(k)
                self.virtual_object_locs[k] = self.tier_mapping[i]

    def vipProcess(self):
        while True:
            # Timeout for a time equal to specified slot length
            yield self.env.timeout(self.slot_len)
            
            # Set VIP counts to 0 for all sourced objects
            if self.is_source:
                for k in self.permastore.contents:
                    self.vip_counts[k] = 0

            # Determine virtual plane caching
            if self.has_caches:
                self.vipCaching()

            # Determine virtual plane forwarding
            vip_alloc = self.vipForwarding()
            fwd_vips = {}

            # Update local VIP counts with sent VIPs
            for b, k in vip_alloc.items():
                fwd_vips[b] = defaultdict(int)
                vip_amount = min(self.vip_counts[k], self.out_links[b].link_cap)
                fwd_vips[b][k] = vip_amount
                self.vip_counts[k] = max(self.vip_counts[k] - vip_amount, 0)

            # Update local vip counts with cache-drained VIPs
            for j, cache in enumerate(self.caches):
                for k in self.virtual_caches[j]:
                    self.vip_counts[k] = max(self.vip_counts[k] - self.slot_len * cache.read_rate, 0) 

            # Send VIPs
            for remote_id in self.ctrl_out_links:
                if remote_id in fwd_vips:
                    self.ctrl_out_links[remote_id].pushVips(fwd_vips[remote_id])
                    for object_id in self.fib_inv[remote_id]:
                        # Record TX VIP stats
                        self.neighbor_vip_tx[remote_id, object_id].append(fwd_vips[remote_id][object_id])
                else:
                    self.ctrl_out_links[remote_id].pushVips({})
            
            # Receive VIPs
            for remote_id in self.ctrl_in_links:
                rx_vips = yield self.ctrl_in_links[remote_id].getVips()
                for object_id in rx_vips:
                    self.vip_counts[object_id] += rx_vips[object_id]
                    # Record RX VIP stats
                    self.vip_rx[object_id] += rx_vips[object_id]                    
            
            # Update VIP RX windows
            for k in range(self.num_objects):
                self.vip_rx_windows[k].append(self.vip_rx[k])
                self.vip_rx[k] = 0

            # Send VIP count updates
            for remote_id in self.ctrl_out_links:
                self.ctrl_out_links[remote_id].pushUpdate(self.vip_counts)

            # Receive VIP count updates
            for remote_id in self.ctrl_in_links:
                self.neighbor_vip_counts[remote_id] = yield self.ctrl_in_links[remote_id].getUpdate()