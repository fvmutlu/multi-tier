# External package imports
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# Builtin imports
from collections import defaultdict

# Internal imports
from .node import IANode as Node
from .utils import wique, invertDict, resetDict
from .logutils import rootlogger


class VIPNode(Node):
    def __init__(self, env, node_id, num_objects, pen_weight, **vip_args):
        super().__init__(env, node_id)
        self.num_objects = num_objects
        self.pw = pen_weight
        self.cacheable_objects = [object_id for object_id in range(self.num_objects)]

        # VIP Parameters
        self.vip_inc = vip_args["vip_inc"]
        self.slot_len = vip_args["vip_slot_len"]
        self.win_size = int(vip_args["vip_win_size"] / vip_args["vip_slot_len"])
        self.vip_ia_factor = [vip_args["vip_ia_factor"]] * num_objects
        self.vip_ia_coeff = vip_args["vip_ia_coeff"]

        # VIP State Variables
        self.vip_counts = [0] * num_objects
        self.vip_rx = [0] * num_objects
        self.vip_rx_windows = [
            wique(maxlen=self.win_size) for _ in range(self.num_objects)
        ]
        self.cache_scores = [0] * num_objects
        self.neighbor_vip_counts = {}
        self.neighbor_vip_tx = {}
        self.virtual_caches = []

        # LRT in VIP to break ties
        self.link_delays = {}
        self.req_timestamps = {}

        # Additional stats relating to VIP
        # self.stats["vip_count_sum"] = []
        # self.stats["pit_count_sum"] = []
        # self.stats["vip_caching_avg_time"] = 0
        # self.stats["vip_ia_factors"] = [0] * num_objects

        # Init VIP process
        self.env.process(self.vipProcess())

    def addInputLink(self, remote_id, link, ctrl_link):
        super().addInputLink(remote_id, link, ctrl_link)
        self.neighbor_vip_counts[remote_id] = [0] * self.num_objects

    def addOutputLink(self, remote_id, link, ctrl_link):
        super().addOutputLink(remote_id, link, ctrl_link)
        self.link_delays[remote_id] = wique(maxlen=1)
        self.req_timestamps[remote_id] = {}

    def packetProcessor(self):
        while True:
            remote_id, pkt = yield self.pkt_buffer.get()
            if pkt.isInterest():
                self.receiveInterest(remote_id, pkt.request)
            elif pkt.isData():
                delay = self.env.now - self.req_timestamps[remote_id].pop(
                    (pkt.request.origin_id, pkt.request.seq_id)
                )
                self.link_delays[remote_id].append(delay)
                self.receiveData(pkt.request)

    def addCache(self, cache):
        if self.has_caches:
            rootlogger.warning(
                "More than one cache added to single tier VIP node."
                + " Only the fastest cache tier will be used."
            )
        super().addCache(cache)
        self.virtual_caches.append(set())

    def addPermastore(self, permastore):
        super().addPermastore(permastore)
        self.cacheable_objects = [
            object_id
            for object_id in range(self.num_objects)
            if object_id not in self.permastore.contents
        ]

    def addFIB(self, fib):
        super().addFIB(fib)
        self.fib_inv = invertDict(fib)
        for object_id in fib:
            for remote_id in fib[object_id]:
                self.neighbor_vip_tx[remote_id, object_id] = wique(maxlen=self.win_size)

    def receiveInterest(self, remote_id, request):
        if remote_id == self.id:
            self.vip_counts[request.object_id] += (
                self.vip_inc / self.vip_ia_factor[request.object_id]
            )
            self.vip_rx[request.object_id] += self.vip_inc
        super().receiveInterest(remote_id, request)

    def forwardInterest(self, request):
        object_id = request.object_id
        neighbors = self.fib[object_id]
        link_profits = np.array(
            [self.neighbor_vip_tx[remote_id, object_id].mean for remote_id in neighbors]
        )

        # LRT to break ties
        link_delays = np.array(
            [self.link_delays[remote_id].mean for remote_id in neighbors]
        )
        max_profit_link_idx = np.where(link_profits == link_profits.max())[0]
        max_profit_link_delays = np.array(
            [link_delays[idx] for idx in max_profit_link_idx]
        )
        idx = max_profit_link_idx[np.argmin(max_profit_link_delays)]

        remote_id = neighbors[idx]

        # LRT tracking
        self.req_timestamps[remote_id][
            (request.origin_id, request.seq_id)
        ] = self.env.now
        # LRT tracking

        self.sendInterestPacket(remote_id, request)

    def decideCaching(self, object_id):
        if self.cache_scores[object_id] <= 0:
            return
        cache = self.caches[0]
        if cache.isFull():
            victim_id = min(cache.contents, key=lambda k: self.cache_scores[k])
            benefit = cache.read_rate * (
                self.cache_scores[object_id] - self.cache_scores[victim_id]
            ) - self.pw * (cache.read_penalty + cache.write_penalty)
            if benefit > 0:
                yield self.env.process(cache.replaceObject(victim_id, object_id))
                object_id = victim_id
        else:
            benefit = (
                cache.read_rate * self.cache_scores[object_id]
                - self.pw * cache.write_penalty
            )
            if benefit > 0:
                cache.cacheObject(object_id)
            return
    
    def vipForwarding(self):
        vip_allocs = defaultdict(list)

        for b in self.fib_inv:
            vip_diff = [
                self.vip_counts[k] - self.neighbor_vip_counts[b][k]
                for k in self.fib_inv[b]
            ]
            k_star_index = np.argmax(vip_diff)
            k_star = self.fib_inv[b][k_star_index]
            vip_allocs[k_star].append((vip_diff[k_star_index], b))

        return vip_allocs

    def updateVipCacheScores(self):
        for k in self.cacheable_objects:
            self.cache_scores[k] = self.vip_counts[k]

    def vipCaching(self):
        cache, virtual_cache = self.caches[0], self.virtual_caches[0]
        temp_cache_scores = [0] * self.num_objects
        for k in self.cacheable_objects:
            # Update cache scores for use outside virtual plane (this can be different for variants)
            # self.cache_scores[k] = self.vip_counts[k]
            # Use local scope scores for virtual plane caching algorithm
            temp_cache_scores[k] = cache.read_rate * self.cache_scores[k]
            if k in virtual_cache:
                temp_cache_scores[k] += self.pw * cache.read_penalty
            else:
                temp_cache_scores[k] -= self.pw * cache.write_penalty

        # Sort from highest to lowest cache score
        sorted_cache_scores = np.flip(np.sort(temp_cache_scores))
        # Obtain object ids for sorted scores
        sorted_cache_scores_idx = np.flip(np.argsort(temp_cache_scores))
        # Find the index of the first zero score
        first_zero_index = np.where(sorted_cache_scores <= 0)[0][0]
        # Cache up to either the first zero score object, or up to capacity
        stop_index = min(first_zero_index, cache.capacity)
        virtual_cache.clear()
        virtual_cache.update(sorted_cache_scores_idx[:stop_index])

    def drainVipsByCaching(self):
        for j, cache in enumerate(self.caches):
            cache_decrement = self.slot_len * cache.read_rate / cache.capacity
            for k in self.virtual_caches[j]:
                self.vip_counts[k] = max(self.vip_counts[k] - cache_decrement, 0)

    def vipProcess(self):
        # avg_cpu_time, cpu_counter = 0, 0
        while True:
            # Set VIP counts to 0 for all sourced objects
            if self.is_source:
                for k in self.permastore.contents:
                    self.vip_counts[k] = 0

            # Update neighbors' information with local node's VIP counts
            for remote_id in self.ctrl_out_links:
                self.ctrl_out_links[remote_id].pushUpdate(self.vip_counts)

            # Update local node's VIP information with neighbors' VIP counts
            for remote_id in self.ctrl_in_links:
                self.neighbor_vip_counts[remote_id] = yield self.ctrl_in_links[
                    remote_id
                ].getUpdate()

            # Determine virtual plane forwarding
            vip_allocs = self.vipForwarding()

            # tic = time.perf_counter()
            if self.has_caches:
                # Update cache scores for use outside virtual plane (this can be different for variants)
                self.updateVipCacheScores()
                # Determine virtual plane caching
                self.vipCaching()
                # Decrement VIP counts due to caching
                self.drainVipsByCaching()
            # toc = time.perf_counter()
            # avg_cpu_time, cpu_counter = (avg_cpu_time * cpu_counter + (toc - tic)) / (cpu_counter + 1), cpu_counter + 1
            # self.stats["vip_caching_avg_time"] = avg_cpu_time

            # Determine actual amount of VIPs that will be forwarded
            fwd_vips = {}
            for b in self.fib_inv:
                fwd_vips[b] = defaultdict(int)

            for k in self.fib:
                vip_allocs[k].sort()
                while (self.vip_counts[k] > 0) and (vip_allocs[k]):
                    _, b = vip_allocs[k].pop()
                    vip_amount = min(
                        self.slot_len * self.out_links[b].link_cap, self.vip_counts[k]
                    )
                    fwd_vips[b][k] = vip_amount

            # Send VIPs
            for remote_id in self.ctrl_out_links:
                if remote_id in fwd_vips:
                    self.ctrl_out_links[remote_id].pushVips(fwd_vips[remote_id])
                    for object_id in self.fib_inv[remote_id]:
                        # Record TX VIP stats
                        self.neighbor_vip_tx[remote_id, object_id].append(
                            fwd_vips[remote_id][object_id]
                        )
                else:
                    self.ctrl_out_links[remote_id].pushVips({})

            # Update local VIP counts with sent VIPs
            for b in self.fib_inv:
                for k in self.fib:
                    self.vip_counts[k] = max(0, self.vip_counts[k] - fwd_vips[b][k])

            # Wait for slot length duration and increment VIP counts due to
            # exogenous arrivals
            yield self.env.timeout(self.slot_len)

            # Receive VIPs from neighbors, increment VIP counts accordingly and
            # update VIP RX windows
            for remote_id in self.ctrl_in_links:
                rx_vips = yield self.ctrl_in_links[remote_id].getVips()
                for object_id in rx_vips:
                    self.vip_counts[object_id] += (
                        rx_vips[object_id] / self.vip_ia_factor[object_id]
                    )
                    self.vip_rx[object_id] += rx_vips[object_id]
            for k in range(self.num_objects):
                self.vip_rx_windows[k].append(self.vip_rx[k] / self.vip_ia_factor[k])
                # Update IA factor
                self.vip_ia_factor[k] *= (1 - self.vip_ia_coeff)
                self.vip_ia_factor[k] += self.vip_ia_coeff * self.vip_rx[k]
                self.vip_ia_factor[k] = max(self.vip_ia_factor[k], 1)
                # Reset time slot rx vip count
                self.vip_rx[k] = 0

            # Update VIP stats
            # self.stats["vip_count_sum"].append(sum(self.vip_counts))
            # self.stats["pit_count_sum"].append(sum([len(q) for q in self.pit.values()]))


class VIP2Node(VIPNode):
    def updateVipCacheScores(self):
        for k in self.cacheable_objects:
            self.cache_scores[k] = self.vip_rx_windows[k].mean


class VIPSBWNode(VIPNode):
    def drainVipsByCaching(self):
        for j, cache in enumerate(self.caches):
            cache_decrement = self.slot_len * cache.read_rate
            if self.virtual_caches[j]:
                k_star = max(self.virtual_caches[j], key=lambda k: self.cache_scores[k])
                self.vip_counts[k_star] = max(
                    self.vip_counts[k_star] - cache_decrement, 0
                )


class VIPSBW2Node(VIPNode):
    def updateVipCacheScores(self):
        for k in self.cacheable_objects:
            self.cache_scores[k] = self.vip_rx_windows[k].mean

    def drainVipsByCaching(self):
        for j, cache in enumerate(self.caches):
            cache_decrement = self.slot_len * cache.read_rate
            if self.virtual_caches[j]:
                k_star = max(self.virtual_caches[j], key=lambda k: self.cache_scores[k])
                self.vip_counts[k_star] = max(
                    self.vip_counts[k_star] - cache_decrement, 0
                )


class MVIPNode(VIPNode):
    def __init__(self, env, node_id, num_objects, pen_weight, **vip_args):
        super().__init__(env, node_id, num_objects, pen_weight, **vip_args)
        self.virtual_object_locs = {key: -2 for key in range(num_objects)}
        self.tier_mapping, self.tier_slices = [], []

    def addCache(self, cache):
        Node.addCache(self, cache)
        self.virtual_caches.append(set())
        self.tier_mapping += [len(self.caches) - 1] * cache.capacity
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
                benefits.append(
                    cache.read_rate
                    * (self.cache_scores[object_id] - self.cache_scores[victim_id])
                    - self.pw * (cache.read_penalty + cache.write_penalty)
                )
                victims.append(victim_id)
            else:
                benefits.append(
                    cache.read_rate * self.cache_scores[object_id]
                    - self.pw * cache.write_penalty
                )
                victims.append(None)
        j = np.argmax(benefits)
        if benefits[j] > 0:
            if victims[j] is not None:
                yield self.env.process(
                    self.caches[j].replaceObject(victims[j], object_id)
                )
                self.env.process(self.decideCaching(victims[j]))
            else:
                self.caches[j].cacheObject(object_id)

    def updateVipCacheScores(self):
        for k in self.cacheable_objects:
            self.cache_scores[k] = self.vip_rx_windows[k].mean

    def vipCaching(self):
        total_cache_size = sum(cache.capacity for cache in self.caches)

        # Initialize cost matrix, with shape J x k
        cost_matrix = np.zeros((total_cache_size, self.num_objects))

        for k in self.cacheable_objects:
            # Locate object
            object_loc = self.virtual_object_locs[k]
            # Update cache score
            # self.cache_scores[k] = self.vip_rx_windows[k].mean
            # The cs_k values act as temp scores for MVIP
            cs_k = self.cache_scores[k]
            for j, cache in enumerate(self.caches):
                tier_slice = self.tier_slices[j]
                if object_loc == j:
                    cost_matrix[tier_slice, k] = (
                        cache.read_rate * cs_k + self.pw * cache.read_penalty
                    )
                else:
                    cost_matrix[tier_slice, k] = (
                        cache.read_rate * cs_k - self.pw * cache.write_penalty
                    )

        # Round small values to zero
        cost_matrix[np.abs(cost_matrix) < 1e-6] = 0

        # Run LSAP. Rows are cache spaces, columns are object ids.
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

        # Clear virtual caches
        for v_cache in self.virtual_caches:
            v_cache.clear()

        # Reset virtual object locs
        self.virtual_object_locs = resetDict(
            self.virtual_object_locs, -2, keys=self.cacheable_objects
        )

        # Assign objects to virtual cache
        for i, k in zip(row_ind, col_ind):
            if cost_matrix[i][k] >= 0 and k in self.cacheable_objects:
                self.virtual_caches[self.tier_mapping[i]].add(k)
                self.virtual_object_locs[k] = self.tier_mapping[i]
