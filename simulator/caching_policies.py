# External package imports
import numpy as np

# Builtin imports

# Internal imports
from .node import Node
from .utils import wique
from .cache import FIFOCache


class LRUNode(Node):
    def lruInit(self, num_objects):
        self.lru_table = [0] * num_objects

    def receiveInterest(self, remote_id, request):
        self.lru_table[request.object_id] = self.env.now
        super().receiveInterest(remote_id, request)

    def decideCaching(self, object_id):
        for cache in self.caches:
            if cache.isFull():
                victim_id = min(cache.contents, key=lambda k: self.lru_table[k])
                if self.lru_table[object_id] > self.lru_table[victim_id]:
                    yield self.env.process(cache.replaceObject(victim_id, object_id))
                    object_id = victim_id
            else:
                cache.cacheObject(object_id)
                return


class LFUNode(Node):
    def lfuInit(self, num_objects):
        self.lfu_table = [0] * num_objects

    def receiveInterest(self, remote_id, request):
        self.lfu_table[request.object_id] += 1
        super().receiveInterest(remote_id, request)

    def decideCaching(self, object_id):
        for j, cache in enumerate(self.caches):
            if cache.isFull():
                victim_id = min(cache.contents, key=lambda k: self.lfu_table[k])
                if self.lfu_table[object_id] > self.lfu_table[victim_id]:
                    yield self.env.process(cache.replaceObject(victim_id, object_id))
                    object_id = victim_id
            else:
                cache.cacheObject(object_id)
                return


class PALFUNode(LFUNode):
    def lfuInit(self, num_objects, pen_weight):
        super().lfuInit(num_objects)
        self.pw = pen_weight

    def decideCaching(self, object_id):
        benefits = []
        victims = []
        for cache in self.caches:
            r_nj = cache.read_rate
            p_rj = self.pw * cache.read_penalty
            p_wj = self.pw * cache.write_penalty
            if cache.isFull():
                victim_id = min(cache.contents, key=lambda k: self.lfu_table[k])
                benefit = r_nj * (
                    self.lfu_table[object_id] - self.lfu_table[victim_id]
                ) - (p_rj + p_wj)
                benefits.append(benefit)
                victims.append(victim_id)
            else:
                benefit = r_nj * self.lfu_table[object_id] - p_wj
                benefits.append(benefit)
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


class WLFUNode(LFUNode):
    def lfuInit(self, num_objects, win_size, period):
        self.lfu_table = [0] * num_objects
        self.lfu_counters = [0] * num_objects
        self.lfu_windows = [wique(maxlen=win_size) for _ in range(num_objects)]
        self.env.process(self.wlfuProcess(num_objects, period))

    def receiveInterest(self, remote_id, request):
        self.lfu_counters[request.object_id] += 1
        Node.receiveInterest(self, remote_id, request)

    def wlfuProcess(self, num_objects, period):
        while True:
            yield self.env.timeout(period)
            for k in range(num_objects):
                self.lfu_windows[k].append(self.lfu_counters[k])
                self.lfu_counters[k] = 0
                self.lfu_table[k] = self.lfu_windows[k].mean


class PAWLFUNode(WLFUNode):
    def lfuInit(self, num_objects, win_size, period, pen_weight):
        super().lfuInit(num_objects, win_size, period)
        self.pw = pen_weight

    def decideCaching(self, object_id):
        benefits = []
        victims = []
        for cache in self.caches:
            r_nj = cache.read_rate
            p_rj = self.pw * cache.read_penalty
            p_wj = self.pw * cache.write_penalty
            if cache.isFull():
                victim_id = min(cache.contents, key=lambda k: self.lfu_table[k])
                benefit = r_nj * (
                    self.lfu_table[object_id] - self.lfu_table[victim_id]
                ) - (p_rj + p_wj)
                benefits.append(benefit)
                victims.append(victim_id)
            else:
                benefit = r_nj * self.lfu_table[object_id] - p_wj
                benefits.append(benefit)
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


class UNIFNode(Node):
    def decideCaching(self, object_id):
        cache = np.random.choice(self.caches)
        if cache.isFull():
            victim_id = np.random.choice(tuple(cache.contents))
            yield self.env.process(cache.replaceObject(victim_id, object_id))
        else:
            cache.cacheObject(object_id)


class FIFONode(Node):
    def addCache(self, cache):
        fifocache = FIFOCache(
            self.env,
            cache.capacity,
            cache.read_rate,
            cache.write_rate,
            cache.read_penalty,
            cache.write_penalty,
        )
        super().addCache(fifocache)

    def decideCaching(self, object_id):
        for j, cache in enumerate(self.caches):
            if cache.isFull():
                victim_id = cache.contents[0]
                yield self.env.process(cache.replaceObject(victim_id, object_id))
                object_id = victim_id
            else:
                cache.cacheObject(object_id)
                return
