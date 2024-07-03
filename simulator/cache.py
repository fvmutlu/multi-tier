# External package imports
import simpy as sp
from numpy.random import randint

# Builtin imports
from collections import deque
from dataclasses import dataclass


class Permastore(object):
    def __init__(self, env: sp.Environment, read_rate: int, contents: set):
        self.env = env
        self.read_rate = read_rate
        self.contents = contents
        self.read_queue = sp.Store(env)
        self.out_buffer = sp.Store(env)

        # Statistics
        self.stats = {"reads": 0, "read_delay": 0}

    def isSourced(self, object_id):
        return object_id in self.contents

    def permastoreController(self):
        while True:
            object_id = yield self.read_queue.get()
            obj = yield self.env.process(self.readProcess(object_id))
            self.out_buffer.put(obj)

    def readProcess(self, object_id):
        yield self.env.timeout(1 / self.read_rate)
        if not self.isSourced(object_id):
            print(
                "ERROR: Algorithm error while trying to read object: object not sourced. Results should be ignored."
            )
        return object_id

    def readObject(self, object_id):
        self.stats["reads"] += 1
        tic = self.env.now
        self.read_queue.put(object_id)
        obj = yield self.out_buffer.get()
        toc = self.env.now
        self.stats["read_delay"] += toc - tic
        return obj

    def getStats(self):
        return self.stats

@dataclass
class CacheTask:
    type: str
    object_id: int
    origin_id: int = None
    seq_id: int = None
    eviction_token: int = None

@dataclass
class CacheReadOutput:
    object_id: int
    origin_id: int = None
    seq_id: int = None
    eviction_token: int = None


class Cache(object):
    def __init__(
        self,
        env: sp.Environment,
        cap: int,
        read_rate: int | float,
        write_rate: int | float,
        read_pen: int | float,
        write_pen: int | float,
    ):
        self.env = env
        self.capacity = cap
        self.read_rate = read_rate
        self.write_rate = write_rate
        self.read_penalty = read_pen
        self.write_penalty = write_pen
        self.contents = set()
        self.cur_size = 0
        self.task_queue = sp.Store(env)
        self.out_buffer = sp.FilterStore(env)

        self.log = False

        # Statistics
        self.stats = {
            "reads": 0,
            "writes": 0,
            "replacements": 0,
            "read_delay": 0,
        }

    def isFull(self):
        return self.cur_size >= self.capacity

    def isCached(self, object_id):
        return object_id in self.contents

    def shouldLog(self):
        self.log = True

    # This looping process observes the task queue
    # and schedules one-off processes accordingly
    def cacheController(self):
        while True:
            task = yield self.task_queue.get()
            if self.log:
                print(f"TIME: {self.env.now:.5f} INFO: Task received: {task}")
            if task.type == "r":
                yield self.env.process(self.readProcess())
                self.out_buffer.put(CacheReadOutput(object_id=task.object_id, origin_id=task.origin_id, seq_id=task.seq_id))
            elif task.type == "e":
                yield self.env.process(self.readProcess())
                self.out_buffer.put(CacheReadOutput(object_id=task.object_id, eviction_token=task.eviction_token))
            elif task.type == "w":
                yield self.env.process(self.writeProcess())
            if self.log:
                print(f"TIME: {self.env.now:.5f} INFO: Task delivered: {task}")
                print(f"TIME: {self.env.now:.5f} output_buffer: {self.out_buffer.items}")

    # Since we don't have actual data content for these simulations
    # read process trivially returns the object id passed
    def readProcess(self):
        yield self.env.timeout(1 / self.read_rate)

    def writeProcess(self):
        yield self.env.timeout(1 / self.write_rate)

    # This one-off process schedules a read task, then yields
    # until it can retrieve an object from the output buffer
    def readObject(self, object_id, origin_id, seq_id):
        if self.isCached(object_id):
            self.stats["reads"] += 1
            task = CacheTask(type="r", object_id=object_id, origin_id=origin_id, seq_id=seq_id)
            tic = self.env.now
            self.task_queue.put(task)
            if self.log:
                print(f"TIME: {self.env.now:.5f} INFO: Reading object:{object_id}, origin:{origin_id}, seq:{seq_id} from cache.")
            obj = yield self.out_buffer.get(lambda output: output.object_id == object_id and output.origin_id == origin_id and output.seq_id == seq_id)
            if self.log:
                print(f"TIME: {self.env.now:.5f} INFO: Read object:{object_id}, origin:{origin_id}, seq:{seq_id} from cache.")
            toc = self.env.now
            self.stats["read_delay"] += toc - tic
            return obj
        else:
            print(
                f"TIME: {self.env.now} ERROR: Algorithm error while trying to read object {object_id}: object not in cache."
            )
            raise IndexError("Object not in cache.")

    # This function schedules a write task to cache an object.
    def cacheObject(self, object_id):
        if self.isFull():
            print(
                f"TIME: {self.env.now} ERROR: Algorithm error while trying to cache object {object_id}: cache already full."
            )
            raise IndexError("Cache full")
        elif self.isCached(object_id):
            return True
        else:
            self.stats["writes"] += 1
            self.contents.add(object_id)
            self.cur_size += 1
            task = CacheTask(type="w", object_id=object_id)
            self.task_queue.put(task)
            return True

    def replaceObject(self, evicted_object_id, cached_object_id):
        if self.isCached(evicted_object_id):
            self.stats["replacements"] += 1
            self.contents.remove(evicted_object_id)
            evict_task = CacheTask(type="e", object_id=evicted_object_id, eviction_token=randint(2**32-1))
            self.task_queue.put(evict_task)
            self.contents.add(cached_object_id)
            write_task = CacheTask(type="w", object_id=cached_object_id)
            self.task_queue.put(write_task)
            obj = yield self.out_buffer.get(lambda output: output.object_id == evicted_object_id and output.eviction_token == evict_task.eviction_token)
            return obj
        else:
            print(
                f"TIME: {self.env.now} ERROR: Algorithm error while trying to evict object {evicted_object_id}: object not in cache."
            )
            raise IndexError(f"Object {evicted_object_id} not in cache.")

    def getStats(self):
        return self.stats


class FIFOCache(Cache):
    def __init__(
        self,
        env: sp.Environment,
        cap: int,
        read_rate: int | float,
        write_rate: int | float,
        read_pen: int | float,
        write_pen: int | float,
    ):
        super().__init__(env, cap, read_rate, write_rate, read_pen, write_pen)
        self.contents = deque(maxlen=self.capacity)

    def cacheObject(self, object_id):
        if self.isFull():
            print(
                f"TIME: {self.env.now} ERROR: Algorithm error while trying to cache object {object_id}: cache already full."
            )
            raise IndexError("Cache full")
        elif self.isCached(object_id):
            return True
        else:
            self.stats["writes"] += 1
            self.contents.append(object_id)
            self.cur_size += 1
            task = CacheTask(type="w", object_id=object_id)
            self.task_queue.put(task)
            return True

    def replaceObject(self, evicted_object_id, cached_object_id):
        if not self.isFull():
            print(
                f"TIME: {self.env.now} ERROR: Algorithm error while trying to replace: cache not full."
            )
        if self.isCached(evicted_object_id):
            if self.contents[0] != evicted_object_id:
                print(
                    f"TIME: {self.env.now} ERROR: FIFO error while trying to evict object {evicted_object_id}: object not first in."
                )
            self.stats["replacements"] += 1
            self.contents.popleft()
            task = CacheTask(type="r", object_id=evicted_object_id)
            self.task_queue.put(task)
            self.contents.append(cached_object_id)
            task = CacheTask(type="w", object_id=cached_object_id)
            self.task_queue.put(task)
            obj = yield self.out_buffer.get()
            return obj
        else:
            print(
                f"TIME: {self.env.now} ERROR: Algorithm error while trying to evict object {evicted_object_id}: object not in cache."
            )
            raise IndexError(f"Object {evicted_object_id} not in cache.")
