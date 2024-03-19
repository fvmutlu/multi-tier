# External package imports
import simpy as sp

# Builtin imports
from collections import namedtuple

class Permastore(object):
    def __init__(self, env: sp.Environment, read_rate: int, contents: set):
        self.env = env
        self.read_rate = read_rate
        self.contents = contents
        self.read_queue = sp.Store(env)
        self.out_buffer = sp.Store(env)

        # Statistics
        self.stats = {
            'reads': 0,
            'read_delay': 0
        }
    
    def isSourced(self, object_id):
        return (object_id in self.contents)
    
    def permastoreController(self):
        while True:
            object_id = yield self.read_queue.get()
            obj = yield self.env.process(self.readProcess(object_id))
            self.out_buffer.put(obj)
    
    def readProcess(self, object_id):
        yield self.env.timeout(1/self.read_rate)
        if not self.isSourced(object_id):
            print("ERROR: Algorithm error while trying to read object: object not sourced. Results should be ignored.")
        return object_id

    def readObject(self, object_id):
        self.stats['reads'] += 1
        tic = self.env.now
        self.read_queue.put(object_id)
        obj = yield self.out_buffer.get()
        toc = self.env.now
        self.stats['read_delay'] += toc - tic
        return obj
        
    def getStats(self):
        return self.stats

CacheTask = namedtuple('CacheTask', ['type', 'object_id'])

class Cache(object):
    def __init__(self, env: sp.Environment, cap: int, read_rate: int | float, write_rate: int | float, read_pen: int | float, write_pen: int | float):
        self.env = env
        self.capacity = cap
        self.read_rate = read_rate
        self.write_rate = write_rate
        self.read_penalty = read_pen
        self.write_penalty = write_pen
        self.contents = set()
        self.cur_size = 0
        self.task_queue = sp.Store(env)
        self.out_buffer = sp.Store(env)

        # Statistics
        self.stats = {
            'reads': 0,
            'writes': 0,
            'replacements': 0,
            'read_delay': 0,
        }
    def isFull(self):
        return (self.cur_size >= self.capacity)

    def isCached(self, object_id):
        return (object_id in self.contents)

    # This looping process observes the task queue
    # and schedules one-off processes accordingly
    def cacheController(self):
        while True:
            task = yield self.task_queue.get()
            if task.type == 'r':
                obj = yield self.env.process(self.readProcess(task.object_id))
                self.out_buffer.put(obj)
            elif task.type == 'w':
                yield self.env.process(self.writeProcess(task.object_id))

    # Since we don't have actual data content for these simulations
    # read process trivially returns the object id passed
    def readProcess(self, object_id):
        yield self.env.timeout(1/self.read_rate)        
        return object_id

    def writeProcess(self, object_id):
        yield self.env.timeout(1/self.write_rate)

    # This one-off process schedules a read task, then yields
    # until it can retrieve an object from the output buffer
    def readObject(self, object_id):
        if self.isCached(object_id):
            self.stats['reads'] += 1            
            task = CacheTask(type = 'r', object_id = object_id)
            tic = self.env.now
            self.task_queue.put(task)
            obj = yield self.out_buffer.get()
            toc = self.env.now
            self.stats['read_delay'] += toc - tic
            return obj        
        else:
            print(f"TIME: {self.env.now} ERROR: Algorithm error while trying to read object {object_id}: object not in cache.")
            raise IndexError("Object not in cache.")

    # This function schedules a write task to cache an object.
    def cacheObject(self, object_id):
        if self.isFull():
            print(f"TIME: {self.env.now} ERROR: Algorithm error while trying to cache object {object_id}: cache already full.")
            raise IndexError("Cache full")
        elif self.isCached(object_id):
            return True
        else:
            self.stats['writes'] += 1
            self.contents.add(object_id)
            self.cur_size += 1
            task = CacheTask(type = 'w', object_id = object_id)
            self.task_queue.put(task)
            return True

    def replaceObject(self, evicted_object_id, cached_object_id):
        if self.isCached(evicted_object_id):
            self.stats['replacements'] += 1
            self.contents.remove(evicted_object_id)
            task = CacheTask(type = 'r', object_id = evicted_object_id)
            self.task_queue.put(task)
            self.contents.add(cached_object_id)
            task = CacheTask(type = 'w', object_id = cached_object_id)
            self.task_queue.put(task)
            obj = yield self.out_buffer.get()
            return obj
        else:
            print(f"TIME: {self.env.now} ERROR: Algorithm error while trying to evict object {evicted_object_id}: object not in cache.")
            raise IndexError(f"Object {evicted_object_id} not in cache.")
    
    def getStats(self):
        return self.stats