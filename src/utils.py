from itertools import product
from collections import deque, defaultdict
import numpy as np

def namedProduct(**items):
    names = items.keys()
    vals = items.values()
    for res in product(*vals):
        yield dict(zip(names, res))

def namedZip(**items):
    names = items.keys()
    vals = items.values()
    for res in zip(*vals):
        yield dict(zip(names, res))

def invertDict(d):
    new_dict = defaultdict(list)
    for key, seq in d.items():
        for item in seq:
            new_dict[item].append(key)
    return dict(new_dict)

def randargmax(arr, axis, seed=1):
    rng = np.random.default_rng(seed)
    return np.apply_along_axis(lambda x: rng.choice(np.where(x==x.max())[0]), axis=axis, arr=arr)

class wique:
    def __init__(self, maxlen = 10):
        self.maxlen = maxlen
        self.curlen = 0
        self.q = deque(maxlen = maxlen)
        self.mean = 0
        self.sum = 0

    def append(self, x):
        if self.curlen < self.maxlen:
            self.sum += x
            self.mean = (self.mean * self.curlen + x) / (self.curlen + 1)
            self.curlen += 1
        else:
            self.sum = self.sum - self.q[0] + x
            self.mean = (self.mean * self.maxlen - self.q[0] + x) / self.maxlen
        self.q.append(x)