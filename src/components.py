import simpy as sp
from collections import namedtuple

class Request(object):
    def __init__(self, env, origin_id, seq_id, object_id):
        self.env = env
        self.origin_id = origin_id
        self.seq_id = seq_id
        self.object_id = object_id
        self.log = {'origin': int(origin_id), 'seq': int(seq_id), 'obj': int(object_id), 'timestamps': [], 'nodes': []}
    
    def logEvent(self, node_id):
        self.log['timestamps'].append(self.env.now)
        self.log['nodes'].append(node_id)
    
    def getLastTimestamp(self):
        return self.log['timestamps'][-1]
    
    def getDelay(self):
        return self.log['timestamps'][-1] - self.log['timestamps'][0]

class Packet(object):
    def __init__(self, request):
        self.request = request
        self.object_id = request.object_id
    
    def isInterest(self):
        return False
    
    def isData(self):
        return False
    
    def getLastTimestamp(self):
        return self.request.getLastTimestamp()

class InterestPacket(Packet):
    def isInterest(self):
        return True

class DataPacket(Packet):
    def isData(self):
        return True