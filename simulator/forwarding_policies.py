# External package imports
import numpy as np

# Builtin imports
from collections import deque

# Internal imports
from .node import IANode as Node
from .utils import wique


class RoundRobinNode(Node):
    def addFIB(self, fib):
        super().addFIB(fib)
        self.link_queues = {}
        for k in fib:
            self.link_queues[k] = deque(fib[k])

    def forwardInterest(self, request):
        object_id = request.object_id
        remote_id = self.link_queues[object_id][0]
        self.link_queues[object_id].rotate(1)
        self.sendInterestPacket(remote_id, request)


class LeastResponseTimeNode(Node):
    def __init__(self, env, node_id):
        super().__init__(env, node_id)
        self.lrtInit()

    def lrtInit(self):
        self.link_delays = {}
        self.req_timestamps = {}

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

    def forwardInterest(self, request):
        object_id = request.object_id
        valid_link_avg_delays = np.array(
            [self.link_delays[link].mean for link in self.fib[object_id]]
        )
        rng = np.random.default_rng(1)
        idx = rng.choice(
            np.where(valid_link_avg_delays == valid_link_avg_delays.min())[0]
        )
        remote_id = self.fib[object_id][idx]
        self.req_timestamps[remote_id][
            (request.origin_id, request.seq_id)
        ] = self.env.now
        self.sendInterestPacket(remote_id, request)
