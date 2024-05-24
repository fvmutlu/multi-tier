# External package imports

# Builtin imports

# Internal imports
from .caching_policies import *
from .forwarding_policies import *


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
        self.lfuInit(num_objects)


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

class LRTPAWLFUNode(LeastResponseTimeNode, PAWLFUNode):
    def __init__(self, env, node_id, num_objects, pen_weight):
        super().__init__(env, node_id)
        self.lfuInit(num_objects, pen_weight)
