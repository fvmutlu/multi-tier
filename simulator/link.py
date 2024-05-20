# External package imports
import simpy as sp

# Builtin imports

# Internal imports


class Link(object):
    def __init__(self, env, link_cap, prop_delay):
        self.env = env
        self.link_cap = link_cap
        self.prop_delay = prop_delay
        self.pipe = sp.Store(env)

    def delayedDelivery(self, pkt):
        yield self.env.timeout(self.prop_delay)
        self.pipe.put(pkt)

    def put(self, pkt):
        self.env.process(self.delayedDelivery(pkt))

    def get(self):
        return self.pipe.get()


class VipLink(object):
    def __init__(self, env):
        self.env = env
        self.update_pipe = sp.Store(env)
        self.vip_pipe = sp.Store(env)

    def pushUpdate(self, pkt):
        self.update_pipe.put(pkt)

    def getUpdate(self):
        return self.update_pipe.get()

    def pushVips(self, pkt):
        self.vip_pipe.put(pkt)

    def getVips(self):
        return self.vip_pipe.get()


def getLink(env, link_cap, prop_delay):
    return Link(env, link_cap, prop_delay)


def getCtrlLink(env, fwd_pol, cache_pol, **kwargs):
    if fwd_pol in ["vip", "vip2", "vipsbw", "vipsbw2", "mvip"]:
        return VipLink(env)
    else:
        return None
