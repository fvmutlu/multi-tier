# External package imports
import matplotlib.pyplot as plt


# Builtin package imports
import argparse

# Internal imports
from .helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment_name", type=str, default="sample")
parser.add_argument("-t", "--topology", type=str, default="abilene")
args = parser.parse_args()


experiment_name = args.experiment_name
topology = args.topology
config_path = "./sim_configs/ct_configs/" + experiment_name + "_config.json"
db_path = "./sim_outputs/" + experiment_name + "_" + topology + "_db.json"
db = getJsonDb(db_path)

x_label, curve_label = "", ""
x_variant, curve_variant = [], []
metric = ""

match experiment_name:
    case "st_cache_size":
        x_label, x_variant = "cache_capacities", [5, 6, 7, 8, 9, 10]
        metric = "delay"
    case "st_req_rate":
        x_label, x_variant = "request_rate", [5, 10, 15, 20, 25]
        curve_label, curve_variant = "cache_capacities", [(5,)]
        metric = "delay"
    case "st_zipf_param":
        x_label, x_variant = "zipf_param", [0.5, 0.625, 0.75, 0.875]
        curve_label, curve_variant = "cache_capacities", [(5,), (10,)]
        metric = "delay"
    case _:
        pass


fig, ax = plt.subplots()
legend = []

def plotter(filters, label, config_path=config_path, topology=topology, db=db, metric=metric, ax=ax):
    param_list = filterParamList(config_path, filters)
    param_hashes = getParamHashList(param_list)
    res = getDataFieldSumsAcrossEntries(topology, db, param_hashes, metric)
    ax.plot(x_variant, res, label=label)

vip_filters = [("cache_pol", "vip2", None)]
lfu_filters = [("cache_pol", "lfu", None)]
if curve_variant:
    vip_filters.append(())
    lfu_filters.append(())
    for curve in curve_variant:
        # VIP plot
        label = "VIP, " + curve_label + " " + str(curve)
        legend.append(label)
        vip_filters[1] = (curve_label, curve, None)
        plotter(vip_filters, label)
        # LFU plot
        label = "LFU, " + curve_label + " " + str(curve)
        legend.append(label)
        lfu_filters[1] = (curve_label, curve, None)
        plotter(lfu_filters, label)
else:
    # VIP plot
    label = "VIP"
    legend.append(label)
    plotter(vip_filters, label)
    # LFU plot
    label = "LFU"
    legend.append(label)
    plotter(lfu_filters, label)

ax.set_title(experiment_name + " on " + topology)
ax.set_xlabel(x_label)
ax.set_ylabel(metric)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(legend)

plt.savefig("./sim_outputs/" + experiment_name + "_" + topology + ".pdf")
