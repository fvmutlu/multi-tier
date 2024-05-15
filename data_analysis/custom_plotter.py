# External package imports
import matplotlib.pyplot as plt

# Builtin package imports
import argparse

# Internal imports
from .helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment_name", type=str, default="sample")
parser.add_argument("-t", "--topology", type=str, default="abilene")
parser.add_argument("--x_label", type=str)
parser.add_argument("--curve_label", type=str, default="")
parser.add_argument("--metric", type=str)
config_mutex_group = parser.add_mutually_exclusive_group()
config_mutex_group.add_argument("--config_local", type=str, dest="config_path")
config_mutex_group.add_argument("--config_url", type=str, dest="config_path")
parser.set_defaults(config_path="./sim_configs/sample_config.json")
args = parser.parse_args()


experiment_name = args.experiment_name
topology = args.topology

config_path = args.config_path
test_config = getTestConfig(config_path)

db_path = "./sim_outputs/" + experiment_name + "_" + topology + "_db.json"
db = getJsonDb(db_path)

x_label, curve_label = args.x_label, args.curve_label
metric = args.metric

x_variant = test_config[x_label]
if isinstance(x_variant[0], list):
    x_variant = list(map(tuple, x_variant))

if curve_label:
    curve_variant = test_config[curve_label]
    if isinstance(curve_variant[0], list):
        curve_variant = list(map(tuple, curve_variant))
else:
    curve_variant = []

fig, ax = plt.subplots()
legend = []


def plotter(
    filters,
    label,
    config_path=config_path,
    topology=topology,
    db=db,
    metric=metric,
    ax=ax,
    x_variant=x_variant,
):
    param_list = filterParamList(config_path, filters)
    res = avgDataFieldSumsAcrossSeeds(topology, db, param_list, metric)
    ax.plot(list(map(str, x_variant)), res, label=label)


for cache_pol in test_config["cache_pol"]:
    filters = [("cache_pol", cache_pol, None)]
    if curve_variant:
        filters.append(())
        for curve in curve_variant:
            label = cache_pol + ", " + curve_label + " " + str(curve)
            legend.append(label)
            filters[1] = (curve_label, curve, None)
            plotter(filters, label)
    else:
        label = cache_pol
        legend.append(label)
        plotter(filters, label)

ax.set_title(experiment_name + " on " + topology)
ax.set_xlabel(x_label)
ax.set_ylabel(metric)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(legend)

plt.savefig("./sim_outputs/" + experiment_name + "_" + topology + ".pdf")
