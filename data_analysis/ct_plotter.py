import matplotlib.pyplot as plt
from .helpers import *


def plotter(experiment_name):
    config_path = "./sim_configs/ct_configs/" + experiment_name + "_config.json"
    db_path = "./sim_outputs/" + experiment_name + "_abilene_db.json"
    db = getJsonDb(db_path)

    variant_a = []
    field_of_interest = "delay"

    # Modify this code block as more experiments are added
    variant_b = []

    if experiment_name not in ["st_cache_size", "st_req_rate", "st_zipf_param"]:
        field_of_interest = "delay"

    match experiment_name:
        case "st_cache_size":
            variant_label, variant_a = "cache size", [5, 6, 7, 8, 9, 10]
        case _:
            pass

    if experiment_name in ["st_cache_size", "st_req_rate", "st_zipf_param"]:
        vip_filters = [("cache_pol", "vip2", None)]
        lfu_filters = [("cache_pol", "lfu", None)]
    else:
        pass
    # End of code block

    vip_param_list = filterParamList(config_path, vip_filters)
    vip_param_hashes = getParamHashList(vip_param_list)
    lfu_param_list = filterParamList(config_path, lfu_filters)
    lfu_param_hashes = getParamHashList(lfu_param_list)

    vip_res = getDataFieldSumsAcrossEntries(
        "abilene", db, vip_param_hashes, field_of_interest
    )
    lfu_res = getDataFieldSumsAcrossEntries(
        "abilene", db, lfu_param_hashes, field_of_interest
    )

    fig, ax = plt.subplots()

    ax.plot(variant_a, vip_res, label="VIP")
    ax.plot(variant_a, lfu_res, label="LFU")

    ax.set_title(experiment_name)
    ax.set_xlabel(variant_label)
    ax.set_ylabel(field_of_interest)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(["VIP", "LFU"])

    plt.savefig("./sim_outputs/" + experiment_name + ".pdf")


#experiments = ["st_cache_size", "st_req_rate", "st_zipf_param"]
plotter("st_cache_size")
