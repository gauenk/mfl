"""

This file is a script to show the trimmed mean attack
on (i) the mean, (ii) the coordiante-wise median,
and (iii) the multi-dimensional median

"""

# -- misc --
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import torch as th
import numpy as np

# -- configure --
from easydict import EasyDict as edict

# -- caching results --
import cache_io

# -- this project --
import mfl

# -- federated learning sim --
import ray
import blades
from blades.simulator import Simulator
from blades.datasets import CIFAR10
from blades.models.cifar10 import CCTNet
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def run_exp(cfg):
    ray.init(include_dashboard=False, num_gpus=cfg.num_gpus)
    agg = mfl.medians.TukeyMedian()
    attack = mfl.attacks.TrimmedMeanAttack(cfg.mode,cfg.total,cfg.percent)
    cifar10 = CIFAR10(num_clients=cfg.total, iid=True)

    # configuration parameters
    conf_args = {
        "dataset": cifar10,
        "aggregator": agg,  # defense: robust aggregation
        "aggregator_kws": None,
        "num_byzantine": cfg.num_byzantine,  # number of byzantine input
        "use_cuda": True,
        "attack": attack,  # attack strategy
        "attack_kws": None,
        "num_actors": cfg.total,  # number of training actors
        "gpu_per_actor": 0.19,
        "log_path": cfg.log_dir,
        "seed": cfg.seed,  # reproducibility
    }
    simulator = Simulator(**conf_args)

    model = CCTNet()
    client_opt = th.optim.Adam(model.parameters(), lr=0.1)
    client_lr_scheduler = th.optim.lr_scheduler.MultiStepLR(
        client_opt, milestones=[150, 300, 500], gamma=0.5
    )
    # runtime parameters
    run_args = {
        "model": model,  # global model
        "server_optimizer": 'SGD', #server_opt, server optimizer
        "client_optimizer": client_opt,  # client optimizer
        "loss": "crossentropy",  # loss funcstion
        "global_rounds": cfg.global_round,  # number of global rounds
        "local_steps": cfg.local_round,  # number of seps "client_lr": 0.1,  # learning rateteps per round
        "server_lr": 1.0,
        # "client_lr": 0.1,  # learning rate
        "validate_interval": 10,
        "client_lr_scheduler": client_lr_scheduler,
    }
    simulator.run(**run_args)


def default_config():
    cfg = edict()
    cfg.device = "cuda:0"
    cfg.seed = 123
    cfg.global_round = 6
    cfg.local_round = 3
    cfg.log_dir = "./output/trimmed_example"
    cfg.num_gpus = 1
    cfg.num_byzantine = 0
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "trimmed_example"
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- get mesh --
    mode = ["full","partial"]
    percent = [0.1,0.2]
    total = [30,40]
    exp_lists = {"mode":mode,"percent":percent,"total":total}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg = default_config()
    cache_io.append_configs(exps,cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records_)

if __name__ == "__main__":
    main()
