from labproject.utils import set_seed, get_cfg
from labproject.data import get_dataset
from labproject.experiments import *


import time
import datetime
import os


def get_log_path(cfg):
    """
    Get the log path for the current experiment run.
    This log path is then used to save the numerical results of the experiment.
    Import this function in the run_{name}.py file and call it to get the log path.
    """

    # get datetime string
    now = datetime.datetime.now()
    if "exp_log_name" not in cfg:
        exp_log_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        exp_log_name = cfg.exp_log_name
        # add datetime to the name
        exp_log_name = exp_log_name + "_" + now.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(f"results/{cfg.running_user}/{exp_log_name}.pkl")
    return log_path


if __name__ == "__main__":

    print("Running experiments...")
    cfg = get_cfg()
    seed = cfg.seed

    set_seed(seed)
    print(f"Seed: {seed}")
    print(f"Experiments: {cfg.experiments}")
    print(f"Data: {cfg.data}")

    dataset_fn = get_dataset(cfg.data)

    for exp_name in cfg.experiments:
        experiment = globals()[exp_name]()
        time_start = time.time()
        dataset1 = dataset_fn(cfg.n, cfg.d)
        dataset2 = dataset_fn(cfg.n, cfg.d)

        output = experiment.run_experiment(dataset1=dataset1, dataset2=dataset2)
        time_end = time.time()
        print(f"Experiment {exp_name} finished in {time_end - time_start}")

        log_path = get_log_path(cfg)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        experiment.log_results(output, log_path)
        print(f"Numerical results saved to {log_path}")

        experiment.plot_experiment(*output, cfg.data)
        print(f"Plots saved to {cfg.data}.png")

    print("Finished running experiments.")
