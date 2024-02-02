
from labproject.utils import set_seed, get_cfg
from labproject.data import get_dataset
from labproject.experiments import *


import time


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

        output = experiment.run_experiment(
                dataset1=dataset1, dataset2=dataset2
            )
        time_end = time.time()
        print(f"Experiment {exp_name} finished in {time_end - time_start}")
        experiment.plot_experiment(*output, cfg.data)
        


    print("Finished running experiments.")
