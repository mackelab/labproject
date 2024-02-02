from labproject.experiments import *
from labproject.plotting import *
from labproject.utils import set_seed

import time

if __name__ == "__main__":

    print("Running experiments...")
    set_seed(42)
    experiment = Experiment()

    experiment_results = {}
    # for exp_name in ['scaling_sliced_wasserstein_samples', 'scaling_kl_samples']:
    for exp_name in ["scaling_kl_samples"]:
        time_start = time.time()
        for i_d, dataset_pair in enumerate(
            [
                [random_dataset(n=100000, d=100), random_dataset(n=100000, d=100)],
            ]
        ):
            dataset1, dataset2 = dataset_pair
            experiment_fn = globals()[exp_name]
            dimensionality, distances = experiment.run_experiment(
                dataset1=dataset1, dataset2=dataset2, experiment_fn=experiment_fn
            )
            experiment_results[(exp_name, i_d)] = (dimensionality, distances)
        time_end = time.time()
        print(f"Experiment {exp_name} finished in {time_end - time_start}")

    # single plot
    # plot_scaling_metric_dimensionality(dimensionality, distances, "Sliced Wasserstein", "Random Dataset")
    plot_scaling_metric_dimensionality(dimensionality, distances, "KL", "Random Dataset")

    print("Finished running experiments.")
