import torch
from metrics.sliced_wasserstein import sliced_wasserstein_distance
from metrics.gaussian_kl import gaussian_kl_divergence
from data import random_dataset
from plotting import plot_scaling_metric_dimensionality


def scaling_sliced_wasserstein_samples(dataset1, dataset2):
    distances = []
    dimensionality = list(range(1, 1000, 100))
    for d in dimensionality:
        distances.append(sliced_wasserstein_distance(dataset1[:, :d], dataset2[:, :d]))
    return dimensionality, distances


def scaling_kl_samples(dataset1, dataset2):
    distances = []
    dimensionality = list(range(2, 1000, 98))
    for d in dimensionality:
        distances.append(gaussian_kl_divergence(dataset1[:, :d], dataset2[:, :d]))
    # print(distances)
    return dimensionality, distances


class Experiment:
    def __init__(self):
        pass

    def run_experiment(self, dataset1, dataset2, experiment_fn):
        return experiment_fn(dataset1, dataset2)


if __name__ == "__main__":

    experiment = Experiment()

    experiment_results = {}
    # for exp_name in ['scaling_sliced_wasserstein_samples', 'scaling_kl_samples']:
    for exp_name in ["scaling_kl_samples"]:
        for i_d1, dataset1 in enumerate([random_dataset(n=100000, d=100)]):
            for i_d2, dataset2 in enumerate([random_dataset(n=100000, d=100)]):
                experiment_fn = globals()[exp_name]
                dimensionality, distances = experiment.run_experiment(
                    dataset1=dataset1, dataset2=dataset2, experiment_fn=experiment_fn
                )
                experiment_results[(exp_name, i_d1, i_d2)] = (dimensionality, distances)
    # single plot
    # plot_scaling_metric_dimensionality(dimensionality, distances, "Sliced Wasserstein", "Random Dataset")
    plot_scaling_metric_dimensionality(dimensionality, distances, "KL", "Random Dataset")
