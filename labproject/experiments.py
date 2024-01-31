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


def run_metric_on_datasets(dataset1, dataset2, metric):
    return metric(dataset1, dataset2)


class Experiment:
    def __init__(self):
        pass

    def run_experiment(self, dataset1, dataset2, experiment_fn):
        return experiment_fn(dataset1, dataset2)
