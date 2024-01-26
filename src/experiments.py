import torch
from data import random_dataset
from metrics.sliced_wasserstein import sliced_wasserstein_distance
from plotting import plot_scaling_metric_dimensionality

def scaling_sliced_wasserstein_samples():
    dataset1 = random_dataset(d=1000)
    dataset2 = random_dataset(d=1000)
    distances = []
    dimensionality = list(range(1, 1000, 100))
    for d in dimensionality:
        distances.append(sliced_wasserstein_distance(dataset1[:,:d], dataset2[:,:d]))

    return dimensionality, distances


if __name__ == "__main__":
    dimensionality, distances = scaling_sliced_wasserstein_samples()
    plot_scaling_metric_dimensionality(dimensionality, distances, "Sliced Wasserstein", "Random Dataset")