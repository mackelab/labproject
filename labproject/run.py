from labproject.experiments import scaling_sliced_wasserstein_samples
from labproject.plotting import plot_scaling_metric_dimensionality

print("Running experiments...")
dimensionality, distances = scaling_sliced_wasserstein_samples()
plot_scaling_metric_dimensionality(dimensionality, distances, "Sliced Wasserstein", "Random Dataset")
print("Finished running experiments.")