import matplotlib.pyplot as plt
import os

plots_path = "plots/"


def plot_scaling_metric_dimensionality(dimensionality, distances, metric_name, dataset_name):
    plt.plot(dimensionality, distances)
    plt.xlabel("Dimensionality")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} with increasing dimensionality for {dataset_name}")
    plt.savefig(os.path.join(plots_path,
                        f"{metric_name.lower().replace(' ', '_')}_dimensionality_{dataset_name.lower().replace(' ', '_')}.png"))
    plt.close()
