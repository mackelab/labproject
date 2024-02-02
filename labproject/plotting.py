import matplotlib.pyplot as plt
import os

# Load matplotlibrc file
plt.style.use("../matplotlibrc")

PLOT_PATH = "../plots/"


def plot_scaling_metric_dimensionality(dimensionality, distances, metric_name, dataset_name):
    plt.plot(dimensionality, distances)
    plt.xlabel("Dimensionality")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} with increasing dimensionality for {dataset_name}")
    plt.savefig(
        os.path.join(
            PLOT_PATH,
            f"{metric_name.lower().replace(' ', '_')}_dimensionality_{dataset_name.lower().replace(' ', '_')}.png",
        )
    )
    plt.close()
