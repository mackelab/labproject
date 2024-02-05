import matplotlib.pyplot as plt
import os

# Load matplotlibrc file
STYLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "matplotlibrc")
)  # Necessary for GitHub Actions/ Calling from other directories
plt.style.use(STYLE_PATH)

PLOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots"))


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
