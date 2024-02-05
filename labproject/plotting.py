import matplotlib.pyplot as plt
import os

# Load matplotlibrc file
STYLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "matplotlibrc")
)  # Necessary for GitHub Actions/ Calling from other directories
plt.style.use(STYLE_PATH)

PLOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots"))


def plot_scaling_metric_dimensionality(
    dimensionality, distances, metric_name, dataset_name, ax=None
):
    """Plot the scaling of a metric with increasing dimensionality."""
    if ax is None:
        plt.plot(dimensionality, distances, label=metric_name)
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
    else:
        ax.plot(dimensionality, distances, label=metric_name)
        ax.set_xlabel("Dimensionality")

        return ax


def plot_scaling_metric_sample_size(
    sample_size, distances, metric_name, dataset_name, ax=None, color=None, label=None
):
    """Plot the behavior of a metric with number of samples."""
    if ax is None:
        if color is not None:
            plt.plot(
                sample_size, distances, color=color, label=metric_name if label is None else label
            )
        else:
            plt.plot(sample_size, distances, label=metric_name)
        plt.xlabel("samples")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} with increasing sample size for {dataset_name}")
        plt.savefig(
            os.path.join(
                PLOT_PATH,
                f"{metric_name.lower().replace(' ', '_')}_sample_size_{dataset_name.lower().replace(' ', '_')}.png",
            )
        )
        plt.close()
    else:
        if color is not None:
            ax.plot(
                sample_size, distances, color=color, label=metric_name if label is None else label
            )
        else:
            ax.plot(sample_size, distances, label=metric_name if label is None else label)
        ax.set_xlabel("samples")

        return ax
