import matplotlib.pyplot as plt
import os

# Load matplotlibrc file
STYLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "matplotlibrc")
)  # Necessary for GitHub Actions/ Calling from other directories
plt.style.use(STYLE_PATH)

PLOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots"))


def cm2inch(cm, INCH=2.54):
    if isinstance(cm, tuple):
        return tuple(i / INCH for i in cm)
    else:
        return cm / INCH


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
    sample_size,
    distances,
    metric_name,
    dataset_name,
    ax=None,
    color=None,
    label=None,
    linestyle="-",
    **kwargs,
):
    """Plot the behavior of a metric with number of samples."""
    if ax is None:
        if color is not None:
            plt.plot(
                sample_size,
                distances,
                color=color,
                label=metric_name if label is None else label,
                linestyle=linestyle,
                **kwargs,
            )
        else:
            plt.plot(sample_size, distances, label=metric_name, linestyle=linestyle, **kwargs)
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
                sample_size,
                distances,
                label=metric_name if label is None else label,
                color=color,
                linestyle=linestyle,
                **kwargs,
            )
        else:
            ax.plot(sample_size, distances, label=metric_name if label is None else label, **kwargs)
        ax.set_xlabel("samples")
        ax.set_ylabel(metric_name, color=color)

        return ax
