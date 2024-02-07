import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import seaborn as sns
import numpy as np

####
# global plot params
###

# Load matplotlibrc file
STYLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "matplotlibrc")
)  # Necessary for GitHub Actions/ Calling from other directories
plt.style.use(STYLE_PATH)

PLOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots"))

# color items -- example usage in application2.ipynb


def generate_palette(hex_color, n_colors=5, saturation="light"):
    if saturation == "light":
        palette = sns.light_palette(hex_color, n_colors=n_colors, as_cmap=False)
    elif saturation == "dark":
        palette = sns.dark_palette(hex_color, n_colors=n_colors, as_cmap=False)
    return palette


color_dict = {"wasserstein": "#cc241d", "mmd": "#eebd35", "c2st": "#458588", "fid": "#8ec07c"}


####
# plotting functions
###


def cm2inch(cm, INCH=2.54):
    if isinstance(cm, tuple):
        return tuple(i / INCH for i in cm)
    else:
        return cm / INCH


def plot_scaling_metric_dimensionality(
    dim_sizes,
    distances,
    errors,
    metric_name,
    dataset_name,
    ax=None,
    label=None,
    **kwargs,
):
    """Plot the scaling of a metric with increasing dimensionality."""
    if ax is None:
        plt.plot(
            dim_sizes,
            distances,
            label=metric_name if label is None else label,
            **kwargs,
        )
        plt.fill_between(
            dim_sizes,
            distances - errors,
            distances + errors,
            alpha=0.2,
            color="black" if kwargs.get("color") is None else kwargs.get("color"),
        )
        plt.xlabel("Dimension")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} with increasing dimensionality size for {dataset_name}")
        plt.savefig(
            os.path.join(
                PLOT_PATH,
                f"{metric_name.lower().replace(' ', '_')}_dimensionality_size_{dataset_name.lower().replace(' ', '_')}.png",
            )
        )
        plt.close()
    else:
        ax.plot(
            dim_sizes,
            distances,
            label=metric_name if label is None else label,
            **kwargs,
        )
        ax.fill_between(
            dim_sizes,
            distances - errors,
            distances + errors,
            alpha=0.2,
            color="black" if kwargs.get("color") is None else kwargs.get("color"),
        )
        ax.set_xlabel("samples")
        ax.set_ylabel(
            metric_name, color="black" if kwargs.get("color") is None else kwargs.get("color")
        )
        return ax


def plot_scaling_metric_sample_size(
    sample_size,
    distances,
    errors,
    metric_name,
    dataset_name,
    ax=None,
    label=None,
    **kwargs,
):
    """Plot the behavior of a metric with number of samples."""
    if ax is None:
        plt.plot(
            sample_size,
            distances,
            label=metric_name if label is None else label,
            **kwargs,
        )
        plt.fill_between(
            sample_size,
            distances - errors,
            distances + errors,
            alpha=0.2,
            color="black" if kwargs.get("color") is None else kwargs.get("color"),
        )
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
        ax.plot(
            sample_size,
            distances,
            label=metric_name if label is None else label,
            **kwargs,
        )
        ax.fill_between(
            sample_size,
            distances - errors,
            distances + errors,
            alpha=0.2,
            color="black" if kwargs.get("color") is None else kwargs.get("color"),
        )
        ax.set_xlabel("samples")
        ax.set_ylabel(
            metric_name, color="black" if kwargs.get("color") is None else kwargs.get("color")
        )
        return ax


def place_boxplot(
    ax,
    x,
    y,
    box_face_color="#77b5d9",
    box_edge_color="#77b5d9",
    box_lw=0.25,
    box_alpha=1.0,
    box_zorder=0,
    whisker_color="#77b5d9",
    whisker_lw=1,
    whisker_zorder=1,
    cap_color="#000000",
    cap_lw=0.25,
    cap_zorder=1,
    median_color="#ffffff",
    median_alpha=1.0,
    median_lw=2,
    median_bar_length=0.75,
    median_zorder=2,
    width=0.5,
    scatter_face_color="#000000",
    scatter_edge_color="#000000",
    scatter_radius=2,
    scatter_lw=0.25,
    scatter_alpha=1,
    scatter_width=0.25,
    scatter=True,
    scatter_zorder=3,
    fill_box=True,
    showcaps=False,
    showfliers=False,
    whis=(0, 100),
    vert=True,
):
    """
    Example:
        X = [1, 2]
        Y = [np.random.normal(0.75, 0.12, size=50), np.random.normal(0.8, 0.20, size=25)]
        fig, ax = plt.subplots(figsize=[1, 1])
        for (x, y) in zip(X, Y):
            place_boxplot(ax, x, y)
    """
    parts = ax.boxplot(
        y,
        positions=[x],
        widths=width,
        showcaps=showcaps,
        showfliers=showfliers,
        whis=whis,
        vert=vert,
    )

    # polish the body
    b = parts["boxes"][0]
    b.set_color(box_edge_color)
    b.set_alpha(box_alpha)
    b.set_linewidth(box_lw)
    b.set_zorder(box_zorder)
    if fill_box:
        if vert:
            x0, x1 = b.get_xdata()[:2]
            y0, y1 = b.get_ydata()[1:3]
            r = Rectangle(
                [x0, y0],
                x1 - x0,
                y1 - y0,
                facecolor=box_face_color,
                alpha=box_alpha,
                edgecolor="none",
            )
            ax.add_patch(r)
        else:
            x0, x1 = b.get_xdata()[1:3]
            y0, y1 = b.get_ydata()[:2]
            r = Rectangle(
                [x0, y0],
                x1 - x0,
                y1 - y0,
                facecolor=box_face_color,
                alpha=box_alpha,
                edgecolor="none",
            )
            ax.add_patch(r)

    # polish the whiskers
    for w in parts["whiskers"]:
        w.set_color(whisker_color)
        w.set_linewidth(whisker_lw)
        w.set_zorder(whisker_zorder)

    # polish the caps
    for c in parts["caps"]:
        c.set_color(cap_color)
        c.set_linewidth(cap_lw)
        c.set_zorder(cap_zorder)

    # polish the median
    m = parts["medians"][0]
    m.set_color(median_color)
    m.set_linewidth(median_lw)
    m.set_alpha(median_alpha)
    m.set_zorder(median_zorder)
    if median_bar_length is not None:
        if vert:
            x0, x1 = m.get_xdata()
            m.set_xdata(
                [
                    x0 - 1 / 2 * (median_bar_length - 1) * (x1 - x0),
                    x1 + 1 / 2 * (median_bar_length - 1) * (x1 - x0),
                ]
            )
        else:
            y0, y1 = m.get_ydata()
            m.set_ydata(
                [
                    y0 - 1 / 2 * (median_bar_length - 1) * (y1 - y0),
                    y1 + 1 / 2 * (median_bar_length - 1) * (y1 - y0),
                ]
            )

    # scatter data
    if scatter:
        if vert:
            x0, x1 = b.get_xdata()[:2]
            ax.scatter(
                np.random.uniform(
                    x0 + 1 / 2 * (1 - scatter_width) * (x1 - x0),
                    x1 - 1 / 2 * (1 - scatter_width) * (x1 - x0),
                    size=len(y),
                ),
                y,
                facecolor=scatter_face_color,
                edgecolor=scatter_edge_color,
                s=scatter_radius,
                linewidth=scatter_lw,
                zorder=scatter_zorder,
                alpha=scatter_alpha,
            )
        else:
            y0, y1 = b.get_ydata()[:2]
            ax.scatter(
                y,
                np.random.uniform(
                    y0 + 1 / 2 * (1 - scatter_width) * (y1 - y0),
                    y1 - 1 / 2 * (1 - scatter_width) * (y1 - y0),
                    size=len(y),
                ),
                facecolor=scatter_face_color,
                edgecolor=scatter_edge_color,
                s=scatter_radius,
                linewidth=scatter_lw,
                zorder=scatter_zorder,
                alpha=scatter_alpha,
            )


def place_violin(
    ax,
    x,
    y,
    violin_face_color="#77b5d9",
    violin_edge_color="k",
    violin_lw=0.25,
    violin_alpha=1.0,
    whisker_color="#77b5d9",
    whisker_lw=1,
    whisker_zorder=1,
    cap_color="none",
    cap_lw=0.25,
    cap_zorder=1,
    median_color="#ffffff",
    median_lw=2,
    median_bar_length=0.75,
    median_zorder=2,
    width=0.5,
    scatter_face_color="#000000",
    scatter_edge_color="#000000",
    scatter_radius=2,
    scatter_lw=0.25,
    scatter_alpha=1,
    scatter_width=0.25,
    scatter=True,
    showextrema=True,
    showmedians=True,
    showmeans=False,
    vert=True,
):
    """
    Example:
        X = [1, 2]
        Y = [np.random.normal(0.75, 0.12, size=50), np.random.normal(0.8, 0.20, size=25)]
        fig, ax = plt.subplots(figsize=[1, 1])
        for (x, y) in zip(X, Y):
            place_violin(ax, x, y)
    """
    parts = ax.violinplot(
        y,
        positions=[x],
        widths=width,
        showmedians=showmedians,
        showmeans=showmeans,
        showextrema=showextrema,
        vert=vert,
    )
    # Color the bodies.
    b = parts["bodies"][0]
    b.set_facecolor(violin_face_color)
    b.set_edgecolor(violin_edge_color)
    b.set_linewidth(violin_lw)
    b.set_alpha(violin_alpha)
    b.set_zorder(0)

    # Color the lines.
    if showextrema:
        parts["cbars"].set_color(whisker_color)
        parts["cbars"].set_linewidth(whisker_lw)
        parts["cbars"].set_zorder(whisker_zorder)
        parts["cmaxes"].set_color(cap_color)
        parts["cmaxes"].set_linewidth(cap_lw)
        parts["cmaxes"].set_zorder(cap_zorder)
        parts["cmins"].set_color(cap_color)
        parts["cmins"].set_linewidth(cap_lw)
        parts["cmins"].set_zorder(cap_zorder)

    if showmeans:
        parts["cmeans"].set_color(median_color)
        parts["cmeans"].set_linewidth(median_lw)
        parts["cmeans"].set_zorder(median_zorder)
        if median_bar_length is not None:
            if vert:
                (_, y0), (_, y1) = parts["cmeans"].get_segments()[0]
                parts["cmeans"].set_segments(
                    [
                        [
                            [x - median_bar_length * width / 2, y0],
                            [x + median_bar_length * width / 2, y1],
                        ]
                    ]
                )
            else:
                (x0, _), (x1, _) = parts["cmeans"].get_segments()[0]
                parts["cmeans"].set_segments(
                    [
                        [
                            [x0, x - median_bar_length * width / 2],
                            [x1, x + median_bar_length * width / 2],
                        ]
                    ]
                )

    if "cmedians" in parts:
        parts["cmedians"].set_color(median_color)
        parts["cmedians"].set_linewidth(median_lw)
        parts["cmedians"].set_zorder(median_zorder)
        if median_bar_length is not None:
            if vert:
                (_, y0), (_, y1) = parts["cmedians"].get_segments()[0]
                parts["cmedians"].set_segments(
                    [
                        [
                            [x - median_bar_length * width / 2, y0],
                            [x + median_bar_length * width / 2, y1],
                        ]
                    ]
                )
            else:
                (x0, _), (x1, _) = parts["cmedians"].get_segments()[0]
                parts["cmedians"].set_segments(
                    [
                        [
                            [x0, x - median_bar_length * width / 2],
                            [x1, x + median_bar_length * width / 2],
                        ]
                    ]
                )

    # scatter data
    if scatter:
        if vert:
            ax.scatter(
                np.random.uniform(
                    x - width / 2 + 1 / 2 * (1 - scatter_width) * width,
                    x + width / 2 - 1 / 2 * (1 - scatter_width) * width,
                    size=len(y),
                ),
                y,
                facecolor=scatter_face_color,
                edgecolor=scatter_edge_color,
                s=scatter_radius,
                linewidth=scatter_lw,
                zorder=5,
                alpha=scatter_alpha,
            )
        else:
            ax.scatter(
                y,
                np.random.uniform(
                    x - width / 2 + 1 / 2 * (1 - scatter_width) * width,
                    x + width / 2 - 1 / 2 * (1 - scatter_width) * width,
                    size=len(y),
                ),
                facecolor=scatter_face_color,
                edgecolor=scatter_edge_color,
                s=scatter_radius,
                linewidth=scatter_lw,
                zorder=5,
                alpha=scatter_alpha,
            )
