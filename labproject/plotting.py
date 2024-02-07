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
    body_face_color="#8189c9",
    body_edge_color="k",
    body_lw=0.25,
    body_alpha=1.0,
    body_zorder=0,
    whisker_color="k",
    whisker_alpha=1.0,
    whisker_lw=1,
    whisker_zorder=1,
    cap_color="k",
    cap_lw=0.25,
    cap_zorder=1,
    median_color="k",
    median_alpha=1.0,
    median_lw=1.5,
    median_bar_length=1.0,
    median_zorder=10,
    width=0.5,
    scatter_face_color="k",
    scatter_edge_color="none",
    scatter_radius=5,
    scatter_lw=0.25,
    scatter_alpha=0.35,
    scatter_width=0.5,
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
    b.set_color(body_edge_color)
    b.set_alpha(body_alpha)
    b.set_linewidth(body_lw)
    b.set_zorder(body_zorder)
    if fill_box:
        if vert:
            x0, x1 = b.get_xdata()[:2]
            y0, y1 = b.get_ydata()[1:3]
            r = Rectangle(
                [x0, y0],
                x1 - x0,
                y1 - y0,
                facecolor=body_face_color,
                alpha=body_alpha,
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
                facecolor=body_face_color,
                alpha=body_alpha,
                edgecolor="none",
            )
            ax.add_patch(r)

    # polish the whiskers
    for w in parts["whiskers"]:
        w.set_color(whisker_color)
        w.set_alpha(whisker_alpha)
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


def tiled_ticks(x0, x1, n_major_ticks, n_minor_ticks, offset):
    X = (
        np.tile(
            np.linspace(
                x0 - offset,
                x0 + offset,
                n_minor_ticks,
            ),
            n_major_ticks,
        ).reshape(n_major_ticks, n_minor_ticks)
        + np.linspace(x0, x1, n_major_ticks)[:, None]
    )
    return X


def find_min_max_recursive(z, zmin=float("inf"), zmax=float("-inf")):
    if isinstance(z, (int, float)):
        return np.nanmin([z, zmin]), np.nanmax([z, zmax])
    if isinstance(z, np.ndarray):
        if not np.any(z):
            return zmin, zmax
        if z.dtype != np.dtype("O"):
            _zmin, _zmax = np.nanmin(z), np.nanmax(z)
            zmin = np.nanmin([zmin, _zmin])
            zmax = np.nanmax([zmax, _zmax])
            return zmin, zmax
        else:
            return find_min_max_recursive(z.tolist(), zmin, zmax)
    if isinstance(z, (list, tuple)):
        if not z:
            return zmin, zmax
        for item in z:
            _zmin, _zmax = find_min_max_recursive(item, zmin, zmax)
            zmin = np.nanmin([zmin, _zmin])
            zmax = np.nanmax([zmax, _zmax])
        return zmin, zmax
    if isinstance(z, dict):
        if not z:
            return zmin, zmax
        for value in z.values():
            _zmin, _zmax = find_min_max_recursive(value, zmin, zmax)
            zmin = np.nanmin([zmin, _zmin])
            zmax = np.nanmax([zmax, _zmax])
        return zmin, zmax
    else:
        raise ValueError(f"{z}")


def get_lims(z, offset, min=None, max=None):
    zmin, zmax = find_min_max_recursive(z)

    if np.isinf(zmin) or np.isinf(zmax):
        return -1, 1

    _range = np.abs(zmax - zmin)
    zmin -= _range * offset
    zmax += _range * offset

    if min is not None:
        zmin = np.min((min, zmin))
    if max is not None:
        zmax = np.max((max, zmax))
    return zmin, zmax


def rm_spines(
    ax,
    spines=("top", "right", "bottom", "left"),
    visible=False,
    rm_xticks=True,
    rm_yticks=True,
):
    for spine in spines:
        ax.spines[spine].set_visible(visible)
    if ("top" in spines or "bottom" in spines) and rm_xticks:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position("none")
    if ("left" in spines or "right" in spines) and rm_yticks:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position("none")
    return ax


def place_violin(
    ax,
    x,
    y,
    body_face_color="#8189c9",
    body_edge_color="k",
    body_lw=0.25,
    body_alpha=1.0,
    body_zorder=0,
    whisker_color="k",
    whisker_alpha=1.0,
    whisker_lw=1,
    whisker_zorder=1,
    cap_color="k",
    cap_lw=0.25,
    cap_zorder=1,
    median_color="k",
    median_alpha=1.0,
    median_lw=1.5,
    median_bar_length=1.0,
    median_zorder=10,
    width=0.5,
    scatter_face_color="k",
    scatter_edge_color="none",
    scatter_radius=5,
    scatter_lw=0.25,
    scatter_alpha=0.35,
    scatter_width=0.5,
    scatter=True,
    scatter_zorder=3,
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
    if not np.any(y):
        return
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
    b.set_facecolor(body_face_color)
    b.set_edgecolor(body_edge_color)
    b.set_linewidth(body_lw)
    b.set_alpha(body_alpha)
    b.set_zorder(body_zorder)

    # Color the lines.
    if showextrema:
        parts["cbars"].set_color(whisker_color)
        parts["cbars"].set_alpha(whisker_alpha)
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
        parts["cmeans"].set_alpha(median_alpha)
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

    if showmedians:
        parts["cmedians"].set_color(median_color)
        parts["cmedians"].set_alpha(median_alpha)
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
                alpha=scatter_alpha,
                zorder=scatter_zorder,
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
