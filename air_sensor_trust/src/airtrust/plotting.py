"""One chart style for every figure in the project."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config

# Categorical slots are used in this fixed order; a series keeps its colour
# across figures (persistence is always the muted grey, and so on).
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fd", "#9ec5f4", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b22f2f"])


def setup():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def save(fig, name):
    config.FIG.mkdir(parents=True, exist_ok=True)
    path = config.FIG / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def end_label(ax, x, y, text, color=INK2, dx=4):
    """Label a line at its last point, in ink rather than in the series colour."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", fontsize=9, color=color)
