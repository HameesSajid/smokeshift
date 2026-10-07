"""One chart style for every figure.

Colours follow the entity, not the chart: a model family keeps its colour in
every figure, and a variant with neighbour features is the same colour drawn
solid where the own-history variant is dashed.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#d03b3b")
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fd", "#9ec5f4", "#2a78d6", "#0d366b"])
SEQ_WARM = LinearSegmentedColormap.from_list("seq_warm", ["#fdf1ea", "#f5b08f", "#eb6834", "#8a2f0b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b22f2f"])

# (colour, line style) per model. Naive references are grey.
MODEL_STYLE = {
    "Persistence": (MUTED, ":"),
    "Seasonal naive": (MUTED, "-."),
    "Damped persistence": (MUTED, "--"),
    "XGBoost": (BLUE, "--"),
    "XGBoost + neighbours": (BLUE, "-"),
    "XGBoost (level label)": (BLUE, ":"),
    "Transformer": (ORANGE, "--"),
    "Transformer + neighbours": (ORANGE, "-"),
    "Chronos-Bolt": (AQUA, "--"),
    "Chronos-Bolt (long context)": (AQUA, "-."),
    "Chronos-2": (AQUA, "-"),
    "Chronos-2 (long context)": (AQUA, ":"),
    "TimesFM": (VIOLET, "-"),
    "TimesFM (long context)": (VIOLET, ":"),
    "Ensemble": (GREEN, "--"),
    "Ensemble + foundation": (GREEN, "-"),
}


def style(model):
    """Colour and line style for a model name."""
    return MODEL_STYLE.get(model, (INK2, "-"))


def setup():
    """Apply the shared matplotlib settings."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9,
    })


def save(fig, name):
    """Save a figure to outputs/figures and close it; returns the file name."""
    config.FIG.mkdir(parents=True, exist_ok=True)
    path = config.FIG / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name
