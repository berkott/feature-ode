"""Figure 1 (right panel): the MLP's learning time for each Hermite target against the feature ODE's prediction.

The points come from recorded_data/learning_time_prediction.json: one MLP run per target
and the feature ODE's predicted learning time. Those runs were made with an earlier version
of this code, so this script plots the recorded numbers rather than rerunning them. The loss
curves in the left panel of Figure 1 came from the same runs and are not included.

Usage:
    uv run python figures/learning_time_prediction.py
"""

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from plotting import EFFECTIVE_TIME_LABEL, apply_style, save_figure


RECORDED_DATA_PATH = Path(__file__).parent / "recorded_data" / "learning_time_prediction.json"

COLORS = ["xkcd:black", "xkcd:red", "xkcd:orange yellow", "xkcd:green", "xkcd:blue", "xkcd:purple"]
DEGREE_LABELS = ["Constant", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
TEXT_SIZE = 12
MARKER_SIZE = 8


def lighten(color, amount=0.5):
    """Lighten a color toward white (amount > 0) or darken it toward black (amount < 0)."""
    rgb = mcolors.to_rgb(mcolors.XKCD_COLORS.get(color, color))
    if amount >= 0:
        new_rgb = tuple(c + (1 - c) * amount for c in rgb)
    else:
        new_rgb = tuple(c * (1 + amount) for c in rgb)
    return tuple(min(max(x, 0), 1) for x in new_rgb)


def target_label(powers):
    """Return a LaTeX label such as $z_{0}z_{1}$ for a target given as {coordinate index: Hermite order}."""
    if not powers:
        return "1"
    return "$" + "".join(f"z_{{{i}}}" + (f"^{p}" if p > 1 else "") for i, p in powers.items()) + "$"


def load_learning_times():
    """Return degrees, MLP effective times, and feature ODE effective times for every recorded target."""
    data = json.loads(RECORDED_DATA_PATH.read_text())
    learning_rate = data["learning_rate"]
    targets = data["targets"]
    degrees = np.array([sum(t["powers"].values()) for t in targets])
    mlp_times = np.array([
        learning_rate * t["mlp_steps"] if t["mlp_steps"] is not None else np.nan for t in targets
    ])
    # The published figure halves the feature ODE's prediction for quadratic targets. This is a
    # visual adjustment: both axes are logarithmic, so it shifts the quadratic points down by log10(2).
    ode_times = np.array([
        learning_rate * t["ode_steps"] / (2 if degree == 2 else 1) if t["ode_steps"] is not None else np.nan
        for t, degree in zip(targets, degrees)
    ])
    labeled = [(t["powers"], t["label_location"]) for t in data["labeled_targets"]]
    return targets, degrees, mlp_times, ode_times, labeled


def main():
    apply_style()
    targets, degrees, mlp_times, ode_times, labeled = load_learning_times()
    is_labeled = np.array([any(t["powers"] == powers for powers, _ in labeled) for t in targets])

    fig, ax = plt.subplots(figsize=(3.6, 2.5), layout="constrained")
    unique_degrees = np.unique(degrees[~np.isnan(ode_times)])
    for degree_index, degree in enumerate(unique_degrees):
        color = COLORS[degree_index % len(COLORS)]
        mask = (degrees == degree) & ~np.isnan(ode_times) & ~np.isnan(mlp_times)
        for subset, alpha in ((mask & ~is_labeled, 0.35), (mask & is_labeled, 0.8)):
            if subset.any():
                ax.errorbar(mlp_times[subset], ode_times[subset], xerr=np.zeros(subset.sum()), fmt="o",
                            color=color, mec=color, mew=1.0, alpha=alpha, markersize=MARKER_SIZE,
                            capsize=4, linewidth=1.5)

    # Label the highlighted targets; label_location picks the corner (0-3) the text sits in.
    for powers, location in labeled:
        j = next(i for i, t in enumerate(targets) if t["powers"] == powers)
        x, y = mlp_times[j], ode_times[j]
        if np.isnan(x) or np.isnan(y):
            continue
        degree_index = int(np.where(unique_degrees == degrees[j])[0][0])
        color = COLORS[degree_index % len(COLORS)]
        ax.scatter(x, y, color=color, alpha=1, zorder=10, s=MARKER_SIZE ** 2)
        offset = 0.08
        ax.annotate(
            target_label(powers),
            (x * (1 + (offset if location < 2 else -offset)), y * (1 + (offset if location % 2 == 1 else -offset))),
            fontsize=13, color=lighten(color, -0.2), alpha=0.7,
            ha="left" if location < 2 else "right", va="bottom" if location % 2 == 1 else "top",
        )

    valid = ode_times[~np.isnan(ode_times)]
    reference = np.array([valid.min() * 0.2, valid.max() * 5])
    ax.plot(reference, reference, color="k", linestyle="--", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(np.nanmin(mlp_times) * 0.7, np.nanmax(mlp_times) * 1.5)
    ax.set_ylim(np.nanmin(mlp_times) * 0.7, np.nanmax(mlp_times) * 1.5)
    ax.set_ylabel("ODE-predicted eff. opt. time", fontsize=TEXT_SIZE)
    ax.set_xlabel(EFFECTIVE_TIME_LABEL, fontsize=TEXT_SIZE)
    ax.text(0.02, 0.97, "MLP @ Hermite Targets", ha="left", va="top", transform=ax.transAxes,
            fontsize=TEXT_SIZE, color="xkcd:dark gray")

    max_degree = int(unique_degrees.max())
    handles = [
        mlines.Line2D([], [], marker="o", color=COLORS[i], markerfacecolor=COLORS[i], markeredgecolor=COLORS[i],
                      linestyle="None", markersize=MARKER_SIZE, label=DEGREE_LABELS[i])
        for i in range(min(max_degree, len(DEGREE_LABELS)))
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, handletextpad=0.6, fontsize=TEXT_SIZE)

    save_figure(fig, "learning_time_prediction")


if __name__ == "__main__":
    main()
