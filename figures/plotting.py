"""Shared style, run loading, and panels for the paper's figures.

The style (`apply_style`) follows misc.rcsetup from dkarkada/expt-core.
Every figure script reads recorded runs from results/ and writes a PDF to figures/output/.
"""

import pickle
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "figures" / "output"

# === Shared plotting constants ===
XLIM_LEFT = 5.0
SMOOTHING = 0.5
MAX_PLOT_POINTS = 2000
THEORY_STYLE = {"color": "black", "linestyle": "--", "zorder": 20}
EFFECTIVE_TIME_LABEL = r"Eff. opt. time $\eta\cdot n_\mathrm{iter}$"

# === Palette: target-complexity colors (matplotlib tab10) ===
COLOR_LINEAR = "#d62728"  # red    — linear targets
COLOR_QUADRATIC = "#ff7f0e"  # orange — quadratic / 2-staircase
COLOR_CUBIC = "#2ca02c"  # green  — cubic / 3-staircase


def apply_style():
    """Set the matplotlib style used by every figure in the paper."""
    plt.rc("figure", dpi=120, facecolor=(1, 1, 1))
    plt.rc("font", family="stixgeneral", size=12)
    plt.rc("axes", facecolor=(1, .99, .95), titlesize=12)
    plt.rc("mathtext", fontset="cm")
    plt.rc("pdf", fonttype=42)  # TrueType
    plt.rc("ps", fonttype=42)


def make_shades(base_color, n, lighten_range=(0.0, 0.55)):
    """Return n shades of base_color, from the color itself toward white."""
    if n == 1:
        return [base_color]
    rgb = mcolors.to_rgb(base_color)
    shades = []
    for i in range(n):
        factor = lighten_range[0] + (lighten_range[1] - lighten_range[0]) * i / (n - 1)
        lightened = tuple(c + (1.0 - c) * factor for c in rgb)
        shades.append(mcolors.to_hex(lightened))
    return shades


# === Loading recorded runs ===

def find_run(configuration_name, experiment_name=None):
    """Return the directory of the latest run of a configuration (and experiment, for grid searches).

    Runs live in results/<timestamp>-<configuration name>/[<experiment name>/], as written by launch/.
    """
    batch_directories = sorted(RESULTS_DIRECTORY.glob(f"*-{configuration_name}"))
    if not batch_directories:
        raise FileNotFoundError(
            f"No run of '{configuration_name}' in {RESULTS_DIRECTORY}. "
            f"Run configurations/{configuration_name}.yaml first (see the README)."
        )
    run_directory = batch_directories[-1]
    return run_directory / experiment_name if experiment_name else run_directory


def load_results(configuration_name, experiment_name=None):
    """Load the results.pkl of the latest run of a configuration."""
    with open(find_run(configuration_name, experiment_name) / "results.pkl", "rb") as f:
        return pickle.load(f)


def save_figure(figure, name):
    """Save a figure as figures/output/<name>.pdf and close it."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"{name}.pdf"
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


# === Curve processing ===

def get_step_lr(results):
    """Effective time per recorded snapshot (handles record_interval)."""
    # Runs store their settings under "config".
    configuration = results["config"]
    record_interval = results.get("record_interval") or 1
    return configuration["learning_rate"] * record_interval


def compute_ema(values, smoothing):
    """Forward-backward exponential moving average, so smoothing does not shift curves in time."""
    if not values or smoothing == 0:
        return list(values)
    fwd = [values[0]]
    for v in values[1:]:
        fwd.append(smoothing * fwd[-1] + (1 - smoothing) * v)
    bwd = [0.0] * len(values)
    bwd[-1] = values[-1]
    for i in range(len(values) - 2, -1, -1):
        bwd[i] = smoothing * bwd[i + 1] + (1 - smoothing) * values[i]
    return [0.5 * (f + b) for f, b in zip(fwd, bwd)]


def aggregate_loss(results, smoothing):
    """Return effective times and the mean and standard deviation of the smoothed loss across init seeds."""
    step_lr = get_step_lr(results)
    init_losses = [np.array(v) for _, v in sorted(results["loss_history"].items())]
    min_len = min(len(x) for x in init_losses)
    arr = np.stack([x[:min_len] for x in init_losses], axis=0)
    smoothed = np.array([compute_ema(row.tolist(), smoothing) for row in arr])
    et = np.arange(min_len) * step_lr
    return et, smoothed.mean(axis=0), smoothed.std(axis=0)


def compute_max_time(results):
    """Return the later of the run's last effective time and the end of its feature ODE curve."""
    theory = results.get("theory") or {}
    any_seed = next(iter(results["loss_history"]))
    max_time = (len(results["loss_history"][any_seed]) - 1) * get_step_lr(results)
    if theory and "effective_times" in theory:
        max_time = max(max_time, theory["effective_times"][-1])
    return max_time


def drop_zero(t, *ys):
    """Drop the t = 0 point, which a log time axis cannot show."""
    if len(t) == 0 or t[0] > 0:
        return (t, *ys)
    return (t[1:], *(y[1:] for y in ys))


def downsample_log(t, *ys, n_points=MAX_PLOT_POINTS):
    """Keep about n_points samples, spaced evenly in log time."""
    n = len(t)
    if n <= n_points:
        return (t, *ys)
    log_idx = np.linspace(np.log(1), np.log(n), n_points)
    idx = np.unique(np.clip(np.round(np.exp(log_idx)).astype(int) - 1, 0, n - 1))
    return (t[idx], *(y[idx] for y in ys))


def format_log_axis(axis):
    """Put decade ticks with 10^k labels and unlabeled minor ticks on a log x-axis."""
    axis.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    axis.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    axis.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())


# === Panels ===

def plot_loss_with_ode(axis, results, color, model_label, right_limit):
    """Plot a model's smoothed loss (mean and one standard deviation across seeds) and the feature ODE's loss."""
    et, mean, std = aggregate_loss(results, SMOOTHING)
    theory = results.get("theory") or {}
    et_d, mean_d, std_d = drop_zero(et, mean, std)
    et_d, mean_d, std_d = downsample_log(et_d, mean_d, std_d)
    lo = np.clip(mean_d - std_d, 0, None)
    hi = mean_d + std_d
    axis.fill_between(et_d, lo, hi, color=color, alpha=0.25, linewidth=0)
    axis.plot(et_d, mean_d, color=color, label=model_label, zorder=11)

    if theory and "loss_history" in theory:
        t_th = np.array(theory["effective_times"])
        y_th = np.array(theory["loss_history"])
        t_th, y_th = drop_zero(t_th, y_th)
        t_th, y_th = downsample_log(t_th, y_th)
        axis.plot(t_th, y_th, label="ODE", **THEORY_STYLE)

    axis.set_xscale("log")
    format_log_axis(axis)
    axis.set_xlim(left=XLIM_LEFT, right=right_limit)
    axis.set_ylim(bottom=0)


def plot_neuron_variance_heatmap(axis, results, max_time):
    """Plot each neuron's preactivation variance over training, with neurons sorted by their final variance.

    Returns the mesh, for a colorbar, or None if the run did not record the full history.
    """
    neuron_z2 = results.get("neuron_z2_full_history") or {}
    if not neuron_z2:
        return None
    seeds = sorted(neuron_z2.keys())
    array = np.asarray(neuron_z2[seeds[0]], dtype=float)  # (T, N)
    if array.size == 0:
        return None
    num_steps, num_neurons = array.shape
    step_lr = get_step_lr(results)

    # Specialists at top
    order = np.argsort(-array[-1, :])
    array = array[:, order]

    # Downsample time
    if num_steps > MAX_PLOT_POINTS:
        step = int(np.ceil(num_steps / MAX_PLOT_POINTS))
        idx = np.arange(0, num_steps, step)
    else:
        idx = np.arange(num_steps)
    array_plot = array[idx, :]
    et = idx * step_lr
    valid = et > 0
    et = et[valid]
    array_plot = array_plot[valid, :]

    # Log-spaced cell edges in x
    log_et = np.log(et)
    log_edges = np.concatenate([
        [2 * log_et[0] - log_et[1]],
        0.5 * (log_et[:-1] + log_et[1:]),
        [2 * log_et[-1] - log_et[-2]],
    ])
    et_edges = np.exp(log_edges)
    neuron_edges = np.arange(num_neurons + 1)

    # LogNorm color, floor at 0.5 * smallest positive
    pos = array_plot > 0
    floor = float(array_plot[pos].min()) * 0.5 if pos.any() else 1e-12
    array_plot = np.clip(array_plot, floor, None)

    mesh = axis.pcolormesh(
        et_edges, neuron_edges, array_plot.T,
        norm=mcolors.LogNorm(vmin=floor, vmax=float(array_plot.max())),
        cmap="viridis",
        shading="flat",
        rasterized=True,
    )
    axis.set_xscale("log")
    format_log_axis(axis)
    axis.set_xlim(left=XLIM_LEFT, right=max_time)
    axis.set_ylim(0, num_neurons)
    axis.invert_yaxis()
    axis.set_xlabel(EFFECTIVE_TIME_LABEL)
    axis.set_ylabel("Neuron rank")
    return mesh
