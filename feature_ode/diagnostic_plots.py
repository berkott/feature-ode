"""Plot functions for ML research experiment results using matplotlib."""

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# === EMA helper ===

def compute_ema(values: List[float], smoothing_factor: float) -> List[float]:
    """Compute forward-backward EMA (zero-lag smoothing).

    Runs EMA forward, then backward, and averages. The forward pass lags
    behind and the backward pass leads ahead, so their average is centered.

    Args:
        values: Raw values.
        smoothing_factor: Smoothing weight in [0, 1]. 0 = no smoothing, 1 = constant.

    Returns:
        List of smoothed values, same length as input.
    """
    if not values:
        return []
    # Forward pass
    fwd = [values[0]]
    for value in values[1:]:
        fwd.append(smoothing_factor * fwd[-1] + (1 - smoothing_factor) * value)
    # Backward pass
    bwd = [0.0] * len(values)
    bwd[-1] = values[-1]
    for i in range(len(values) - 2, -1, -1):
        bwd[i] = smoothing_factor * bwd[i + 1] + (1 - smoothing_factor) * values[i]
    # Average
    return [0.5 * (f + b) for f, b in zip(fwd, bwd)]


# === Downsampling helper ===

def downsample(
    x_values: List[float],
    y_values: List[float],
    max_plot_points: int,
) -> Tuple[List[float], List[float]]:
    """Downsample to at most max_plot_points points, preserving real x-axis values."""
    num_points = len(x_values)
    if num_points <= max_plot_points:
        return x_values, y_values
    step = math.ceil(num_points / max_plot_points)
    indices = list(range(0, num_points, step))
    return [x_values[i] for i in indices], [y_values[i] for i in indices]


# === Color palette ===

SEED_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]


def get_seed_color(index: int) -> str:
    return SEED_COLORS[index % len(SEED_COLORS)]


# === Common figure setup ===

def create_figure():
    """Create a matplotlib figure with consistent styling."""
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.grid(True, alpha=0.3, linewidth=0.5)
    axis.set_xlabel("Effective time (learning rate × epoch)")
    return figure, axis


def save_figure(figure, output_path: Path):
    """Save figure as PDF and close."""
    figure.tight_layout()
    figure.savefig(str(output_path), format="pdf", bbox_inches="tight", dpi=150)
    plt.close(figure)


# === Plot functions ===

def plot_loss_curves(
    loss_history: Dict[int, List[float]],
    output_path: Path,
    learning_rate: float,
    max_plot_points: int,
    smoothing_factor: float = 0.5,
    theory_loss_history: Optional[List[float]] = None,
    theory_effective_times: Optional[List[float]] = None,
    log_scale: bool = False,
    y_max: Optional[float] = None,
) -> None:
    """Plot training loss curves with EMA overlay, one line per init seed.

    Args:
        loss_history: Maps init_seed -> list of loss values per epoch.
        output_path: Path to save the plot (pdf).
        learning_rate: Base learning rate, used for effective time x-axis.
        max_plot_points: Maximum number of points plotted on the x-axis.
        smoothing_factor: EMA smoothing factor.
        theory_loss_history: Optional feature ODE theory loss curve.
        theory_effective_times: Effective time values for theory points.
        log_scale: If True, use log scale on y-axis.
        y_max: If set, clip y-axis at this value.
    """
    figure, axis = create_figure()
    sorted_seeds = sorted(loss_history.keys())

    for i, init_seed in enumerate(sorted_seeds):
        color = get_seed_color(i)
        losses = loss_history[init_seed]
        effective_time = [learning_rate * epoch for epoch in range(len(losses))]
        ema_values = compute_ema(losses, smoothing_factor)

        x_raw, y_raw = downsample(effective_time, losses, max_plot_points)
        x_ema, y_ema = downsample(effective_time, ema_values, max_plot_points)

        axis.plot(x_raw, y_raw, color=color, alpha=0.3, linewidth=0.5)
        axis.plot(x_ema, y_ema, color=color, linewidth=1.5, label=f"seed {init_seed}")

    if theory_loss_history is not None and theory_effective_times is not None:
        x_theory, y_theory = downsample(theory_effective_times, theory_loss_history, max_plot_points)
        axis.plot(x_theory, y_theory, color="black", linestyle="--", linewidth=2, label="theory")

    if log_scale:
        axis.set_yscale("log")
    if y_max is not None:
        axis.set_ylim(bottom=0, top=y_max)
    axis.set_ylabel("Loss")
    axis.set_title("Training loss")
    axis.legend()
    save_figure(figure, output_path)


def plot_all_a_elements(
    a_history: Dict[int, Dict[Tuple[int, int], List[float]]],
    a_elements: Tuple[Tuple[int, int], ...],
    output_path: Path,
    learning_rate: float,
    max_plot_points: int,
    smoothing_factor: float = 0.5,
    theory_a_histories: Optional[Dict[Tuple[int, int], List[float]]] = None,
    theory_effective_times: Optional[List[float]] = None,
) -> None:
    """Plot all A matrix elements on one figure.

    Each (row, col) element gets its own color. Seeds for the same element
    share the same color (raw = faint, EMA = solid).

    Args:
        a_history: Maps init_seed -> {(row, col): [values]}.
        a_elements: Tuple of (row, col) pairs to plot.
        output_path: Path to save the plot (pdf).
        learning_rate: Base learning rate, used for effective time x-axis.
        max_plot_points: Maximum number of points plotted on the x-axis.
        smoothing_factor: EMA smoothing factor.
        theory_a_histories: Optional {(row, col): [theory_values]} from the feature ODE.
        theory_effective_times: Effective time values for theory points.
    """
    figure, axis = create_figure()
    sorted_seeds = sorted(a_history.keys())

    for element_index, (row, col) in enumerate(a_elements):
        color = get_seed_color(element_index)

        for seed_index, init_seed in enumerate(sorted_seeds):
            values = a_history[init_seed][(row, col)]
            effective_time = [learning_rate * epoch for epoch in range(1, len(values) + 1)]
            ema_values = compute_ema(values, smoothing_factor)

            x_raw, y_raw = downsample(effective_time, values, max_plot_points)
            x_ema, y_ema = downsample(effective_time, ema_values, max_plot_points)

            axis.plot(x_raw, y_raw, color=color, alpha=0.15, linewidth=0.5)
            label = f"A[{row},{col}]" if seed_index == 0 else None
            axis.plot(x_ema, y_ema, color=color, linewidth=1.5, label=label)

    # === Theory curves (same color as experimental, dashed) ===
    if theory_a_histories is not None and theory_effective_times is not None:
        for element_index, (row, col) in enumerate(a_elements):
            if (row, col) in theory_a_histories:
                color = get_seed_color(element_index)
                theory_values = theory_a_histories[(row, col)]
                x_theory, y_theory = downsample(theory_effective_times, theory_values, max_plot_points)
                axis.plot(x_theory, y_theory, color=color, linestyle="--", linewidth=2,
                          label=f"theory A[{row},{col}]")

    axis.set_ylabel("A element value")
    axis.set_title("A matrix elements over training")
    axis.legend()
    save_figure(figure, output_path)


def plot_neuron_z2_histogram(
    neuron_z2_init: Dict[int, List[float]],
    neuron_z2_final: Dict[int, List[float]],
    output_path: Path,
    num_bins: int = 60,
) -> None:
    """Histogram of per-neuron preactivation variance <z_k^2> at init vs final.

    Log-x density histogram. One curve per (seed, time) pair: dotted = init,
    solid = final. Seeds share colors for the same time-step distinction.
    """
    figure, axis = plt.subplots(figsize=(8, 5))
    sorted_seeds = sorted(neuron_z2_init.keys())

    init_arrays = [np.asarray(neuron_z2_init[s], dtype=float) for s in sorted_seeds]
    final_arrays = [np.asarray(neuron_z2_final[s], dtype=float) for s in sorted_seeds]
    all_values = np.concatenate(init_arrays + final_arrays)
    eps = max(1e-12, float(all_values[all_values > 0].min()) * 0.5) if (all_values > 0).any() else 1e-12
    lo = max(eps, all_values[all_values > 0].min() if (all_values > 0).any() else eps)
    hi = max(all_values.max(), lo * 10)
    bins = np.logspace(np.log10(lo), np.log10(hi), num_bins)

    for index, init_seed in enumerate(sorted_seeds):
        color = get_seed_color(index)
        clipped_init = np.clip(init_arrays[index], eps, None)
        clipped_final = np.clip(final_arrays[index], eps, None)
        axis.hist(clipped_init, bins=bins, density=True, histtype="step",
                  linewidth=1.0, linestyle=":", color=color,
                  label=f"seed {init_seed} init")
        axis.hist(clipped_final, bins=bins, density=True, histtype="step",
                  linewidth=1.5, color=color,
                  label=f"seed {init_seed} final")

    axis.set_xscale("log")
    axis.set_xlabel(r"per-neuron preactivation variance $\langle z_k^2 \rangle$")
    axis.set_ylabel("density")
    axis.set_title("per-neuron preactivation variance distribution")
    axis.grid(True, alpha=0.3, linewidth=0.5)
    axis.legend()
    save_figure(figure, output_path)


def plot_neuron_z2_percentiles_over_time(
    neuron_z2_percentiles_history: Dict[int, List[List[float]]],
    percentile_probabilities: List[float],
    output_path: Path,
    learning_rate: float,
    max_plot_points: int,
) -> None:
    """Percentile traces of <z_k^2> across neurons over training time.

    Shows the first init seed only. The 0/100, 20/80, and 50 percentiles are
    each highlighted in a distinct color; all other percentiles share a
    default gray. No background shading.
    """
    figure, axis = create_figure()
    sorted_seeds = sorted(neuron_z2_percentiles_history.keys())
    if not sorted_seeds:
        save_figure(figure, output_path)
        return
    init_seed = sorted_seeds[0]
    per_step = neuron_z2_percentiles_history[init_seed]
    if not per_step:
        save_figure(figure, output_path)
        return

    num_steps = len(per_step)
    effective_time = [learning_rate * step for step in range(num_steps)]
    traces = list(zip(*per_step))  # one sequence per percentile
    downsampled = [downsample(effective_time, list(trace), max_plot_points) for trace in traces]
    x_axis = downsampled[0][0]
    ys = [pair[1] for pair in downsampled]

    # === Color and style assignment per percentile ===
    median_color = "#000000"
    extreme_color = "#9467bd"   # 0% and 100%
    ribbon_color = "#ff7f0e"    # 20% and 80%
    default_color = "#bbbbbb"

    def style_for(probability: float):
        if math.isclose(probability, 0.5):
            return median_color, 2.0, "-"
        if math.isclose(probability, 0.0) or math.isclose(probability, 1.0):
            return extreme_color, 1.6, "-"
        if math.isclose(probability, 0.2) or math.isclose(probability, 0.8):
            return ribbon_color, 1.4, "-"
        return default_color, 0.9, "-"

    seen_labels = set()
    for probability, trace_y in zip(percentile_probabilities, ys):
        color, line_width, line_style = style_for(probability)
        percent = int(round(probability * 100))
        if color is median_color:
            label_text = "median (50%)"
        elif color is extreme_color:
            label_text = "min / max (0% & 100%)"
        elif color is ribbon_color:
            label_text = "20% & 80%"
        else:
            label_text = "other deciles"
        label = label_text if label_text not in seen_labels else None
        seen_labels.add(label_text)
        axis.plot(x_axis, trace_y, color=color, linewidth=line_width,
                  linestyle=line_style, label=label)

    axis.set_yscale("log")
    axis.set_ylabel(r"per-neuron $\langle z_k^2 \rangle$ (percentiles across $k$)")
    axis.set_title(f"per-neuron preactivation variance — percentile traces (seed {init_seed})")
    axis.legend(loc="best", fontsize=8)
    save_figure(figure, output_path)


def plot_neuron_z2_heatmap(
    neuron_z2_full_history: Dict[int, "np.ndarray"],
    output_path: Path,
    learning_rate: float,
    max_plot_points: int,
) -> None:
    """Heatmap of per-neuron preactivation variance <z_k^2> over training.

    Rows are neurons (sorted by final <z_k^2> descending so specialists are at the top),
    columns are training time, color is log <z_k^2>. Shows the first init seed only.
    """
    figure, axis = plt.subplots(figsize=(10, 6))
    sorted_seeds = sorted(neuron_z2_full_history.keys())
    if not sorted_seeds:
        save_figure(figure, output_path)
        return
    init_seed = sorted_seeds[0]
    array = neuron_z2_full_history[init_seed]
    if array is None or np.size(array) == 0:
        save_figure(figure, output_path)
        return

    array = np.asarray(array, dtype=float)  # (T, N)
    num_steps, num_neurons = array.shape

    # === Sort neurons by final <z_k^2> descending (specialists at top) ===
    final_values = array[-1, :]
    order = np.argsort(-final_values)
    array = array[:, order]

    # === Downsample time axis to keep image width manageable ===
    if num_steps > max_plot_points:
        step = math.ceil(num_steps / max_plot_points)
        time_indices = np.arange(0, num_steps, step)
    else:
        time_indices = np.arange(num_steps)
    array_plot = array[time_indices, :]
    effective_time_left = float(time_indices[0]) * learning_rate
    effective_time_right = float(time_indices[-1]) * learning_rate

    # === Log color norm; floor by smallest positive value to avoid log(0) ===
    positive_mask = array_plot > 0
    if positive_mask.any():
        floor_value = float(array_plot[positive_mask].min()) * 0.5
    else:
        floor_value = 1e-12
    array_plot = np.clip(array_plot, floor_value, None)

    image = axis.imshow(
        array_plot.T,  # (N, T_plot): y = neuron rank, x = time
        aspect="auto",
        origin="upper",
        norm=mcolors.LogNorm(vmin=floor_value, vmax=float(array_plot.max())),
        cmap="viridis",
        extent=(effective_time_left, effective_time_right, num_neurons, 0),
        interpolation="nearest",
    )

    axis.set_xlabel("effective time (learning rate × epoch)")
    axis.set_ylabel(r"neuron rank (sorted by final $\langle z_k^2 \rangle$)")
    axis.set_title(
        rf"per-neuron preactivation variance — heatmap (seed {init_seed}, $N$={num_neurons})"
    )
    figure.colorbar(image, ax=axis, label=r"$\langle z_k^2 \rangle$")
    save_figure(figure, output_path)


def plot_all_b_elements(
    b_history: Dict[int, Dict[tuple, List[float]]],
    b_elements: Tuple[Tuple[int, ...], ...],
    output_path: Path,
    learning_rate: float,
    max_plot_points: int,
    smoothing_factor: float = 0.5,
    theory_b_histories: Optional[Dict[tuple, List[float]]] = None,
    theory_effective_times: Optional[List[float]] = None,
) -> None:
    """Plot all B tensor elements on one figure.

    Args:
        b_history: Maps init_seed -> {indices_tuple: [values]}.
        b_elements: Tuple of index tuples to plot.
        output_path: Path to save the plot (pdf).
        learning_rate: Base learning rate, used for effective time x-axis.
        max_plot_points: Maximum number of points plotted on the x-axis.
        smoothing_factor: EMA smoothing factor.
        theory_b_histories: Optional {indices_tuple: [theory_values]} from the feature ODE.
        theory_effective_times: Effective time values for theory points.
    """
    figure, axis = create_figure()
    sorted_seeds = sorted(b_history.keys())

    for element_index, indices in enumerate(b_elements):
        indices_tuple = tuple(indices)
        color = get_seed_color(element_index)

        if indices_tuple:
            index_label = ",".join(str(i) for i in indices_tuple)
            order = len(indices_tuple)
            element_name = f"B^({order})[{index_label}]"
        else:
            element_name = "B^(0)"

        for seed_index, init_seed in enumerate(sorted_seeds):
            values = b_history[init_seed][indices_tuple]
            effective_time = [learning_rate * epoch for epoch in range(1, len(values) + 1)]
            ema_values = compute_ema(values, smoothing_factor)

            x_raw, y_raw = downsample(effective_time, values, max_plot_points)
            x_ema, y_ema = downsample(effective_time, ema_values, max_plot_points)

            axis.plot(x_raw, y_raw, color=color, alpha=0.15, linewidth=0.5)
            label = element_name if seed_index == 0 else None
            axis.plot(x_ema, y_ema, color=color, linewidth=1.5, label=label)

    # === Theory curves (same color as experimental, dashed) ===
    if theory_b_histories is not None and theory_effective_times is not None:
        for element_index, indices in enumerate(b_elements):
            indices_tuple = tuple(indices)
            if indices_tuple in theory_b_histories:
                color = get_seed_color(element_index)
                theory_values = theory_b_histories[indices_tuple]
                x_theory, y_theory = downsample(theory_effective_times, theory_values, max_plot_points)
                if indices_tuple:
                    index_label = ",".join(str(i) for i in indices_tuple)
                    order = len(indices_tuple)
                    theory_label = f"theory B^({order})[{index_label}]"
                else:
                    theory_label = "theory B^(0)"
                axis.plot(x_theory, y_theory, color=color, linestyle="--", linewidth=2,
                          label=theory_label)

    axis.set_ylabel("B element value")
    axis.set_title("B tensor elements over training")
    axis.legend()
    save_figure(figure, output_path)
