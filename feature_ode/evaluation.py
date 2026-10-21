"""Evaluation: per-run diagnostic plots with feature ODE theory overlays."""

from pathlib import Path
from typing import Dict, Any

from feature_ode.configuration import Configuration
from feature_ode.ode import compute_theory_curves
from feature_ode.diagnostic_plots import (
    plot_loss_curves,
    plot_all_a_elements,
    plot_all_b_elements,
    plot_neuron_z2_heatmap,
    plot_neuron_z2_histogram,
    plot_neuron_z2_percentiles_over_time,
)


# === Main evaluation ===

def evaluate(results: Dict[str, Any], configuration: Configuration, output_directory: Path) -> Dict[str, Any]:
    """Generate all evaluation plots from training results.

    Runs the feature ODE alongside and overlays dashed theory curves.

    Args:
        results: Training results dict with keys:
            - "loss_history": {init_seed: [loss_per_epoch]}
            - "a_history": {init_seed: {(row, col): [value_per_epoch]}}
            - "b_history": {init_seed: {indices_tuple: [value_per_epoch]}} (DPR only)
        configuration: Configuration with a_elements_plot and b_elements_plot.
        output_directory: Directory to save plots into.

    Returns:
        Theory curves dict with effective_times, loss_history, a_history, b_history.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    # === Compute feature ODE theory curves ===
    any_seed = next(iter(results["loss_history"]))
    record_interval = results.get("record_interval", 1)
    num_training_steps = len(results["loss_history"][any_seed]) * record_interval
    total_effective_time = num_training_steps * configuration.learning_rate

    theory = compute_theory_curves(configuration, total_effective_time)
    theory_times = theory["effective_times"]

    # === Effective learning rate per recorded point ===
    effective_lr = configuration.learning_rate * record_interval

    # === Loss curves ===
    plot_loss_curves(
        loss_history=results["loss_history"],
        output_path=output_directory / "loss.pdf",
        learning_rate=effective_lr,
        max_plot_points=configuration.max_plot_points,
        smoothing_factor=configuration.smoothing_factor,
        theory_loss_history=theory["loss_history"],
        theory_effective_times=theory_times,
    )

    # === Loss curves (clipped y-axis) ===
    plot_loss_curves(
        loss_history=results["loss_history"],
        output_path=output_directory / "loss_clipped.pdf",
        learning_rate=effective_lr,
        max_plot_points=configuration.max_plot_points,
        smoothing_factor=configuration.smoothing_factor,
        theory_loss_history=theory["loss_history"],
        theory_effective_times=theory_times,
        y_max=1.0,
    )

    # === All A elements in one plot ===
    if configuration.a_elements_plot and results.get("a_history"):
        plot_all_a_elements(
            a_history=results["a_history"],
            a_elements=configuration.a_elements_plot,
            output_path=output_directory / "a_elements.pdf",
            learning_rate=effective_lr,
            max_plot_points=configuration.max_plot_points,
            smoothing_factor=configuration.smoothing_factor,
            theory_a_histories=theory["a_history"],
            theory_effective_times=theory_times,
        )

    # === All B elements in one plot ===
    if configuration.b_elements_plot and results.get("b_history"):
        plot_all_b_elements(
            b_history=results["b_history"],
            b_elements=configuration.b_elements_plot,
            output_path=output_directory / "b_elements.pdf",
            learning_rate=effective_lr,
            max_plot_points=configuration.max_plot_points,
            smoothing_factor=configuration.smoothing_factor,
            theory_b_histories=theory["b_history"],
            theory_effective_times=theory_times,
        )

    # === Per-neuron preactivation variance plots ===
    neuron_z2_init = results.get("neuron_z2_init") or {}
    neuron_z2_final = results.get("neuron_z2_final") or {}
    if configuration.neuron_histogram and neuron_z2_init and neuron_z2_final:
        plot_neuron_z2_histogram(
            neuron_z2_init=neuron_z2_init,
            neuron_z2_final=neuron_z2_final,
            output_path=output_directory / "neuron_z2_histogram.pdf",
        )
    neuron_z2_percentiles_history = results.get("neuron_z2_percentiles_history") or {}
    percentile_probabilities = results.get("neuron_z2_percentile_probabilities") or [0.0, 0.25, 0.5, 0.75, 1.0]
    if configuration.neuron_percentiles and any(neuron_z2_percentiles_history.values()):
        plot_neuron_z2_percentiles_over_time(
            neuron_z2_percentiles_history=neuron_z2_percentiles_history,
            percentile_probabilities=percentile_probabilities,
            output_path=output_directory / "neuron_z2_percentiles.pdf",
            learning_rate=effective_lr,
            max_plot_points=configuration.max_plot_points,
        )
    neuron_z2_full_history = results.get("neuron_z2_full_history") or {}
    if configuration.neuron_heatmap and any(v is not None for v in neuron_z2_full_history.values()):
        plot_neuron_z2_heatmap(
            neuron_z2_full_history=neuron_z2_full_history,
            output_path=output_directory / "neuron_z2_heatmap.pdf",
            learning_rate=effective_lr,
            max_plot_points=configuration.max_plot_points,
        )

    return theory
