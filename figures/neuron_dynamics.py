"""Appendix figures: loss, feature trajectories, and per-neuron preactivation variance for the MLP or the semi-frozen network.

Usage:
    uv run python figures/neuron_dynamics.py                    # all six figures
    uv run python figures/neuron_dynamics.py --model mlp --target two_term_staircase
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plotting import (
    COLOR_LINEAR, COLOR_QUADRATIC, EFFECTIVE_TIME_LABEL, SMOOTHING, XLIM_LEFT,
    apply_style, compute_ema, compute_max_time, downsample_log, drop_zero, format_log_axis,
    get_step_lr, load_results, make_shades, plot_loss_with_ode, plot_neuron_variance_heatmap, save_figure,
)


MODELS = {
    "mlp": {"label": "MLP", "colorbar_label": r"$\mathrm{Var}[\mathbf{w}_{\mathrm{in},k}^\top \mathbf{x}]$"},
    "semi_frozen_network": {
        "label": "SFN", "colorbar_label": r"$\mathrm{Var}[\mathbf{w}_{\mathrm{frozen},k}^\top \mathbf{A}\mathbf{x}]$",
    },
}

# (model, target) -> run (configuration, experiment), curve color, and legend placement for the A_ii panel
FIGURES = {
    ("mlp", "linear"): (("mlp_linear_and_pure_hermite", "linear"), COLOR_LINEAR, "split"),
    ("mlp", "pure_hermite"): (("mlp_linear_and_pure_hermite", "pure_hermite"), COLOR_QUADRATIC, "best"),
    ("mlp", "two_term_staircase"): (("mlp_two_term_staircase", None), COLOR_QUADRATIC, "best"),
    ("semi_frozen_network", "linear"): (("semi_frozen_network_all_targets", "linear"), COLOR_LINEAR, "upper_left"),
    ("semi_frozen_network", "pure_hermite"): (("semi_frozen_network_pure_hermite", None), COLOR_QUADRATIC, "best"),
    ("semi_frozen_network", "two_term_staircase"): (
        ("semi_frozen_network_all_targets", "two_term_staircase"), COLOR_QUADRATIC, "best"),
}


def aggregate_a(results, smoothing):
    """Return {element: (effective times, mean, std)} of the smoothed A (or sqrt NFM) trajectories across seeds."""
    step_lr = get_step_lr(results)
    a_hist = results.get("a_history") or {}
    if not a_hist:
        return {}
    seeds = sorted(a_hist.keys())
    elements = sorted(a_hist[seeds[0]].keys())
    out = {}
    for elem in elements:
        traces = [np.array(a_hist[s][elem]) for s in seeds]
        min_l = min(len(t) for t in traces)
        stacked = np.stack([t[:min_l] for t in traces], axis=0)
        sm = np.array([compute_ema(row.tolist(), smoothing) for row in stacked])
        et = np.arange(min_l) * step_lr
        out[elem] = (et, sm.mean(axis=0), sm.std(axis=0))
    return out


def plot_a(axis, results, color, model_label, max_time, legend_placement):
    """Plot the diagonal feature trajectories A_11 and A_22 (sqrt(NFM_ii) for the MLP) against the feature ODE."""
    a_dict = aggregate_a(results, SMOOTHING)
    theory = results.get("theory") or {}
    # Only plot diagonal A_{1,1} and A_{2,2} (math-indexed; code (0,0) and (1,1)).
    allowed = [(0, 0), (1, 1)]
    elem_order = [e for e in allowed if e in a_dict]
    shades = make_shades(color, len(elem_order), lighten_range=(0.0, 0.55))
    model_prefix = r"\sqrt{\mathrm{NFM}}" if model_label == "MLP" else "A"
    for i, elem in enumerate(elem_order):
        et, mean, std = a_dict[elem]
        et_d, mean_d, std_d = drop_zero(et, mean, std)
        et_d, mean_d, std_d = downsample_log(et_d, mean_d, std_d)
        lo = np.clip(mean_d - std_d, 0, None)
        hi = mean_d + std_d
        label = rf"${model_prefix}_{{{elem[0]+1},{elem[1]+1}}}$"
        axis.fill_between(et_d, lo, hi, color=shades[i], alpha=0.25, linewidth=0)
        axis.plot(et_d, mean_d, color=shades[i], label=label, zorder=11)

    if theory and "a_history" in theory:
        theory_elements = [e for e in allowed if e in theory["a_history"]]
        ode_shades = make_shades("#000000", len(theory_elements), lighten_range=(0.0, 0.5))
        t_th_full = np.array(theory["effective_times"])
        for i, elem in enumerate(theory_elements):
            values = np.array(theory["a_history"][elem])
            t_th, values = drop_zero(t_th_full, values)
            t_th, values = downsample_log(t_th, values)
            label = rf"ODE $A_{{{elem[0]+1},{elem[1]+1}}}$"
            axis.plot(t_th, values, color=ode_shades[i], linestyle="--",
                      label=label, zorder=20)

    axis.set_xscale("log")
    format_log_axis(axis)
    axis.set_xlim(left=XLIM_LEFT, right=max_time)
    axis.set_ylim(bottom=0)
    axis.set_ylabel(r"$A_{ii}$")
    axis.set_xlabel(EFFECTIVE_TIME_LABEL)

    legend_style = {"frameon": False, "fontsize": 11, "handlelength": 1.2, "handletextpad": 0.4}
    if legend_placement == "split":
        # Split the 4-entry legend: model curves top-left, ODE curves bottom-right.
        handles, labels = axis.get_legend_handles_labels()
        mid = len(handles) // 2
        model_legend = axis.legend(handles[:mid], labels[:mid], loc="upper left", borderaxespad=0.3, **legend_style)
        axis.add_artist(model_legend)
        axis.legend(handles[mid:], labels[mid:], loc="lower right", bbox_to_anchor=(1.0, 0.10), **legend_style)
    elif legend_placement == "upper_left":
        axis.legend(ncol=2, columnspacing=0.8, loc="upper left", borderaxespad=0.3, **legend_style)
    else:
        axis.legend(ncol=2, columnspacing=0.8, loc="best", **legend_style)


def make_figure(model, target):
    (configuration_name, experiment_name), color, legend_placement = FIGURES[(model, target)]
    model_label = MODELS[model]["label"]
    results = load_results(configuration_name, experiment_name)
    max_time = compute_max_time(results)

    fig = plt.figure(figsize=(8.0, 3.2))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.05, 1.5],
        hspace=0.12, wspace=0.28,
        left=0.07, right=0.97, top=0.96, bottom=0.15,
    )
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[1, 0], sharex=ax_loss)
    ax_heat = fig.add_subplot(gs[:, 1])

    plot_loss_with_ode(ax_loss, results, color, model_label, right_limit=max_time)
    ax_loss.set_ylabel("Online MSE")
    ax_loss.tick_params(labelbottom=False)
    ax_loss.legend(frameon=False, fontsize=11)

    plot_a(ax_a, results, color, model_label, max_time, legend_placement)

    mesh = plot_neuron_variance_heatmap(ax_heat, results, max_time)
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=ax_heat, pad=0.02)
        cbar.set_label(MODELS[model]["colorbar_label"])

    # Align ylabels on the two left panels at the same x position.
    ax_loss.yaxis.set_label_coords(-0.18, 0.5)
    ax_a.yaxis.set_label_coords(-0.18, 0.5)

    save_figure(fig, f"neuron_dynamics_{model}_{target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--target", choices=sorted({target for _, target in FIGURES}))
    args = parser.parse_args()

    apply_style()
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 14})
    for model, target in FIGURES:
        if (args.model in (None, model)) and (args.target in (None, target)):
            make_figure(model, target)


if __name__ == "__main__":
    main()
