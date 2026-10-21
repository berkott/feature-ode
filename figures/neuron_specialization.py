"""Figure 4: MLP loss on the two-term staircase over each neuron's preactivation variance, on a shared time axis.

Usage:
    uv run python figures/neuron_specialization.py
"""

import matplotlib.pyplot as plt

from plotting import (
    COLOR_QUADRATIC, apply_style, compute_max_time, load_results, plot_loss_with_ode,
    plot_neuron_variance_heatmap, save_figure,
)


# The heatmap is twice as tall as the loss panel.
HEATMAP_TO_LOSS_HEIGHT = 2.0


def main():
    apply_style()
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 14})

    results = load_results("mlp_two_term_staircase")
    max_time = compute_max_time(results)

    fig = plt.figure(figsize=(8.0, 4.6))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, HEATMAP_TO_LOSS_HEIGHT],
        hspace=0.10,
        left=0.09, right=0.88, top=0.97, bottom=0.11,
    )
    ax_loss = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1], sharex=ax_loss)

    plot_loss_with_ode(ax_loss, results, COLOR_QUADRATIC, "MLP", right_limit=max_time)
    ax_loss.set_ylabel("Online MSE")
    ax_loss.tick_params(labelbottom=False)
    ax_loss.legend(frameon=False, fontsize=11)

    mesh = plot_neuron_variance_heatmap(ax_heat, results, max_time)
    # Inset colorbar so the heatmap keeps the loss panel's width and the time axes line up.
    cax = ax_heat.inset_axes([1.01, 0.0, 0.015, 1.0])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$\mathrm{Var}[\mathbf{w}_{\mathrm{in},k}^\top \mathbf{x}]$")

    fig.align_ylabels([ax_loss, ax_heat])
    save_figure(fig, "neuron_specialization")


if __name__ == "__main__":
    main()
