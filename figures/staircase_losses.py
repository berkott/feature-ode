"""Figure 3: MLP loss curves for the two- and three-term staircases, with the feature ODE's prediction.

Usage:
    uv run python figures/staircase_losses.py
"""

import matplotlib.pyplot as plt

from plotting import (
    COLOR_CUBIC, COLOR_QUADRATIC, EFFECTIVE_TIME_LABEL,
    apply_style, compute_max_time, load_results, plot_loss_with_ode, save_figure,
)


# (configuration, title, color) for each panel
PANELS = [
    ("mlp_two_term_staircase", r"$f^* = z_1 + z_1 z_2$", COLOR_QUADRATIC),
    ("mlp_three_term_staircase", r"$f^* = z_1 + z_1 z_2 + z_1 z_2 z_3$", COLOR_CUBIC),
]


def main():
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8224))
    for i, (configuration_name, title, color) in enumerate(PANELS):
        results = load_results(configuration_name)
        plot_loss_with_ode(axes[i], results, color, "MLP", right_limit=compute_max_time(results))
        if i == 0:
            axes[i].set_ylabel("Online MSE")
        axes[i].set_title(title)
        if i == 0:
            axes[i].legend(frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.supxlabel(EFFECTIVE_TIME_LABEL, y=0.04)
    save_figure(fig, "staircase_losses")


if __name__ == "__main__":
    main()
