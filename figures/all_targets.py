"""Appendix figures: DPR or the semi-frozen network learning four targets, with the feature ODE's prediction.

Usage:
    uv run python figures/all_targets.py                       # both models
    uv run python figures/all_targets.py --model dpr
    uv run python figures/all_targets.py --model semi_frozen_network
"""

import argparse

import matplotlib.pyplot as plt

from plotting import (
    COLOR_CUBIC, COLOR_LINEAR, COLOR_QUADRATIC, EFFECTIVE_TIME_LABEL,
    apply_style, compute_max_time, load_results, plot_loss_with_ode, save_figure,
)


# (experiment, title, color) for each panel
PANELS = [
    ("linear", r"$f^* = z_1$", COLOR_LINEAR),
    ("quadratic", r"$f^* = \mathrm{He}_2(z_1)$", COLOR_QUADRATIC),
    ("two_term_staircase", r"$f^* = z_1 + z_1 z_2$", COLOR_QUADRATIC),
    ("three_term_staircase", r"$f^* = z_1 + z_1 z_2 + z_1 z_2 z_3$", COLOR_CUBIC),
]

# model -> (configuration, legend label, right x-limit as a multiple of the run's length)
MODELS = {
    "dpr": ("dpr_all_targets", "DPR", 1.5),
    "semi_frozen_network": ("semi_frozen_network_all_targets", "SFN", 1.0),
}


def make_figure(model):
    configuration_name, model_label, right_limit_factor = MODELS[model]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.0))
    flat = axes.flatten()
    for i, (experiment_name, title, color) in enumerate(PANELS):
        results = load_results(configuration_name, experiment_name)
        plot_loss_with_ode(flat[i], results, color, model_label,
                           right_limit=right_limit_factor * compute_max_time(results))
        if i % 2 == 0:
            flat[i].set_ylabel("Online MSE")
        flat[i].set_title(title)
        if i == 0:
            flat[i].legend(frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.11)
    fig.supxlabel(EFFECTIVE_TIME_LABEL, y=0.02)
    save_figure(fig, f"all_targets_{model}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), help="Make only this model's figure.")
    args = parser.parse_args()

    apply_style()
    for model in [args.model] if args.model else MODELS:
        make_figure(model)


if __name__ == "__main__":
    main()
