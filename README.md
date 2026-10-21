# feature-ode

This repository contains the code for the paper
**A "feature ODE" describing the learning behavior of shallow MLPs on simple functions**
by Joseph Turnbull, Berkan Ottlik, and Jamie Simon
([arXiv:2610.25652](https://arxiv.org/abs/2610.25652), [PDF](https://arxiv.org/pdf/2610.25652)),
which was accepted at the HiLD workshop at ICML 2026.

The paper replaces the elementwise nonlinearity of a shallow MLP with a polynomial featurization, which gives a model we call deep polynomial regression (DPR).
Gradient flow on DPR reduces to an ODE of a few scalar variables, the *feature ODE*, and this ODE predicts the loss curves and learning times of shallow MLPs on simple target functions.
The code trains MLPs, DPR, and the semi-frozen network (SFN) from the paper's appendix on anisotropic Gaussian data, integrates the feature ODE, and draws the paper's figures.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.10 or later.

```bash
uv sync                  # installs the package, its dependencies, and pytest
uv sync --extra modal    # also installs Modal, if you want to run on Modal
```

## Quick start

The smoke-test configuration trains a tiny MLP, SFN, and DPR and integrates the feature ODE, all on CPU in under a minute:

```bash
uv run python launch/local.py --configuration configurations/smoke_test.yaml
uv run pytest
```

The first command writes each run's settings, results, and diagnostic plots to `results/<timestamp>-smoke_test/`.

## Running experiments

Each YAML file in `configurations/` describes one batch of runs.
Its `data`, `optimizer`, `model`, and `evaluate` sections set the run's settings, and an optional `grid_search` section expands into one run per combination of values.
Unknown keys raise an error, so a typo cannot silently fall back to a default.
When `max_epochs` is `-1`, a run trains for twice the time the feature ODE takes to converge.

Three launchers run a configuration and write the same `results/<timestamp>-<configuration>/` layout:

```bash
uv run python launch/local.py --configuration configurations/<name>.yaml        # this machine
uv run modal run launch/modal_app.py --configuration configurations/<name>.yaml # Modal, one GPU per run
uv run python launch/slurm.py --configuration configurations/<name>.yaml        # a SLURM cluster
```

The SLURM launcher takes the cluster's partition, GPU type, modules, and setup commands as flags (`uv run python launch/slurm.py --help`).
The paper's runs use networks of width 8192 and batches of 10,000 samples, so we recommend a GPU for them.

## Reproducing the figures

Each figure script reads the latest run of its configurations from `results/` and writes a PDF to `figures/output/`.

| Figure | Configurations | Command |
|---|---|---|
| Figure 1, right panel | recorded data (see below) | `uv run python figures/learning_time_prediction.py` |
| Figure 3 | `mlp_two_term_staircase`, `mlp_three_term_staircase` | `uv run python figures/staircase_losses.py` |
| Figure 4 | `mlp_two_term_staircase` | `uv run python figures/neuron_specialization.py` |
| Figure 5 | `semi_frozen_network_all_targets` | `uv run python figures/all_targets.py --model semi_frozen_network` |
| Figure 6 | `dpr_all_targets` | `uv run python figures/all_targets.py --model dpr` |
| Figures 7–9 | `mlp_linear_and_pure_hermite`, `mlp_two_term_staircase`, `semi_frozen_network_all_targets`, `semi_frozen_network_pure_hermite` | `uv run python figures/neuron_dynamics.py` |

Figure 2 is a diagram and has no code.
The neuron-dynamics script draws each panel of Figures 7–9 as its own PDF; `--model` and `--target` select a single one.

### Notes on the figures

- **Figure 1 uses recorded numbers.** We made the runs behind Figure 1 with an earlier version of this code, so `figures/learning_time_prediction.py` plots the learning times recorded from those runs (`figures/recorded_data/learning_time_prediction.json`) rather than rerunning them. The loss curves in the left panel came from the same runs and are not included.
- **Figure 1 halves the feature ODE's prediction for quadratic targets.** This is a visual adjustment. Both axes are logarithmic, so the halving moves the quadratic points down by a constant log10(2) and leaves their trend unchanged. The adjustment happens in `load_learning_times` in `figures/learning_time_prediction.py`.
- **The feature ODE uses c_ℓ where the paper writes √c_ℓ.** The two agree for the cosine activation (c_ℓ = 1), which every run in `configurations/` uses except Figure 3(b), which uses shifted tanh. We kept c_ℓ so that the code reproduces the published curves; see the comment in `FeatureODE` in `feature_ode/ode.py`. Figure 1's recorded predictions came from the earlier implementation, which used the paper's √c_ℓ.
- **We rebuilt the configurations from the settings that the original runs saved.** With the original runs in `results/`, the feature ODE reproduces their recorded theory curves to within 10⁻¹² (`tests/test_recorded_runs.py`), and the figure scripts reproduce the published PDFs pixel for pixel. Exact matching requires Matplotlib 3.10.8, which `pyproject.toml` pins. We have not rerun the GPU training since this cleanup.
- **The original runs are not in this repository**, because together they take up about 5 GB. To regenerate a figure from scratch, run its configurations first.

## Testing

```bash
uv run pytest
```

The tests check that every configuration loads and rejects unknown keys, that the feature ODE's derivatives are the gradient of its loss, that single-term dynamics conserve a_i² − α_i b² (the quantity behind the paper's learning-time integral), and that every model trains for a few steps on CPU.
If you have the original runs in `results/`, setting `FEATURE_ODE_CHECK_RECORDED_RUNS=1` also checks the feature ODE against their recorded curves.

## Repository layout

```
feature_ode/          the package
  configuration.py    YAML configurations
  activations.py      nonlinearities and their c_ℓ coefficients
  data.py             Gaussian data and Hermite or monomial targets
  models/             mlp.py, semi_frozen_network.py, deep_polynomial_regression.py
  ode.py              the feature ODE
  training.py         online SGD over several init seeds in parallel
  pipeline.py         runs one configuration and saves its results
  evaluation.py       per-run diagnostic plots with feature ODE overlays
configurations/       one YAML file per batch of runs
launch/               local, Modal, and SLURM launchers
figures/              one script per paper figure, and shared plotting code
tests/
```

## Citation

```bibtex
@misc{turnbull2026featureode,
  title         = {A ``feature {ODE}'' describing the learning behavior of shallow {MLPs} on simple functions},
  author        = {Turnbull, Joseph and Ottlik, Berkan and Simon, Jamie},
  year          = {2026},
  eprint        = {2610.25652},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.25652},
  note          = {HiLD Workshop at ICML 2026}
}
```

## License

This code is released under the MIT License; see `LICENSE`.
