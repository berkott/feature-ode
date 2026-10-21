# CLAUDE.md

Guidance for coding agents working in this repository. Read `README.md` first; this file adds the conventions and pitfalls that the README leaves out.

## Commands

```bash
uv sync                                                                     # install
uv run pytest                                                               # fast tests, about 10 seconds
FEATURE_ODE_CHECK_RECORDED_RUNS=1 uv run pytest tests/test_recorded_runs.py # needs the original runs in results/
uv run python launch/local.py --configuration configurations/smoke_test.yaml
uv run python figures/<figure>.py                                           # writes figures/output/<figure>.pdf
```

Do not launch Modal or SLURM jobs unless the user asks; they cost money or cluster time.

## Where the paper lives in the code

- The feature ODE (Section 3 of the paper) is `FeatureODE` in `feature_ode/ode.py`. Its `rate_constants` are the paper's c̃_α, except that the code uses c_ℓ where the paper writes √c_ℓ (see the comment there). Keep this convention unless the user decides otherwise, because the published figures depend on it.
- DPR is `DPR` in `feature_ode/models/deep_polynomial_regression.py`, the semi-frozen network (SFN) is `SemiFrozenNetwork`, and the MLP is `MLP`. Training for all three is the vmapped online SGD loop in `feature_ode/training.py`.
- Each paper figure has one script in `figures/`, and the README's table maps figures to scripts and configurations.

## Rules

- Never commit `results/` or `figures/output/`. Runs are large, and figures are regenerated from them.
- Never edit a configuration that a figure depends on. Add a new configuration instead, so every figure keeps a record of the settings that made it.
- Keep results backward compatible: runs store their settings under `results["config"]`, and the figure scripts read that key from older runs.
- Figure 1 halves the feature ODE's prediction for quadratic targets on purpose (`figures/learning_time_prediction.py`). Keep the comment that says so.
- Keep Matplotlib at 3.10.8 unless the user agrees to let the figures drift from the published PDFs.
- After changing plotting or ODE code, regenerate the figures and compare them with the previous PDFs, and run the recorded-runs test if the runs are available.

## Naming

Spell names out in plain English, for files, functions, variables, flags, and configuration keys. Write `configuration`, `evaluate`, and `statistics`, not `cfg`, `eval`, or `stats`, and do not invent acronyms or prefixes. DPR, SFN, MLP, and ODE are the paper's own names and are fine. Name things by what they are (`staircase_losses`), not by their position (`fig3`), and use the same name for an experiment in its configuration, its figure script, and its outputs.
