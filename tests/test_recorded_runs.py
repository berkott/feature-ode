"""Check the feature ODE against the curves stored with the recorded runs behind the paper's figures.

These runs are not in the repository. To run this test, place them in results/ (see the README)
and set FEATURE_ODE_CHECK_RECORDED_RUNS=1. Loading the runs takes a minute or two.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from feature_ode.configuration import load_experiment_configuration
from feature_ode.ode import compute_theory_curves

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RUNS = [
    ("mlp_two_term_staircase", None),
    ("mlp_three_term_staircase", None),
    ("dpr_all_targets", "linear"),
    ("dpr_all_targets", "quadratic"),
    ("dpr_all_targets", "two_term_staircase"),
    ("dpr_all_targets", "three_term_staircase"),
    ("semi_frozen_network_all_targets", "linear"),
    ("semi_frozen_network_all_targets", "quadratic"),
    ("semi_frozen_network_all_targets", "two_term_staircase"),
    ("semi_frozen_network_all_targets", "three_term_staircase"),
    ("mlp_linear_and_pure_hermite", "linear"),
    ("mlp_linear_and_pure_hermite", "pure_hermite"),
    ("semi_frozen_network_pure_hermite", None),
]

pytestmark = pytest.mark.skipif(
    os.environ.get("FEATURE_ODE_CHECK_RECORDED_RUNS") != "1",
    reason="set FEATURE_ODE_CHECK_RECORDED_RUNS=1 and place the recorded runs in results/",
)


@pytest.mark.parametrize("configuration_name, experiment_name", RUNS)
def test_feature_ode_reproduces_recorded_theory_curves(configuration_name, experiment_name):
    batch_directories = sorted((REPOSITORY_ROOT / "results").glob(f"*-{configuration_name}"))
    assert batch_directories, f"no run of {configuration_name} in results/"
    run_directory = batch_directories[-1] / (experiment_name or "")
    with open(run_directory / "results.pkl", "rb") as f:
        recorded = pickle.load(f)

    base_configuration, experiments = load_experiment_configuration(
        REPOSITORY_ROOT / "configurations" / f"{configuration_name}.yaml")
    configuration = base_configuration
    if experiment_name is not None:
        configuration = base_configuration.with_yaml_overrides(
            next(experiment for experiment in experiments if experiment["name"] == experiment_name))

    any_seed = next(iter(recorded["loss_history"]))
    record_interval = recorded.get("record_interval", 1)
    total_effective_time = len(recorded["loss_history"][any_seed]) * record_interval * configuration.learning_rate
    theory = compute_theory_curves(configuration, total_effective_time)

    np.testing.assert_allclose(theory["effective_times"], recorded["theory"]["effective_times"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(theory["loss_history"], recorded["theory"]["loss_history"], rtol=1e-12, atol=1e-15)
    for element, values in recorded["theory"]["a_history"].items():
        np.testing.assert_allclose(theory["a_history"][element], values, rtol=1e-12, atol=1e-15)
