import math
from pathlib import Path

import pytest
import yaml

from feature_ode.configuration import load_experiment_configuration
from feature_ode.pipeline import save_outputs, train_and_evaluate

SMOKE_TEST_PATH = Path(__file__).resolve().parent.parent / "configurations" / "smoke_test.yaml"
BASE_CONFIGURATION, EXPERIMENTS = load_experiment_configuration(SMOKE_TEST_PATH)


@pytest.mark.parametrize("experiment", EXPERIMENTS, ids=lambda experiment: experiment["name"])
def test_model_trains_and_saves(experiment, tmp_path):
    configuration = BASE_CONFIGURATION.with_yaml_overrides(experiment)
    results = train_and_evaluate(configuration)

    for losses in results["loss_history"].values():
        assert all(math.isfinite(loss) for loss in losses)
        assert losses[-1] < 0.5 * losses[0]

    run_directory = save_outputs(results, run_directory=tmp_path / experiment["name"])
    assert (run_directory / "results.pkl").exists()
    assert (run_directory / "loss.pdf").exists()
    yaml.safe_load((run_directory / "configuration.yaml").read_text())
