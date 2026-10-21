from pathlib import Path

import pytest
import yaml

from feature_ode.configuration import Configuration, load_experiment_configuration

CONFIGURATIONS_DIRECTORY = Path(__file__).resolve().parent.parent / "configurations"


@pytest.mark.parametrize("path", sorted(CONFIGURATIONS_DIRECTORY.glob("*.yaml")), ids=lambda path: path.stem)
def test_every_configuration_loads(path):
    base_configuration, experiments = load_experiment_configuration(path)
    for experiment in experiments or []:
        base_configuration.with_yaml_overrides(experiment)


def test_unknown_key_raises(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("optimizer:\n  learning_rat: 0.1\n")
    with pytest.raises(ValueError, match="learning_rat"):
        Configuration.from_yaml(path)


def test_unknown_section_raises(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("optimiser:\n  learning_rate: 0.1\n")
    with pytest.raises(ValueError, match="optimiser"):
        Configuration.from_yaml(path)


def test_dict_round_trip_and_plain_yaml():
    configuration = Configuration(d=3, a_elements_plot=((0, 0), (1, 1)), orders=(0, 1, 2))
    as_dict = configuration.to_dict()
    # safe_dump refuses Python-specific types such as tuples, so this checks the saved YAML stays plain.
    yaml.safe_dump(as_dict)
    assert Configuration.from_dict(as_dict) == configuration
