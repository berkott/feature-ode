"""Run every experiment in a configuration file on this machine, one after another.

Usage:
    uv run python launch/local.py --configuration configurations/smoke_test.yaml
"""

import argparse
from pathlib import Path

from feature_ode.configuration import load_experiment_configuration
from feature_ode.pipeline import create_batch_directory, save_outputs, train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Run experiments locally.")
    parser.add_argument("--configuration", type=str, required=True, help="Path to a YAML configuration file.")
    args = parser.parse_args()

    configuration_path = Path(args.configuration)
    print(f"\nLoading configuration from {configuration_path}")

    base_configuration, experiments = load_experiment_configuration(configuration_path)

    # === Create batch directory once ===
    batch_directory = create_batch_directory(configuration_path)
    print(f"Batch directory: {batch_directory}")

    if experiments:
        # === Multi-experiment mode: run sequentially ===
        print(f"\nRunning {len(experiments)} experiments sequentially...")

        for i, experiment in enumerate(experiments):
            name = experiment.get("name", f"experiment_{i}")
            print(f"\n[{i + 1}/{len(experiments)}] Starting {name}...")

            experiment_configuration = base_configuration.with_yaml_overrides(experiment)
            results = train_and_evaluate(experiment_configuration)
            save_outputs(results, run_directory=batch_directory / name)

        print(f"\nDone: {len(experiments)} experiments saved to {batch_directory}")

    else:
        # === Single experiment mode ===
        print("\nRunning single experiment...")

        results = train_and_evaluate(base_configuration)
        save_outputs(results, run_directory=batch_directory)


if __name__ == "__main__":
    main()
