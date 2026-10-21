"""Run every experiment in a configuration file in parallel on Modal (https://modal.com).

Usage (requires `uv sync --extra modal` and a Modal account):
    uv run modal run launch/modal_app.py --configuration configurations/mlp_two_term_staircase.yaml
"""

import pickle
from pathlib import Path

import modal

# === Modal setup ===
app = modal.App("feature-ode")

project_root = Path(__file__).parent.parent

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.11",
        "numpy>=2.0.2",
        "attrs>=25.4.0",
        "pyyaml>=6.0.3",
        "matplotlib==3.10.8",
    )
    .add_local_dir(project_root / "feature_ode", remote_path="/root/feature_ode")
)


# === Remote function for Modal ===
@app.function(
    image=training_image,
    gpu="A100",
    timeout=3600,
)
def run_training_job(configuration_dict: dict):
    """Train one experiment on Modal and return its pickled results."""
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from feature_ode.configuration import Configuration
    from feature_ode.pipeline import train_and_evaluate

    configuration = Configuration.from_dict(configuration_dict)
    results = train_and_evaluate(configuration)
    return pickle.dumps(results)


@app.local_entrypoint()
def main(configuration: str):
    """Run experiments on Modal.

    Args:
        configuration: Path to a YAML configuration file.
    """
    from feature_ode.configuration import load_experiment_configuration
    from feature_ode.pipeline import create_batch_directory, save_outputs

    configuration_path = Path(configuration)
    print(f"\nLoading configuration from {configuration_path}")

    base_configuration, experiments = load_experiment_configuration(configuration_path)

    # === Create batch directory once ===
    batch_directory = create_batch_directory(configuration_path)
    print(f"Batch directory: {batch_directory}")

    if experiments:
        # === Multi-experiment mode: run in parallel on Modal ===
        print(f"\nRunning {len(experiments)} experiments in parallel on Modal...")

        configuration_dicts = []
        experiment_names = []
        for experiment in experiments:
            experiment_configuration = base_configuration.with_yaml_overrides(experiment)
            configuration_dicts.append(experiment_configuration.to_dict())
            experiment_names.append(experiment.get("name", f"experiment_{len(configuration_dicts)}"))

        saved_count = 0
        for name, result_bytes in zip(experiment_names, run_training_job.map(configuration_dicts)):
            results = pickle.loads(result_bytes)
            saved_count += 1
            print(f"  [{saved_count}/{len(experiment_names)}] Saving {name}...")
            save_outputs(results, run_directory=batch_directory / name)

        print(f"\nDone: {saved_count} experiments saved to {batch_directory}")

    else:
        # === Single experiment mode: save directly into batch directory ===
        print("\nRunning single experiment on Modal...")

        result_bytes = run_training_job.remote(base_configuration.to_dict())
        results = pickle.loads(result_bytes)
        save_outputs(results, run_directory=batch_directory)
