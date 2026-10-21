"""Training and evaluation pipeline."""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from feature_ode.configuration import Configuration
from feature_ode.ode import train as integrate_feature_ode
from feature_ode.training import train as train_in_parallel


OUTPUT_DIRECTORY = Path("results")


def train_and_evaluate(configuration: Configuration) -> Dict[str, Any]:
    """Train the model a configuration describes, or integrate the feature ODE when model_class is "feature_ode".

    Returns:
        Dict with loss_history, a_history, b_history, and the configuration (under "config").
    """
    gammas = configuration.get_gammas()
    gammas_text = ", ".join(f"{g:.4f}" for g in gammas.tolist())
    print("=" * 60)
    print(f"Model: {configuration.model_class} | d={configuration.d} | "
          f"alpha={configuration.powerlaw_alpha} | data_seed={configuration.data_seed}")
    print(f"Gammas: [{gammas_text}]")
    print(f"Init seeds: {configuration.init_seeds}")
    print(f"Target: {configuration.target} (hermite={configuration.target_hermite})")
    print("=" * 60)

    if configuration.model_class == "feature_ode":
        return integrate_feature_ode(configuration)
    return train_in_parallel(configuration)


def create_batch_directory(
    configuration_path: Path,
    output_directory: Optional[Path] = None,
) -> Path:
    """Create a timestamped batch directory for a set of experiments.

    Directory name format: YYYY_MM_DD-HH_MM_SS-<configuration file stem>

    Args:
        configuration_path: Path to the YAML configuration file (stem used in directory name).
        output_directory: Base directory for outputs. Defaults to results/.

    Returns:
        Path to the created batch directory.
    """
    if output_directory is None:
        output_directory = OUTPUT_DIRECTORY

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    batch_directory = output_directory / f"{timestamp}-{configuration_path.stem}"
    batch_directory.mkdir(parents=True, exist_ok=True)
    return batch_directory


def save_outputs(
    results: Dict[str, Any],
    run_directory: Path,
) -> Path:
    """Save training results and evaluation plots to disk.

    Saves into the given run_directory:
        - configuration.yaml
        - results.pkl
        - loss.pdf and loss_clipped.pdf
        - a_elements.pdf and b_elements.pdf (when elements are tracked)
        - per-neuron preactivation variance plots (when enabled)

    Args:
        results: Training results dict from train_and_evaluate.
        run_directory: Directory to save all outputs into.

    Returns:
        Path to the run directory.
    """
    run_directory.mkdir(parents=True, exist_ok=True)

    # === Save configuration ===
    with open(run_directory / "configuration.yaml", "w") as f:
        yaml.safe_dump(results["config"], f, default_flow_style=False, sort_keys=False)

    # === Generate evaluation plots and compute theory curves ===
    from feature_ode.evaluation import evaluate
    configuration = Configuration.from_dict(results["config"])
    theory = evaluate(results, configuration, run_directory)

    # === Save raw results (including theory curves) ===
    results["theory"] = theory
    with open(run_directory / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"  Saved to {run_directory}")
    return run_directory
