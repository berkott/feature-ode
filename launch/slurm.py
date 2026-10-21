"""Run the experiments in a configuration file as SLURM jobs.

Usage:
    Submit one job per experiment:
        uv run python launch/slurm.py --configuration configurations/mlp_two_term_staircase.yaml --partition gpu

    Submit one job that runs every experiment in turn:
        uv run python launch/slurm.py --configuration ... --bundle

    Run a single experiment by index (called from inside the SLURM jobs):
        uv run python launch/slurm.py --configuration ... --worker --experiment-index 3 --batch-directory results/...

Cluster-specific setup goes in flags: --module (repeatable) adds `module load` lines,
and --setup-command (repeatable) adds any other shell line, such as sourcing lmod.
"""

import argparse
import subprocess
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_job_header(job_name: str, log_name: str, batch_directory: str, args) -> str:
    """Return the #SBATCH lines and environment setup shared by both script styles."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=feature_ode_{job_name}",
        f"#SBATCH --gres=gpu:{args.gpu}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --output={batch_directory}/slurm_logs/{log_name}_%j.out",
        f"#SBATCH --error={batch_directory}/slurm_logs/{log_name}_%j.err",
    ]
    if args.partition:
        lines.insert(2, f"#SBATCH --partition={args.partition}")
    lines.append("")
    lines.extend(args.setup_command)
    lines.extend(f"module load {module}" for module in args.module)
    lines.extend(["", "export PYTHONUNBUFFERED=1", "", f"cd {args.repository_path}"])
    return "\n".join(lines) + "\n"


def generate_sbatch_script(experiment_index: int, experiment_name: str, batch_directory: str, args) -> str:
    """Generate a SLURM batch script for a single experiment."""
    header = generate_job_header(experiment_name, experiment_name, batch_directory, args)
    return header + textwrap.dedent(f"""\
        uv run python -u launch/slurm.py \\
            --configuration {args.configuration} \\
            --worker \\
            --experiment-index {experiment_index} \\
            --batch-directory {batch_directory}
    """)


def generate_bundled_sbatch_script(num_experiments: int, batch_directory: str, args) -> str:
    """Generate a single SLURM batch script that runs all experiments sequentially."""
    header = generate_job_header("bundle", "bundle", batch_directory, args)
    return header + textwrap.dedent(f"""\
        for i in $(seq 0 {num_experiments - 1}); do
            echo "===== Experiment $i / {num_experiments - 1} ====="
            uv run python -u launch/slurm.py \\
                --configuration {args.configuration} \\
                --worker \\
                --experiment-index $i \\
                --batch-directory {batch_directory}
        done
    """)


def submit_script(script: str, script_path: Path) -> str:
    """Write an sbatch script, submit it, and return the job id or an error message."""
    script_path.write_text(script)
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode == 0:
        return f"job {result.stdout.strip().split()[-1]}"
    return f"FAILED: {result.stderr.strip()}"


def submit(args):
    """Expand the grid search and submit the SLURM jobs."""
    from feature_ode.configuration import load_experiment_configuration
    from feature_ode.pipeline import create_batch_directory

    configuration_path = Path(args.configuration)
    print(f"\nLoading configuration from {configuration_path}")

    _, experiments = load_experiment_configuration(configuration_path)

    # === Create batch directory once ===
    batch_directory = create_batch_directory(configuration_path)
    slurm_logs_directory = batch_directory / "slurm_logs"
    slurm_logs_directory.mkdir(parents=True, exist_ok=True)
    print(f"Batch directory: {batch_directory}")

    if not experiments:
        experiments = [{"name": "single"}]

    # === Bundle mode: one SLURM job runs all experiments sequentially ===
    if args.bundle:
        print(f"\nSubmitting 1 bundled job ({len(experiments)} experiments)...")
        script = generate_bundled_sbatch_script(len(experiments), str(batch_directory), args)
        print(f"  bundle -> {submit_script(script, slurm_logs_directory / 'bundle.sbatch')}")
        print(f"\nResults will be saved to: {batch_directory}")
        return

    print(f"\nSubmitting {len(experiments)} jobs...")
    for i, experiment in enumerate(experiments):
        name = experiment.get("name", f"experiment_{i}")
        script = generate_sbatch_script(i, name, str(batch_directory), args)
        print(f"  [{i + 1}/{len(experiments)}] {name} -> {submit_script(script, slurm_logs_directory / f'{name}.sbatch')}")

    print("Monitor with: squeue -u $USER")
    print(f"Results will be saved to: {batch_directory}")


def worker(args):
    """Run a single experiment by index (called from within a SLURM job)."""
    from feature_ode.configuration import load_experiment_configuration
    from feature_ode.pipeline import save_outputs, train_and_evaluate

    base_configuration, experiments = load_experiment_configuration(Path(args.configuration))
    batch_directory = Path(args.batch_directory)

    if not experiments:
        # === Single experiment mode ===
        print("\nRunning single experiment...")
        results = train_and_evaluate(base_configuration)
        save_outputs(results, run_directory=batch_directory)
    else:
        experiment = experiments[args.experiment_index]
        name = experiment.get("name", f"experiment_{args.experiment_index}")
        print(f"\nRunning experiment {args.experiment_index}: {name}")

        experiment_configuration = base_configuration.with_yaml_overrides(experiment)
        results = train_and_evaluate(experiment_configuration)
        save_outputs(results, run_directory=batch_directory / name)


def main():
    parser = argparse.ArgumentParser(description="Run experiments on a SLURM cluster.")
    parser.add_argument("--configuration", type=str, required=True, help="Path to a YAML configuration file.")
    parser.add_argument("--worker", action="store_true", help="Worker mode: run a single experiment.")
    parser.add_argument("--experiment-index", type=int, help="Experiment index (worker mode).")
    parser.add_argument("--batch-directory", type=str, help="Batch directory (worker mode).")

    # === SLURM options (submit mode) ===
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition; omitted when not given.")
    parser.add_argument("--time", type=str, default="04:00:00")
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="1", help="Value for --gres=gpu:<value>, e.g. 1 or A100:1.")
    parser.add_argument("--repository-path", type=str, default=str(PROJECT_ROOT),
                        help="Path to this repository on the cluster (default: where this script lives).")
    parser.add_argument("--module", action="append", default=[], help="Module to `module load` in each job.")
    parser.add_argument("--setup-command", action="append", default=[], help="Shell line to run before loading modules.")
    parser.add_argument("--bundle", action="store_true", help="Submit all experiments in one SLURM job.")

    args = parser.parse_args()

    if args.worker:
        worker(args)
    else:
        submit(args)


if __name__ == "__main__":
    main()
