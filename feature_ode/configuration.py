"""Experiment configuration with YAML support."""

import copy
import itertools
import math
from pathlib import Path
from typing import Optional, List, Dict, Any

import attrs
import yaml

from feature_ode.activations import ACTIVATION_C_ELL, ACTIVATION_OPTIONS


MODEL_CLASS_OPTIONS = ("dpr", "feature_ode", "semi_frozen_network", "mlp")

# === Keys allowed in each YAML section ===
# Loading raises on any other key, so a typo cannot silently fall back to a default.
SECTION_KEYS = {
    "data": ("d", "batch_size", "powerlaw_alpha", "data_seed", "target_hermite", "target", "gammas"),
    "optimizer": (
        "learning_rate", "learning_rate_a", "learning_rate_b", "gradient_clip", "convergence_threshold",
        "max_epochs", "ode_step_size", "ode_convergence_threshold", "ode_max_steps",
    ),
    "model": (
        "model_class", "a_0", "width", "init_seeds", "orders", "c", "sigma_frozen",
        "hidden_variance_from_data", "mup_init", "activation", "subtract_init",
    ),
    "evaluate": (
        "a_elements_plot", "b_elements_plot", "smoothing_factor", "num_loss_prints", "max_plot_points",
        "neuron_histogram", "neuron_percentiles", "neuron_heatmap",
    ),
}
EXPERIMENT_KEYS = ("grid_search", "experiments", "name")
TUPLE_KEYS = ("gammas", "learning_rate_b", "init_seeds", "orders", "c")
NESTED_TUPLE_KEYS = ("a_elements_plot", "b_elements_plot")


def _as_tuple(key: str, value: Any) -> Any:
    """Convert YAML lists to the tuples the Configuration fields expect."""
    if value is None:
        return None
    if key in TUPLE_KEYS:
        return tuple(value)
    if key in NESTED_TUPLE_KEYS:
        return tuple(tuple(element) for element in value)
    return value


def _tuples_to_lists(value: Any) -> Any:
    """Recursively convert tuples to lists so saved YAML stays plain."""
    if isinstance(value, (list, tuple)):
        return [_tuples_to_lists(element) for element in value]
    if isinstance(value, dict):
        return {key: _tuples_to_lists(element) for key, element in value.items()}
    return value


@attrs.define
class Configuration:
    """Data, optimizer, model, and evaluation settings for one experiment."""

    # === Data ===
    d: int = 10
    batch_size: int = 1024
    powerlaw_alpha: float = 1.5
    data_seed: int = 0
    target_hermite: bool = False
    target: str = "x_{0}"  # e.g. "x_{0} x_{1} x_{2} + x_{5}"
    gammas: Optional[tuple[float, ...]] = None  # Custom eigenvalues; overrides powerlaw_alpha when set

    # === Optimizer ===
    learning_rate: float = 1e-3
    learning_rate_a: Optional[float] = None  # Only for DPR and the semi-frozen network (effective lr = learning_rate_a * learning_rate)
    learning_rate_b: Optional[tuple[float, ...]] = None  # Per-order B tensor lr multipliers for DPR (effective lr = learning_rate_b[i] * learning_rate)
    gradient_clip: Optional[float] = None  # Max gradient norm; None = no clipping
    convergence_threshold: float = 1e-6
    max_epochs: int = 100_000  # Number of SGD steps, one fresh batch each; -1 = twice the feature ODE's convergence time
    ode_step_size: float = 1e-3  # Step size for integrating the feature ODE (RK4)
    ode_convergence_threshold: float = 1e-6  # Loss threshold for feature ODE convergence (used to set max_epochs when max_epochs == -1)
    ode_max_steps: int = 10_000_000  # Maximum number of feature ODE integration steps

    # === Model ===
    a_0: Optional[float] = None  # Initial diagonal of A (DPR, semi-frozen network, feature ODE)
    model_class: str = "mlp"  # "dpr", "feature_ode", "semi_frozen_network", or "mlp"
    width: int = 4096  # Only for the MLP and the semi-frozen network
    init_seeds: tuple[int, ...] = (0,)  # Parallelized on a single device
    orders: Optional[tuple[int, ...]] = None  # Active polynomial orders for DPR, e.g. (0, 1, 2, 3, 4)
    c: Optional[tuple[float, ...]] = None  # Polynomial order coefficients for DPR, matched to orders
    sigma_frozen: Optional[float] = None  # Preactivation scale (overrides mup_init/hidden_variance_from_data for the semi-frozen network)
    hidden_variance_from_data: bool = False  # Semi-frozen network: set frozen hidden weight variance to min(gammas) / 2
    mup_init: bool = False  # μP initialization: a_0 = 1/sqrt(d), semi-frozen network hidden_std = 1/sqrt(d)
    activation: str = "cos"  # One of ACTIVATION_OPTIONS
    subtract_init: bool = False  # MLP: init output layer as N(0, 1/width) and subtract f(x;θ_0) for feature learning

    # === Evaluate ===
    a_elements_plot: tuple[tuple[int, int], ...] = ()  # A matrix elements to plot, e.g. ((0,0), (1,1))
    b_elements_plot: tuple[tuple[int, ...], ...] = ()  # B tensor elements to plot (DPR only); index length = order, e.g. ((0,1,0), (0,0), ())
    smoothing_factor: float = 0.5  # EMA smoothing factor for plots (0 = no smoothing, 1 = max smoothing)
    num_loss_prints: int = 10  # Number of times the loss is printed during training
    max_plot_points: int = 2000  # Maximum number of points plotted on the x-axis
    neuron_histogram: bool = False  # MLP/semi-frozen network: capture per-neuron <z_k^2> at init and end of training and plot histogram
    neuron_percentiles: bool = False  # MLP/semi-frozen network: track percentile traces of <z_k^2> across neurons over training and plot
    neuron_heatmap: bool = False  # MLP/semi-frozen network: store full per-neuron <z_k^2> at every recorded step and plot heatmap (rows = neurons sorted by final norm, cols = time)

    def __attrs_post_init__(self):
        """Validate the configuration after initialization."""
        if self.activation not in ACTIVATION_OPTIONS:
            raise ValueError(
                f"activation must be one of {ACTIVATION_OPTIONS}, got '{self.activation}'"
            )
        if self.model_class not in MODEL_CLASS_OPTIONS:
            raise ValueError(
                f"model_class must be one of {MODEL_CLASS_OPTIONS}, got '{self.model_class}'"
            )
        if self.orders is not None and self.c is not None and len(self.orders) != len(self.c):
            raise ValueError(
                f"orders and c must have the same length, got {len(self.orders)} and {len(self.c)}"
            )
        if self.orders is not None and self.learning_rate_b is not None and len(self.orders) != len(self.learning_rate_b):
            raise ValueError(
                f"orders and learning_rate_b must have the same length, got {len(self.orders)} and {len(self.learning_rate_b)}"
            )
        if self.gammas is not None and len(self.gammas) != self.d:
            raise ValueError(
                f"gammas must have length d={self.d}, got {len(self.gammas)}"
            )

    @staticmethod
    def _flatten_yaml_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a nested YAML dict (data, optimizer, model, evaluate sections) into flat keyword arguments.

        Works for both full configuration files and partial experiment overrides.
        Raises ValueError on unknown sections or keys.
        """
        unknown_sections = set(data) - set(SECTION_KEYS) - set(EXPERIMENT_KEYS)
        if unknown_sections:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown_sections)}")

        flat = {}
        for section, allowed_keys in SECTION_KEYS.items():
            for key, value in (data.get(section) or {}).items():
                if key not in allowed_keys:
                    raise ValueError(f"Unknown key '{key}' in configuration section '{section}'")
                flat[key] = _as_tuple(key, value)
        return flat

    @classmethod
    def from_yaml(cls, path: Path) -> "Configuration":
        """Load a configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**cls._flatten_yaml_dict(data))

    def get_a_0(self) -> float:
        """Return a_0, applying the μP default (1/sqrt(d)) if mup_init is set and a_0 is not explicitly specified."""
        if self.a_0 is not None:
            return self.a_0
        if self.mup_init:
            return 1.0 / math.sqrt(self.d)
        return 0.1

    def get_a_0_hidden_std(self) -> tuple:
        """Return (a_0, hidden_std) for the semi-frozen network, respecting sigma_frozen/mup_init."""
        a_0 = self.get_a_0()
        if self.sigma_frozen is not None:
            return a_0, self.sigma_frozen
        hidden_std = 1.0 / math.sqrt(self.d) if self.mup_init else 1.0
        if self.hidden_variance_from_data:
            gammas = self.get_gammas()
            hidden_std = math.sqrt(gammas.min().item() / 2.0)
        return a_0, hidden_std

    def get_c_ell(self, max_order: int = 10) -> tuple:
        """Return c_ell coefficients for the configured activation, up to max_order (inclusive)."""
        all_c = ACTIVATION_C_ELL[self.activation]
        if max_order >= len(all_c):
            raise ValueError(
                f"activation '{self.activation}' only has c_ell up to order {len(all_c) - 1}, "
                f"but order {max_order} was requested"
            )
        return all_c[:max_order + 1]

    def get_gammas(self) -> "torch.Tensor":
        """Return eigenvalues as a tensor. Uses custom gammas if set, otherwise a power law."""
        import torch
        if self.gammas is not None:
            return torch.tensor(self.gammas, dtype=torch.float64)
        from feature_ode.data import generate_covariance_eigenvalues
        return generate_covariance_eigenvalues(self.d, self.powerlaw_alpha)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict of lists and scalars, safe for YAML and JSON."""
        result = _tuples_to_lists(attrs.asdict(self))
        # Always include resolved eigenvalues for reproducibility
        result["gammas_resolved"] = self.get_gammas().tolist()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Configuration":
        """Create from a dict produced by to_dict. Raises ValueError on unknown keys."""
        d = {key: value for key, value in d.items() if key != "gammas_resolved"}
        valid_keys = {f.name for f in attrs.fields(cls)}
        unknown_keys = set(d) - valid_keys
        if unknown_keys:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown_keys)}")
        return cls(**{key: _as_tuple(key, value) for key, value in d.items()})

    def with_overrides(self, overrides: Dict[str, Any]) -> "Configuration":
        """Create a new configuration with the given flat overrides."""
        return attrs.evolve(self, **overrides)

    def with_yaml_overrides(self, yaml_overrides: Dict[str, Any]) -> "Configuration":
        """Create a new configuration from nested YAML-style overrides.

        Accepts the same nested section format as the YAML files
        (e.g. {"data": {"powerlaw_alpha": 2.0}}) and flattens it before applying.
        """
        return self.with_overrides(self._flatten_yaml_dict(yaml_overrides))


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base, returning a new dict.

    For nested dicts, merges keys. For everything else, override wins.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def expand_grid_search(axes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand a grid_search specification into a flat list of experiment dicts.

    Each axis is a dict with:
        - axis: str — name of this axis (used in experiment names)
        - values: list of dicts — each dict has an optional 'label' key
          and the rest are nested YAML-style overrides.

    Returns a list of experiment dicts (with 'name' key) representing
    the Cartesian product of all axes.
    """
    # === Parse axes into (axis_name, [(label, overrides), ...]) ===
    parsed_axes = []
    for axis_spec in axes:
        axis_name = axis_spec["axis"]
        values = axis_spec["values"]
        parsed_values = []
        for i, value_dict in enumerate(values):
            value_dict = copy.deepcopy(value_dict)
            label = value_dict.pop("label", f"{axis_name}_{i}")
            parsed_values.append((label, value_dict))
        parsed_axes.append(parsed_values)

    # === Take Cartesian product and merge overrides ===
    experiments = []
    for combination in itertools.product(*parsed_axes):
        labels = []
        merged_overrides = {}
        for label, overrides in combination:
            labels.append(label)
            merged_overrides = _deep_merge_dicts(merged_overrides, overrides)
        merged_overrides["name"] = "_".join(labels)
        experiments.append(merged_overrides)

    return experiments


def load_experiment_configuration(path: Path) -> tuple[Configuration, Optional[List[Dict]]]:
    """Load a base configuration and an optional experiments list from YAML.

    Supports three modes:
        1. Single experiment (no experiments or grid_search key)
        2. Explicit experiments list (experiments key)
        3. Grid search (grid_search key) — expands Cartesian product into experiments

    Returns:
        (base_configuration, experiments) where experiments is None for single-run files
        or a list of dicts with overrides for multi-experiment files.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    base_configuration = Configuration.from_yaml(path)

    # Grid search takes priority if present
    if "grid_search" in data:
        experiments = expand_grid_search(data["grid_search"])
    else:
        experiments = data.get("experiments", None)

    # Validate every experiment's overrides now, before any training starts
    for experiment in experiments or []:
        base_configuration.with_yaml_overrides(experiment)

    return base_configuration, experiments
