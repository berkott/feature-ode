"""Training loop that runs several init seeds in parallel with torch.func.vmap.

Every model (MLP, semi-frozen network, DPR) trains with online SGD on a fresh
batch of Gaussian data at each step. The feature ODE has its own integrator in
feature_ode.ode.
"""

import math
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.func import functional_call, vmap, stack_module_state, grad

from feature_ode.activations import build_nonlinearity
from feature_ode.configuration import Configuration
from feature_ode.data import parse_target_string, evaluate_target
from feature_ode.models.deep_polynomial_regression import DPR
from feature_ode.models.mlp import MLP
from feature_ode.models.semi_frozen_network import SemiFrozenNetwork
from feature_ode.ode import compute_convergence_time


# === Model creation ===

def create_model(configuration: Configuration) -> nn.Module:
    """Instantiate the model specified by configuration.model_class."""
    a_0 = configuration.get_a_0()

    if configuration.model_class == "dpr":
        model_configuration = {
            "hidden_dim": configuration.d,
            "a_0": a_0,
        }
        if configuration.orders is not None:
            model_configuration["orders"] = list(configuration.orders)
        # Use explicit c if provided, otherwise derive from activation
        orders = model_configuration.get("orders", list(range(5)))
        if configuration.c is not None:
            c_orders = configuration.orders if configuration.orders is not None else range(len(configuration.c))
            model_configuration["c"] = {o: v for o, v in zip(c_orders, configuration.c)}
        else:
            c_ell = configuration.get_c_ell(max_order=max(orders))
            model_configuration["c"] = {o: c_ell[o] for o in orders}
        if configuration.sigma_frozen is not None:
            # Scale c values by sigma_frozen^order to match semi-frozen network hidden variance
            existing_c = model_configuration["c"]
            model_configuration["c"] = {o: v * configuration.sigma_frozen ** o for o, v in existing_c.items()}
        return DPR(input_dim=configuration.d, model_configuration=model_configuration)

    elif configuration.model_class == "semi_frozen_network":
        a_0, hidden_std = configuration.get_a_0_hidden_std()
        nonlinearity = build_nonlinearity(configuration.activation)
        return SemiFrozenNetwork(d_in=configuration.d, width=configuration.width, d_out=1, a_0=a_0,
                                 hidden_std=hidden_std, nonlinearity=nonlinearity)

    elif configuration.model_class == "mlp":
        nonlinearity = build_nonlinearity(configuration.activation)
        # Keep W1 ~ N(0, 1/d) so sqrt(NFM) starts at a_0 = 1/sqrt(d). sigma_frozen
        # is applied as a preactivation multiplier inside the model forward pass,
        # not folded into hidden_std.
        hidden_std = 1.0 / math.sqrt(configuration.d) if configuration.mup_init else 1.0
        return MLP(d_in=configuration.d, width=configuration.width, d_out=1, nonlinearity=nonlinearity,
                   hidden_std=hidden_std, sigma_frozen=configuration.sigma_frozen,
                   subtract_init=configuration.subtract_init)

    else:
        raise ValueError(f"Unknown model_class: {configuration.model_class}")


# === Parameter classification ===

def get_a_parameter_names(configuration: Configuration) -> set:
    """Return names of the A matrix parameters (these get learning_rate_a * learning_rate)."""
    if configuration.model_class == "dpr":
        return {"A"}
    elif configuration.model_class == "semi_frozen_network":
        return {"input_layer.weight"}
    else:
        return set()


# === Per-neuron preactivation variance ===

def compute_per_neuron_preactivation_variance(
    model, gammas: torch.Tensor, model_class: str
) -> torch.Tensor:
    """Per-neuron preactivation variance <z_k^2> across the data distribution.

    For MLP:  z_k = sigma_frozen * sum_j W1[k,j] x_j  ->  <z_k^2> = sigma_frozen^2 * sum_j W1[k,j]^2 gamma_j.
    For semi-frozen network: z_k = sum_j (F @ A)[k,j] x_j (sigma_frozen is folded into F's std)
                     ->  <z_k^2> = sum_j (F @ A)[k,j]^2 gamma_j.

    Returns a length-N tensor on the model's device. None for unsupported model_class.
    """
    if model_class not in ("mlp", "semi_frozen_network"):
        return None
    with torch.no_grad():
        device = next(model.parameters()).device
        gammas_device = gammas.to(device=device, dtype=torch.float32)
        if model_class == "mlp":
            input_weight = model.input_layer.weight.detach()  # (N, d)
            sigma_frozen = model.sigma_frozen if model.sigma_frozen is not None else 1.0
            preactivation_matrix = input_weight  # (N, d)
            sigma_squared = sigma_frozen ** 2
        else:  # semi-frozen network
            hidden_weight = model.hidden_layer.weight.detach()  # (N, d)
            input_weight = model.input_layer.weight.detach()    # (d, d)
            preactivation_matrix = hidden_weight @ input_weight  # (N, d)
            sigma_squared = 1.0
        return sigma_squared * (preactivation_matrix.pow(2) * gammas_device.unsqueeze(0)).sum(dim=1)


def compute_per_neuron_preactivation_variance_stacked(
    stacked_params: Dict[str, torch.Tensor],
    gammas: torch.Tensor,
    sigma_frozen,
    model_class: str,
) -> torch.Tensor:
    """Stacked-params version of compute_per_neuron_preactivation_variance for vmap path.

    Returns shape (num_models, N). None for unsupported model_class.
    """
    if model_class not in ("mlp", "semi_frozen_network"):
        return None
    if model_class == "mlp":
        input_weight = stacked_params["input_layer.weight"]  # (M, N, d)
        sigma_squared = (sigma_frozen ** 2) if sigma_frozen is not None else 1.0
        preactivation_matrix = input_weight
    else:  # semi-frozen network
        hidden_weight = stacked_params["hidden_layer.weight"]  # (M, N, d)
        input_weight = stacked_params["input_layer.weight"]    # (M, d, d)
        preactivation_matrix = torch.matmul(hidden_weight, input_weight)  # (M, N, d)
        sigma_squared = 1.0
    gammas_device = gammas.to(device=preactivation_matrix.device, dtype=preactivation_matrix.dtype)
    return sigma_squared * (preactivation_matrix.pow(2) * gammas_device.view(1, 1, -1)).sum(dim=-1)


# === Training ===

def train(configuration: Configuration) -> Dict[str, Any]:
    """Train multiple models in parallel using vmap over init seeds.

    Uses torch.func.grad + vmap for parallel forward and gradient
    computation. Parameter updates are manual SGD (param -= lr * grad),
    also fully parallel across models.

    Returns:
        Dict with keys:
            - "loss_history": {init_seed: [loss_per_epoch]}
            - "a_history": {init_seed: {(row, col): [value_per_epoch]}}
            - "config": configuration.to_dict()
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    has_a_matrix = configuration.model_class in ("dpr", "semi_frozen_network")
    init_seeds = list(configuration.init_seeds)
    num_models = len(init_seeds)

    # === Auto-determine max_epochs from ODE convergence time ===
    max_epochs = configuration.max_epochs
    if max_epochs == -1:
        ode_time = compute_convergence_time(configuration)
        max_epochs = math.ceil(2 * ode_time / configuration.learning_rate)
        print(f"  Auto max_epochs = ceil(2 * {ode_time:.2f} / {configuration.learning_rate}) = {max_epochs}")

    # === Precompute data generation constants ===
    gammas = configuration.get_gammas()
    standard_deviations = gammas.sqrt().float().to(device)
    monomials = parse_target_string(configuration.target)
    data_generator = torch.Generator(device=device).manual_seed(configuration.data_seed)
    gammas_device = gammas.float().to(device)

    def sample_batch():
        z = torch.randn(configuration.batch_size, configuration.d, generator=data_generator, device=device)
        x = z * standard_deviations.unsqueeze(0)
        y = evaluate_target(x, monomials, configuration.target_hermite, gammas_device)
        return x, y

    # === Create one model per init seed ===
    models = []
    for init_seed in init_seeds:
        torch.manual_seed(init_seed)
        model = create_model(configuration).to(device)
        models.append(model)

    base_model = models[0]

    # === Determine which params are trainable and their learning rates ===
    effective_learning_rate_a = (configuration.learning_rate_a or 1.0) * configuration.learning_rate
    a_param_names = get_a_parameter_names(configuration)

    # Build per-order B tensor learning rate map: "Bs.<order>" -> effective lr
    # Each order gets an automatic p! factor to compensate the 1/p! in the DPR
    # forward pass (c/p! shrinks B gradients; p! in lr restores the balance).
    b_param_learning_rates = {}
    if configuration.model_class == "dpr":
        orders = configuration.orders if configuration.orders is not None else tuple(range(5))
        for i, order in enumerate(orders):
            lr_multiplier = 1.0
            if configuration.learning_rate_b is not None:
                lr_multiplier = configuration.learning_rate_b[i]
            b_param_learning_rates[f"Bs.{order}"] = (
                lr_multiplier * math.factorial(order) * configuration.learning_rate
            )

    trainable_param_names = set()
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            trainable_param_names.add(name)

    # === Stack parameters for vmap ===
    stacked_params, stacked_buffers = stack_module_state(models)

    # Output layer lr = lr/N for semi-frozen network and MLP (both without 1/N in
    # forward): lr/N on an output gradient that sums N O(1) terms keeps the
    # per-step function change Δf = O(η).
    # MLP also gets lr*N on its input layer W1. Under μP (W2 ~ N(0, 1/N^2)),
    # the W1 gradient is O(1/N) per entry; lr*N·(1/N) = η gives per-entry
    # feature learning O(η) per step, independent of N.
    output_layer_lr = None
    if configuration.model_class in ("semi_frozen_network", "mlp"):
        output_layer_lr = configuration.learning_rate / configuration.width
    mlp_input_layer_lr = None
    if configuration.model_class == "mlp":
        mlp_input_layer_lr = configuration.learning_rate * configuration.width

    param_learning_rates = {}
    for name in stacked_params:
        if name not in trainable_param_names:
            param_learning_rates[name] = 0.0
        elif name in a_param_names:
            param_learning_rates[name] = effective_learning_rate_a
        elif name in b_param_learning_rates:
            param_learning_rates[name] = b_param_learning_rates[name]
        elif output_layer_lr is not None and name == "output_layer.weight":
            param_learning_rates[name] = output_layer_lr
        elif mlp_input_layer_lr is not None and name == "input_layer.weight":
            param_learning_rates[name] = mlp_input_layer_lr
        else:
            param_learning_rates[name] = configuration.learning_rate

    # === Define vmapped loss + grad function ===
    # x and y are sampled fresh each epoch and stored in these lists
    # so the closures always see the current batch.
    current_batch = [None, None]  # [x, y]

    def compute_loss(params, buffers):
        """Compute MSE loss for a single model given its params/buffers."""
        y_predicted = functional_call(base_model, (params, buffers), args=(current_batch[0],))
        y_predicted = y_predicted.reshape(-1)
        residuals = y_predicted - current_batch[1]
        loss = 0.5 * (residuals * residuals).mean()
        return loss, loss.detach()

    grad_fn = grad(compute_loss, argnums=0, has_aux=True)
    batched_grad_fn = vmap(grad_fn, in_dims=(0, 0), randomness="different")

    # === History tracking ===
    loss_history = {seed: [] for seed in init_seeds}

    track_a = bool(configuration.a_elements_plot)
    a_history = (
        {seed: {(r, c): [] for r, c in configuration.a_elements_plot} for seed in init_seeds}
        if track_a else {}
    )

    track_b = configuration.model_class == "dpr" and bool(configuration.b_elements_plot)
    b_history = (
        {seed: {tuple(idx): [] for idx in configuration.b_elements_plot} for seed in init_seeds}
        if track_b else {}
    )

    # Output-layer norm tracking: one scalar per step per seed.
    # MLP/semi-frozen network: ||W_out||. DPR: ||B^(ell)|| per order.
    if configuration.model_class in ("semi_frozen_network", "mlp"):
        output_norm_history = {seed: [] for seed in init_seeds}
    elif configuration.model_class == "dpr":
        _orders = configuration.orders if configuration.orders is not None else tuple(range(5))
        output_norm_history = {seed: {int(o): [] for o in _orders} for seed in init_seeds}
    else:
        output_norm_history = {}

    # === Per-neuron <z^2> tracking (MLP/semi-frozen network only) ===
    is_neuron_model = configuration.model_class in ("mlp", "semi_frozen_network")
    track_neuron_histogram = configuration.neuron_histogram and is_neuron_model
    track_neuron_percentiles = configuration.neuron_percentiles and is_neuron_model
    track_neuron_heatmap = configuration.neuron_heatmap and is_neuron_model
    track_per_step_z2 = track_neuron_percentiles or track_neuron_heatmap
    PERCENTILE_PROBABILITIES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    neuron_z2_init = {}
    neuron_z2_final = {}
    neuron_z2_percentiles_history = (
        {seed: [] for seed in init_seeds} if track_neuron_percentiles else {}
    )
    # Per-seed buffers of per-step (N,) tensors; stacked at the end into (T, N) numpy arrays.
    neuron_z2_full_buffers = (
        {seed: [] for seed in init_seeds} if track_neuron_heatmap else {}
    )
    neuron_z2_full_history = {}
    if track_neuron_percentiles:
        percentile_probabilities_tensor = torch.tensor(
            PERCENTILE_PROBABILITIES, device=device, dtype=torch.float32
        )
    if track_neuron_histogram:
        with torch.no_grad():
            initial_z2 = compute_per_neuron_preactivation_variance_stacked(
                stacked_params, gammas_device, configuration.sigma_frozen, configuration.model_class
            )  # (M, N)
            initial_cpu = initial_z2.cpu()
            for i, init_seed in enumerate(init_seeds):
                neuron_z2_init[init_seed] = initial_cpu[i].tolist()

    print_interval = max(1, max_epochs // configuration.num_loss_prints)

    print(f"Training {num_models} models in parallel with vmap ({max_epochs} epochs)")

    # === Precompute A-element indices for vectorized extraction ===
    a_rows = [r for r, c in configuration.a_elements_plot] if track_a else []
    a_cols = [c for r, c in configuration.a_elements_plot] if track_a else []

    for epoch in range(max_epochs):
        # === Sample fresh batch ===
        current_batch[0], current_batch[1] = sample_batch()

        # === Compute gradients + loss in one pass (loss returned as aux) ===
        all_grads, all_losses = batched_grad_fn(stacked_params, stacked_buffers)

        # === Record losses (batch transfer to CPU, no per-element .item()) ===
        loss_cpu = all_losses.detach().cpu()
        for i, init_seed in enumerate(init_seeds):
            loss_history[init_seed].append(loss_cpu[i].item())

        # === Track A elements (vectorized across seeds) ===
        if track_a:
            with torch.no_grad():
                if configuration.model_class == "dpr":
                    a_all = stacked_params["A"].detach()  # (num_models, d, d)
                elif configuration.model_class == "semi_frozen_network":
                    a_all = stacked_params["input_layer.weight"].detach()
                elif configuration.model_class == "mlp":
                    w = stacked_params["input_layer.weight"].detach()  # (num_models, N, d)
                    # Compute NFM elements: nfm[r,c] = (1/N) * w[:,r] @ w[:,c]. For
                    # diagonals, plot sqrt(nfm) so the init matches a_0 = 1/sqrt(d)
                    # and is comparable to semi-frozen network A / DPR B^(1) diagonals.
                    a_vals_list = []
                    for r, c in configuration.a_elements_plot:
                        nfm = (w[:, :, r] * w[:, :, c]).sum(dim=1) / configuration.width
                        if r == c:
                            # clamp to avoid sqrt of tiny negative from float noise
                            nfm = nfm.clamp(min=0.0).sqrt()
                        a_vals_list.append(nfm)
                    a_vals = torch.stack(a_vals_list, dim=1)  # (num_models, num_elements)
                    a_vals_cpu = a_vals.cpu()
                    for i, init_seed in enumerate(init_seeds):
                        for j, (row, col) in enumerate(configuration.a_elements_plot):
                            a_history[init_seed][(row, col)].append(a_vals_cpu[i, j].item())
                    a_all = None  # skip the generic path

                if a_all is not None:
                    a_vals_cpu = a_all[:, a_rows, a_cols].cpu()  # (num_models, num_elements)
                    for i, init_seed in enumerate(init_seeds):
                        for j, (row, col) in enumerate(configuration.a_elements_plot):
                            a_history[init_seed][(row, col)].append(a_vals_cpu[i, j].item())

        # === Track B elements (DPR only) ===
        if track_b:
            with torch.no_grad():
                for indices in configuration.b_elements_plot:
                    order = len(indices)
                    b_stacked = stacked_params[f"Bs.{order}"].detach()
                    if indices:
                        vals_cpu = b_stacked[(slice(None),) + tuple(indices)].cpu()
                    else:
                        vals_cpu = b_stacked.cpu()
                    for i, init_seed in enumerate(init_seeds):
                        b_history[init_seed][tuple(indices)].append(vals_cpu[i].item())

        # === Track output-layer norm (scalar per seed) ===
        with torch.no_grad():
            if configuration.model_class in ("semi_frozen_network", "mlp"):
                w = stacked_params["output_layer.weight"].detach()
                norms = w.reshape(num_models, -1).norm(dim=1).cpu()
                for i, init_seed in enumerate(init_seeds):
                    output_norm_history[init_seed].append(norms[i].item())
            elif configuration.model_class == "dpr":
                for order in _orders:
                    b = stacked_params[f"Bs.{order}"].detach()
                    norms = b.reshape(num_models, -1).norm(dim=1).cpu()
                    for i, init_seed in enumerate(init_seeds):
                        output_norm_history[init_seed][int(order)].append(norms[i].item())

            # === Per-neuron <z^2>: percentiles and/or full vector ===
            if track_per_step_z2:
                z2_step = compute_per_neuron_preactivation_variance_stacked(
                    stacked_params, gammas_device, configuration.sigma_frozen, configuration.model_class
                )  # (M, N)
                if track_neuron_percentiles:
                    # torch.quantile over the neuron dim returns shape (P, M); transpose to (M, P).
                    percentiles_per_seed = torch.quantile(
                        z2_step, percentile_probabilities_tensor, dim=-1
                    ).transpose(0, 1).cpu()  # (M, P)
                    for i, init_seed in enumerate(init_seeds):
                        neuron_z2_percentiles_history[init_seed].append(
                            percentiles_per_seed[i].tolist()
                        )
                if track_neuron_heatmap:
                    z2_step_cpu = z2_step.detach().cpu()
                    for i, init_seed in enumerate(init_seeds):
                        neuron_z2_full_buffers[init_seed].append(z2_step_cpu[i].clone())

        # === Print progress ===
        mean_loss = loss_cpu.mean().item()
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{max_epochs} | mean loss={mean_loss:.6e}")

        # === Check convergence (threshold on absolute loss) ===
        if configuration.convergence_threshold > 0:
            if (loss_cpu < configuration.convergence_threshold).all():
                print(f"  All models converged at epoch {epoch + 1} (mean loss={mean_loss:.6e})")
                break

        # === Gradient clipping (per-model norm clipping across stacked params) ===
        if configuration.gradient_clip is not None:
            grad_norms = torch.zeros(num_models, device=device)
            for name in all_grads:
                grad_norms += all_grads[name].reshape(num_models, -1).pow(2).sum(dim=1)
            grad_norms = grad_norms.sqrt()
            clip_scale = (configuration.gradient_clip / grad_norms.clamp(min=configuration.gradient_clip)).unsqueeze(-1)
            all_grads = {
                name: all_grads[name] * clip_scale.reshape(num_models, *([1] * (all_grads[name].dim() - 1)))
                for name in all_grads
            }

        # === SGD update (fully parallel across models) ===
        stacked_params = {
            name: (stacked_params[name] - param_learning_rates[name] * all_grads[name]).detach()
            for name in stacked_params
        }

    else:
        mean_loss = loss_cpu.mean().item()
        print(f"  Hit max epochs {max_epochs} (mean loss={mean_loss:.6e})")

    # === Capture per-neuron <z^2> at end of training ===
    if track_neuron_histogram:
        with torch.no_grad():
            final_z2 = compute_per_neuron_preactivation_variance_stacked(
                stacked_params, gammas_device, configuration.sigma_frozen, configuration.model_class
            )  # (M, N)
            final_cpu = final_z2.cpu()
            for i, init_seed in enumerate(init_seeds):
                neuron_z2_final[init_seed] = final_cpu[i].tolist()
    if track_neuron_heatmap:
        for init_seed in init_seeds:
            buffer = neuron_z2_full_buffers[init_seed]
            neuron_z2_full_history[init_seed] = (
                torch.stack(buffer).numpy() if buffer else None
            )

    return {
        "loss_history": loss_history,
        "a_history": a_history,
        "b_history": b_history,
        "output_norm_history": output_norm_history,
        "neuron_z2_init": neuron_z2_init,
        "neuron_z2_final": neuron_z2_final,
        "neuron_z2_percentile_probabilities": list(PERCENTILE_PROBABILITIES),
        "neuron_z2_percentiles_history": neuron_z2_percentiles_history,
        "neuron_z2_full_history": neuron_z2_full_history,
        # Stored under "config" for compatibility with the recorded runs behind the paper's figures.
        "config": configuration.to_dict(),
    }
