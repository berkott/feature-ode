"""The feature ODE: gradient flow of DPR reduced to a few scalar variables, integrated with RK4.

Assembles the population gradient flow for DPR from any target function
(a sum of monomials). The state is [a_1, ..., a_d, b_1, ..., b_J], where J is
the number of monomial terms in the target. No autograd, no data, and no
randomness are involved: the derivatives are analytical and integrated with RK4.

The loss being minimized is:
    L = sum_j (1/2)(beta_j - rate_constant_j * prod_i a_i^{alpha_j_i} * b_j)^2

where rate_constant_j = c_{|alpha_j|} * sigma_frozen^|alpha_j| * sqrt(prod_i gamma_i^{alpha_j_i})
and beta_j is the target coefficient on monomial j (defaults to 1).
This is the feature ODE of Section 3 of the paper, with rate_constant_j playing the role of c̃_α.
"""

import math
from typing import Dict, Any

import numpy as np

from feature_ode.configuration import Configuration
from feature_ode.data import parse_target_string


class FeatureODE:
    """The feature ODE for one target function.

    Args:
        d: Input dimension.
        target: Target function string, e.g. "x_{0} x_{1} x_{2} + x_{5}".
        gammas: Covariance eigenvalues, array of shape (d,).
        a_0: Initial value for all a_i.
        sigma_frozen: Preactivation scale; each order-ell term picks up sigma_frozen^ell.
        c_ell: Activation coefficients indexed by polynomial order; None means 1 for every order.
    """

    def __init__(self, d, target, gammas, a_0, sigma_frozen=1.0, c_ell=None):
        self.d = d
        self.gammas = np.array(gammas, dtype=np.float64)

        # === Parse target into monomials ===
        self.monomials = parse_target_string(target)
        self.num_terms = len(self.monomials)

        # === Build multi-index alpha and target coefficient for each monomial term ===
        # alpha_j[i] = power of variable i in term j; target_coefficients[j] = beta_j
        self.alphas = []
        self.target_coefficients = np.zeros(self.num_terms, dtype=np.float64)
        for j, (coefficient, monomial) in enumerate(self.monomials):
            alpha = np.zeros(d, dtype=np.float64)
            for variable_index, power in monomial:
                alpha[variable_index] = power
            self.alphas.append(alpha)
            self.target_coefficients[j] = coefficient

        # === Compute rate constants ===
        # rate_constant_j = c_{|alpha_j|} * sigma_frozen^|alpha_j| * sqrt(prod_i gamma_i^{alpha_j_i})
        # The 1/order! factorial scaling cancels with the order! learning rate
        # multiplier on the B tensors, so neither appears in the rate constant.
        # The sigma_frozen factor accounts for the preactivation scale of the
        # MLP and the semi-frozen network.
        # Convention: the paper writes sqrt(c_ell) here, while this code uses c_ell.
        # The two agree for the cosine activation (c_ell = 1), which every run in
        # configurations/ uses except Figure 3(b) (shifted tanh). We keep c_ell so the
        # code reproduces the published curves exactly.
        self.rate_constants = np.zeros(self.num_terms, dtype=np.float64)
        for j, alpha in enumerate(self.alphas):
            total_degree = int(np.sum(alpha))
            gamma_product = np.prod(self.gammas ** alpha)
            c_factor = 1.0 if c_ell is None else float(c_ell[total_degree])
            self.rate_constants[j] = c_factor * (sigma_frozen ** total_degree) * np.sqrt(gamma_product)

        # === Initialize state ===
        # b starts slightly above zero so the dynamics can leave the saddle at b = 0.
        self.a = np.full(d, float(a_0), dtype=np.float64)
        self.b = np.full(self.num_terms, 1e-10, dtype=np.float64)

    def _compute_derivatives_from_state(self, a, b):
        """Compute da/dt and db/dt for arbitrary state (a, b).

        Used by the RK4 solver, which evaluates derivatives at intermediate points.
        """
        # === Compute a products ===
        a_products = np.zeros(self.num_terms, dtype=np.float64)
        for j, alpha in enumerate(self.alphas):
            a_products[j] = np.prod(a ** alpha)

        residuals = self.target_coefficients - self.rate_constants * a_products * b

        # === db/dt ===
        db = residuals * self.rate_constants * a_products

        # === da/dt ===
        da = np.zeros(self.d, dtype=np.float64)
        for j, alpha in enumerate(self.alphas):
            for i in range(self.d):
                if alpha[i] == 0:
                    continue
                if a[i] == 0:
                    continue
                da[i] += (residuals[j] * self.rate_constants[j]
                          * alpha[i] * a_products[j] / a[i]
                          * b[j])

        return da, db

    def compute_loss(self):
        """Compute L = sum_j (1/2)(beta_j - rate_constant_j * a_product_j * b_j)^2."""
        a_products = np.zeros(self.num_terms, dtype=np.float64)
        for j, alpha in enumerate(self.alphas):
            a_products[j] = np.prod(self.a ** alpha)
        residuals = self.target_coefficients - self.rate_constants * a_products * self.b
        return 0.5 * np.sum(residuals ** 2)

    def step(self, dt):
        """One RK4 step."""
        a, b = self.a, self.b

        # k1
        da1, db1 = self._compute_derivatives_from_state(a, b)

        # k2
        da2, db2 = self._compute_derivatives_from_state(
            a + 0.5 * dt * da1, b + 0.5 * dt * db1)

        # k3
        da3, db3 = self._compute_derivatives_from_state(
            a + 0.5 * dt * da2, b + 0.5 * dt * db2)

        # k4
        da4, db4 = self._compute_derivatives_from_state(
            a + dt * da3, b + dt * db3)

        self.a = a + (dt / 6.0) * (da1 + 2*da2 + 2*da3 + da4)
        self.b = b + (dt / 6.0) * (db1 + 2*db2 + 2*db3 + db4)


def build_feature_ode(configuration: Configuration) -> FeatureODE:
    """Build the feature ODE that matches an experiment configuration."""
    gammas = configuration.get_gammas()
    max_order = max(configuration.orders) if configuration.orders else 4
    c_ell = configuration.get_c_ell(max_order=max_order)
    return FeatureODE(
        d=configuration.d,
        target=configuration.target,
        gammas=gammas.numpy(),
        a_0=configuration.get_a_0(),
        sigma_frozen=configuration.sigma_frozen or 1.0,
        c_ell=c_ell,
    )


def compute_convergence_time(configuration: Configuration) -> float:
    """Run the feature ODE until convergence and return the effective time taken.

    Used when max_epochs == -1 to set the training length automatically.
    Convergence is detected when loss < ode_convergence_threshold.
    """
    model = build_feature_ode(configuration)

    step_size = configuration.ode_step_size
    ode_threshold = configuration.ode_convergence_threshold
    max_ode_steps = configuration.ode_max_steps
    current_time = 0.0
    print_interval = max(1, max_ode_steps // 10)

    print(f"  Computing ODE convergence time (up to {max_ode_steps} steps, threshold={ode_threshold:.1e})...")

    for step in range(max_ode_steps):
        loss = model.compute_loss()
        if loss < ode_threshold:
            break
        if step % print_interval == 0:
            print(f"    ODE step {step}/{max_ode_steps} | loss={loss:.6e} | time={current_time:.1f}")
        model.step(step_size)
        current_time += step_size

    print(f"  ODE converged at effective time {current_time:.2f} (loss={model.compute_loss():.2e})")
    return current_time


def compute_theory_curves(
    configuration: Configuration,
    total_effective_time: float,
) -> Dict[str, Any]:
    """Run the feature ODE to generate theory curves for plotting.

    Records the ODE state at every ODE step (no resampling), and returns
    the actual effective times alongside the values.

    Args:
        configuration: Configuration with target, d, powerlaw_alpha, ode_step_size, etc.
        total_effective_time: Total effective time to integrate over.

    Returns:
        Dict with:
            - "effective_times": list of effective time values (one per ODE step)
            - "loss_history": list of loss values at each ODE step
            - "a_history": {(row, col): [values]} for tracked elements
            - "b_history": {(indices_tuple): [values]} for tracked elements
    """
    model = build_feature_ode(configuration)

    step_size = configuration.ode_step_size
    num_ode_steps = math.ceil(total_effective_time / step_size)

    print(f"  Feature ODE theory: rate_constants={model.rate_constants}, "
          f"a_0={model.a[0]:.4f}, b_0={model.b[0]:.2e}, step_size={step_size}, "
          f"total_time={total_effective_time:.1f}, num_ode_steps={num_ode_steps}")

    effective_times = []
    loss_history = []
    a_history = {}
    b_history = {}

    if configuration.a_elements_plot:
        for row, col in configuration.a_elements_plot:
            a_history[(row, col)] = []

    if configuration.b_elements_plot:
        for indices in configuration.b_elements_plot:
            b_history[tuple(indices)] = []

    def record_sample(t):
        effective_times.append(t)
        loss_history.append(model.compute_loss())
        for row, col in a_history:
            if row == col and row < configuration.d:
                a_history[(row, col)].append(float(model.a[row]))
            else:
                a_history[(row, col)].append(0.0)
        for indices_tuple in b_history:
            term_index = indices_tuple[0] if len(indices_tuple) == 1 else 0
            if term_index < model.num_terms:
                b_history[indices_tuple].append(float(model.b[term_index]))
            else:
                b_history[indices_tuple].append(0.0)

    current_time = 0.0

    for ode_step in range(num_ode_steps):
        record_sample(current_time)
        model.step(step_size)
        current_time += step_size

    # Record final state
    record_sample(current_time)

    return {
        "effective_times": effective_times,
        "loss_history": loss_history,
        "a_history": a_history,
        "b_history": b_history,
    }


def train(configuration: Configuration) -> Dict[str, Any]:
    """Integrate the feature ODE with RK4 as if it were a model being trained (model_class "feature_ode").

    Returns:
        Dict with keys:
            - "loss_history": {0: [loss_per_step]}
            - "a_history": {0: {(row, col): [value_per_step]}}
            - "b_history": {0: {(term_index,): [value_per_step]}}
            - "config": configuration.to_dict()
    """
    model = build_feature_ode(configuration)

    # === Set max_epochs from ODE convergence ===
    max_epochs = configuration.max_epochs
    if max_epochs == -1:
        # Use a large number of steps and rely on the convergence threshold
        max_epochs = configuration.ode_max_steps

    # === History tracking ===
    seed = 0
    loss_history = {seed: []}

    track_a = bool(configuration.a_elements_plot)
    a_history = (
        {seed: {(r, c): [] for r, c in configuration.a_elements_plot}}
        if track_a else {}
    )

    track_b = bool(configuration.b_elements_plot)
    b_history = (
        {seed: {tuple(idx): [] for idx in configuration.b_elements_plot}}
        if track_b else {}
    )

    print_interval = max(1, max_epochs // configuration.num_loss_prints)
    print(f"Integrating the feature ODE with RK4 ({max_epochs} steps)")

    for epoch in range(max_epochs):
        loss = model.compute_loss()
        loss_history[seed].append(loss)

        # === Track A elements (a_i values on the diagonal) ===
        if track_a:
            for row, col in configuration.a_elements_plot:
                if row == col and row < configuration.d:
                    a_history[seed][(row, col)].append(model.a[row])
                else:
                    a_history[seed][(row, col)].append(0.0)

        # === Track B elements (b_j values) ===
        if track_b:
            for indices in configuration.b_elements_plot:
                term_index = indices[0] if len(indices) == 1 else 0
                if term_index < model.num_terms:
                    b_history[seed][tuple(indices)].append(model.b[term_index])
                else:
                    b_history[seed][tuple(indices)].append(0.0)

        # === Print progress ===
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Step {epoch + 1}/{max_epochs} | loss={loss:.6e}")

        # === Check convergence ===
        if configuration.convergence_threshold > 0 and loss < configuration.convergence_threshold:
            print(f"  Converged at step {epoch + 1} (loss={loss:.6e})")
            break

        # === RK4 step ===
        model.step(configuration.ode_step_size)

    else:
        print(f"  Hit max steps {max_epochs} (loss={loss:.6e})")

    return {
        "loss_history": loss_history,
        "a_history": a_history,
        "b_history": b_history,
        # Stored under "config" for compatibility with the recorded runs behind the paper's figures.
        "config": configuration.to_dict(),
    }
