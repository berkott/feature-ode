import numpy as np
import pytest

from feature_ode.ode import FeatureODE


def loss_at(ode, a, b):
    saved = ode.a, ode.b
    ode.a, ode.b = a, b
    loss = ode.compute_loss()
    ode.a, ode.b = saved
    return loss


def test_derivatives_are_the_negative_gradient_of_the_loss():
    ode = FeatureODE(d=3, target="x_{0} + x_{0} x_{1} + 0.5 x_{2}^2", gammas=[0.5, 0.3, 0.2], a_0=0.6, c_ell=(1.0, 0.8, 0.6))
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0.3, 1.5, size=3), rng.uniform(-1.0, 1.0, size=ode.num_terms)
    da, db = ode._compute_derivatives_from_state(a, b)
    epsilon = 1e-6
    for i in range(3):
        shift = np.zeros(3)
        shift[i] = epsilon
        gradient = (loss_at(ode, a + shift, b) - loss_at(ode, a - shift, b)) / (2 * epsilon)
        assert da[i] == pytest.approx(-gradient, rel=1e-6, abs=1e-9)
    for j in range(ode.num_terms):
        shift = np.zeros(ode.num_terms)
        shift[j] = epsilon
        gradient = (loss_at(ode, a, b + shift) - loss_at(ode, a, b - shift)) / (2 * epsilon)
        assert db[j] == pytest.approx(-gradient, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("target, powers", [("x_{0}", {0: 1}), ("x_{0}^2", {0: 2}), ("x_{0} x_{1}", {0: 1, 1: 1})])
def test_single_term_conserves_a_squared_minus_alpha_b_squared(target, powers):
    # Gradient flow on a single term conserves a_i^2 - alpha_i b^2, the quantity behind the paper's learning-time integral.
    ode = FeatureODE(d=2, target=target, gammas=[0.6, 0.4], a_0=0.7)
    conserved_at_start = {i: ode.a[i] ** 2 - alpha * ode.b[0] ** 2 for i, alpha in powers.items()}
    for _ in range(20000):
        ode.step(0.01)
    assert ode.compute_loss() < 1e-6
    for i, alpha in powers.items():
        assert ode.a[i] ** 2 - alpha * ode.b[0] ** 2 == pytest.approx(conserved_at_start[i], abs=1e-6)
