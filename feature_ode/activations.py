"""Activation functions and their polynomial expansion coefficients c_ell."""

import math

import torch
import torch.nn.functional as F


ACTIVATION_OPTIONS = ("cos", "shifted_silu", "shifted_tanh")

# === Polynomial expansion coefficients c_ell for each activation function ===
# c_ell[ell] is the coefficient for order ell in the polynomial kernel expansion.
# For cos: c_ell = 1 for all orders.
# For shifted_silu (silu(x + 0.5)) and shifted_tanh (tanh(x + 0.5)): numerically computed coefficients.
ACTIVATION_C_ELL = {
    "cos": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "shifted_silu": (
        9.686390e-2,
        5.475426e-1,
        1.946831e-1,
        4.877418e-2,
        1.109193e-1,
        3.498909e-1,
        3.824663e-1,
        8.236804e+0,
        1.320299e+0,
        4.413091e+2,
        8.978149e+1,
    ),
    "shifted_tanh": (
        2.135523e-1,
        6.185000e-1,
        5.283283e-1,
        3.194615e-1,
        1.562004e+1,
        1.067124e+1,
        1.316205e+3,
        2.025604e+4,
        1.086156e+5,
        2.053715e+7,
        1.145446e+7,
    ),
}


def build_nonlinearity(activation: str):
    """Return the nonlinearity function for the given activation name."""
    if activation == "cos":
        return lambda z: math.sqrt(2) * torch.cos(z + math.pi / 4)
    elif activation == "shifted_silu":
        return lambda z: F.silu(z + 0.5)
    elif activation == "shifted_tanh":
        return lambda z: torch.tanh(z + 0.5)
    else:
        raise ValueError(f"Unknown activation: {activation}")
