import math

import torch
import torch.nn as nn


class SemiFrozenNetwork(nn.Module):
    """Semi-frozen network (SFN): the intermediate model between the MLP and DPR in the paper's appendix.

    Forward:  f(x) = W_out · σ(W_frozen · A · x)

    A is a trainable d x d matrix initialized to a_0 · I, W_frozen is a fixed
    Gaussian N(0, hidden_std^2) matrix of shape (width, d), and W_out is a
    trainable readout initialized to zero. Freezing W_frozen removes the
    per-neuron feature learning of an MLP while keeping its elementwise
    nonlinearity.
    """

    def __init__(self, d_in=1, width=4096, d_out=1, bias=False, nonlinearity=None, a_0=0.1, hidden_std=1.0):
        super().__init__()
        self.d_in, self.width, self.d_out = d_in, width, d_out

        self.input_layer = nn.Linear(d_in, d_in, bias)
        self.hidden_layer = nn.Linear(d_in, width, bias)
        self.hidden_layer.weight.requires_grad_(False)
        if self.hidden_layer.bias is not None:
            self.hidden_layer.bias.requires_grad_(False)
        self.output_layer = nn.Linear(width, d_out, bias)
        self.nonlinearity = nonlinearity if nonlinearity is not None else (
            lambda z: math.sqrt(2) * torch.cos(z + math.pi / 4)
        )

        self._initialize(a_0, hidden_std)

    def _initialize(self, a_0, hidden_std):
        """Initialize: input = a_0 * I, hidden = N(0, hidden_std^2), output = 0."""
        with torch.no_grad():
            # Input layer: a_0 * I
            nn.init.eye_(self.input_layer.weight)
            self.input_layer.weight.mul_(a_0)

            # Hidden layer: Gaussian N(0, hidden_std^2)
            nn.init.normal_(self.hidden_layer.weight, std=hidden_std)

            # Output layer: zeros
            nn.init.zeros_(self.output_layer.weight)
            if self.output_layer.bias is not None:
                nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        h = self.input_layer(x)
        h = self.nonlinearity(self.hidden_layer(h))
        out = self.output_layer(h)
        return out
