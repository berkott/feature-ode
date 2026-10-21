import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """One-hidden-layer MLP with canonical μP parameterization.

    Forward:  f(x) = W2 · σ(σ_frozen · W1 · x)

    μP init and learning rates (set externally):
      - W1 ~ N(0, 1/d),   LR_W1 = η · N
      - W2 ~ N(0, 1/N^2), LR_W2 = η / N         (subtract_init=True)
      - W2 = 0,           LR_W2 = η / N         (subtract_init=False)

    These keep Δf = O(η) per step and give per-entry W1 evolution O(η) per
    step, so feature learning completes in O(1/(η√d)) steps independent of
    width N.

    Output-layer initialization modes:
    - subtract_init=True (standard μP with subtraction): W2 ~ N(0, 1/N^2)
      and the initial function f(x; θ_0) is subtracted (as a detached
      constant) so f(x; θ_0) = 0 for all x, matching the semi-frozen network's init.
      The W1 gradient is alive from step 0.
    - subtract_init=False (μP zero-init): W2 = 0 initially, no subtraction
      (f(x; θ_0) = 0 trivially). The W1 gradient is 0 at step 0, but W2
      grows by O(η/N) per step, so coherent W1 feature learning kicks in
      within O(1/η) steps — still width-independent.

    sigma_frozen (if set) is applied as a preactivation multiplier inside
    the forward, NOT folded into W1's init. This keeps W1 ~ N(0, 1/d) so
    the NFM diagonal starts at 1/d and sqrt(NFM_ii) = 1/sqrt(d) = a_0 at
    init, directly comparable to the semi-frozen network's A and DPR's B^(1) diagonals.
    """

    def __init__(self, d_in=1, width=4096, d_out=1, bias=False, nonlinearity=None,
                 hidden_std=1.0, sigma_frozen=None, subtract_init=False):
        super().__init__()
        self.d_in = d_in
        self.width = width
        self.d_out = d_out
        self.subtract_init = subtract_init
        self.sigma_frozen = sigma_frozen

        self.input_layer = nn.Linear(d_in, width, bias)
        self.output_layer = nn.Linear(width, d_out, bias)
        self.nonlinearity = nonlinearity if nonlinearity is not None else nn.ReLU()

        self._initialize(hidden_std)

    def _initialize(self, hidden_std):
        """Initialize weights and store initial copies as buffers if subtract_init."""
        # Input layer: N(0, hidden_std^2 / d_in)
        nn.init.normal_(self.input_layer.weight, std=hidden_std / math.sqrt(self.d_in))
        if self.input_layer.bias is not None:
            nn.init.zeros_(self.input_layer.bias)

        if self.subtract_init:
            # μP: W2 ~ N(0, 1/N^2) per entry, so |W2[n]| = O(1/N) and
            # f(x; θ_0) = Σ W2[n] σ_n has magnitude O(1/√N); subtracted below.
            nn.init.normal_(self.output_layer.weight, std=1.0 / self.width)
        else:
            nn.init.zeros_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

        # Store initial weight copies as buffers for f(θ_0) subtraction.
        # These are non-trainable and travel with the model through
        # stack_module_state / functional_call in the vmapped training loop.
        if self.subtract_init:
            self.register_buffer('w1_init', self.input_layer.weight.data.clone())
            self.register_buffer('w2_init', self.output_layer.weight.data.clone())
        else:
            self.register_buffer('w1_init', None)
            self.register_buffer('w2_init', None)

    def forward(self, x):
        pre = self.input_layer(x)
        if self.sigma_frozen is not None:
            pre = self.sigma_frozen * pre
        hidden = self.nonlinearity(pre)
        output = self.output_layer(hidden)
        if self.subtract_init and self.w1_init is not None:
            # Subtract f(x; θ_0) — detached via buffers (not in param graph)
            pre_init = F.linear(x, self.w1_init)
            if self.sigma_frozen is not None:
                pre_init = self.sigma_frozen * pre_init
            hidden_init = self.nonlinearity(pre_init)
            output_init = F.linear(hidden_init, self.w2_init)
            output = output - output_init
        return output
