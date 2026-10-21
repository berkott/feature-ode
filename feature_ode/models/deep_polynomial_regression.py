from math import factorial

import torch
import torch.nn as nn

class DPR(nn.Module):
    """
    Deep polynomial regression (DPR) model.

    Implements the function:
    f(x) = sum_{i in orders} (c_i / i!) B^{(i)} ((Ax)^⊗i)

    Where:
    - A is a trainable hidden_dim x input_dim matrix, initialized to a_0 · I
    - B^{(i)} are trainable polynomial coefficient tensors for order i, initialized to zero
    - c_i are fixed per-order coefficients (by default the activation's c_ell)

    Args:
        input_dim (int): Dimension of input features (d)
        model_configuration (dict, optional): Model settings with optional keys
            "orders" (list of polynomial orders, default 0 through 4),
            "hidden_dim" (default input_dim), "a_0" (default 0.1), and
            "c" (dict mapping order to coefficient, default 1 for every order).
    """

    def __init__(self, input_dim, model_configuration=None):
        super(DPR, self).__init__()

        model_configuration = model_configuration or {}
        orders = model_configuration.get("orders", range(5))

        self.input_dim = input_dim
        self.hidden_dim = model_configuration.get("hidden_dim", input_dim)
        self.a_0 = model_configuration.get("a_0", 0.1)
        raw_c = model_configuration.get("c", {o: 1.0 for o in (orders if orders is not None else range(5))})
        self.c = {o: v / factorial(o) if o > 0 else v for o, v in raw_c.items()}

        if orders is None:
            self.orders = list(range(5))
        else:
            self.orders = orders

        self.A  = nn.Parameter(torch.empty(self.hidden_dim, self.input_dim))
        self.Bs = nn.ParameterDict({
            str(o): nn.Parameter(torch.empty(() if o == 0 else (self.hidden_dim,) * o))
            for o in self.orders
        })
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.zeros_(self.A)
            m = min(self.hidden_dim, self.input_dim)
            nn.init.eye_(self.A[:m, :m])
            self.A[:m, :m].mul_(self.a_0)

            # Bs = 0
            for B in self.Bs.values():
                nn.init.zeros_(B)
    
    def forward(self, batch_x):
        """Forward pass: f(x) = sum_i c_i B^(i) (Ax)^{otimes i}.

        Uses no in-place ops and no explicit shape allocation so that
        the forward pass is compatible with torch.func.vmap.

        Args:
            batch_x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size,).
        """
        Ax = torch.einsum("kd, bd -> bk", self.A, batch_x)

        contributions = []
        for order in self.orders:
            if order == 0:
                contributions.append(self.Bs[str(order)] * self.c[order])
                continue

            operands = [self.Bs[str(order)], list(range(1, order + 1))]
            for i in range(order):
                operands += [Ax, [0, i + 1]]

            contributions.append(torch.einsum(*operands, [0]) * self.c[order])

        # Sum contributions — broadcasting handles scalar (order 0) + vector (order > 0)
        output = contributions[0]
        for contribution in contributions[1:]:
            output = output + contribution

        return output

    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    