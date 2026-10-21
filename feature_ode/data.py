"""Data for the experiments: anisotropic Gaussian inputs and Hermite or monomial targets.

Generates Gaussian data with power-law decaying covariance and labels
using either Hermite polynomial or standard monomial target functions.
"""

import math
import re
from typing import List, Tuple

import torch


# === Hermite polynomials ===

def get_hermite_polynomials():
    """Return probabilist's Hermite polynomials He_n(x) up to degree 20."""
    return {
        1: lambda x: x,
        2: lambda x: x**2 - 1,
        3: lambda x: x**3 - 3*x,
        4: lambda x: x**4 - 6*x**2 + 3,
        5: lambda x: x**5 - 10*x**3 + 15*x,
        6: lambda x: x**6 - 15*x**4 + 45*x**2 - 15,
        7: lambda x: x**7 - 21*x**5 + 105*x**3 - 105*x,
        8: lambda x: x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105,
        9: lambda x: x**9 - 36*x**7 + 378*x**5 - 1260*x**3 + 945*x,
        10: lambda x: x**10 - 45*x**8 + 630*x**6 - 3150*x**4 + 4725*x**2 - 945,
        11: lambda x: x**11 - 55*x**9 + 990*x**7 - 6930*x**5 + 17325*x**3 - 10395*x,
        12: lambda x: x**12 - 66*x**10 + 1485*x**8 - 13860*x**6 + 51975*x**4 - 62370*x**2 + 10395,
        13: lambda x: x**13 - 78*x**11 + 2145*x**9 - 25740*x**7 + 135135*x**5 - 270270*x**3 + 135135*x,
        14: lambda x: x**14 - 91*x**12 + 3003*x**10 - 45045*x**8 + 315315*x**6 - 945945*x**4 + 945945*x**2 - 135135,
        15: lambda x: x**15 - 105*x**13 + 4095*x**11 - 76545*x**9 + 675675*x**7 - 2702700*x**5 + 4729725*x**3 - 2027025*x,
        16: lambda x: x**16 - 120*x**14 + 5460*x**12 - 120120*x**10 + 1351350*x**8 - 7567560*x**6 + 18918900*x**4 - 20270250*x**2 + 34459425,
        17: lambda x: x**17 - 136*x**15 + 7140*x**13 - 180180*x**11 + 2297295*x**9 - 15315300*x**7 + 51081030*x**5 - 87513450*x**3 + 34459425*x,
        18: lambda x: x**18 - 153*x**16 + 9180*x**14 - 257400*x**12 + 3783780*x**10 - 30630600*x**8 + 122522400*x**6 - 229729500*x**4 + 172972500*x**2 - 34459425,
        19: lambda x: x**19 - 171*x**17 + 11628*x**15 - 375375*x**13 + 6432420*x**11 - 61261200*x**9 + 306306000*x**7 - 765765000*x**5 + 875134500*x**3 - 310134825*x,
        20: lambda x: x**20 - 190*x**18 + 14250*x**16 - 513513*x**14 + 10210200*x**12 - 117117000*x**10 + 765765000*x**8 - 2677114440*x**6 + 4670678100*x**4 - 3101348250*x**2 + 654729075,
    }


# === Target function parsing ===

def parse_target_string(target_string: str) -> List[Tuple[float, List[Tuple[int, int]]]]:
    """Parse a target function string into a list of (coefficient, factors) terms.

    Each term has an optional leading sign and an optional numeric coefficient
    (decimal or scientific notation), followed by one or more factors of the
    form x_{i} or x_{i}^p. Missing coefficient defaults to 1; a leading "-"
    flips the sign.

    Examples:
        "x_{0} x_{1} x_{2} + x_{5}"     → [(1.0, [(0,1),(1,1),(2,1)]), (1.0, [(5,1)])]
        "x_{0} + 0.1 x_{0}^2"           → [(1.0, [(0,1)]), (0.1, [(0,2)])]
        "2 x_{0} x_{1} - 3 x_{2}^2"     → [(2.0, [(0,1),(1,1)]), (-3.0, [(2,2)])]
        "1.5e-2 x_{0}"                  → [(0.015, [(0,1)])]

    Returns:
        List of (coefficient, factors) tuples, where factors is a list of
        (variable_index, power) tuples.
    """
    factor_pattern = re.compile(r"x_\{?(\d+)\}?(?:\^\{?(\d+)\}?)?")
    term_pattern = re.compile(
        r"([+-]?)\s*"  # optional sign
        r"((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?\s*\*?\s*"  # optional coefficient (and optional *)
        r"((?:x_\{?\d+\}?(?:\^\{?\d+\}?)?\s*)+)"  # one or more x factors
    )

    monomials = []
    cursor = 0
    for match in term_pattern.finditer(target_string):
        between = target_string[cursor:match.start()]
        if between.strip():
            raise ValueError(
                f"Unparseable text in target string: '{between}' (full target: '{target_string}')"
            )

        sign_text = match.group(1)
        coefficient_text = match.group(2)
        factors_text = match.group(3)

        magnitude = float(coefficient_text) if coefficient_text else 1.0
        coefficient = -magnitude if sign_text == "-" else magnitude

        factors = []
        for factor_match in factor_pattern.finditer(factors_text):
            variable_index = int(factor_match.group(1))
            power = int(factor_match.group(2)) if factor_match.group(2) else 1
            factors.append((variable_index, power))

        if not factors:
            raise ValueError(f"No factors found in term: '{match.group(0)}'")

        monomials.append((coefficient, factors))
        cursor = match.end()

    trailing = target_string[cursor:]
    if trailing.strip():
        raise ValueError(
            f"Trailing unparseable text: '{trailing}' (full target: '{target_string}')"
        )
    if not monomials:
        raise ValueError(f"No terms parsed from target: '{target_string}'")

    return monomials


def evaluate_target(
    x: torch.Tensor,
    monomials: List[Tuple[float, List[Tuple[int, int]]]],
    target_hermite: bool,
    gammas: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the target function on standardized variables z_i = x_i / sqrt(gamma_i).

    Args:
        x: Input data of shape (batch_size, d).
        monomials: Parsed target structure from parse_target_string —
            list of (coefficient, factors).
        target_hermite: If True, x_{i}^k means He_k(z_i). If False, means z_i^k.
        gammas: Covariance eigenvalues of shape (d,).

    Returns:
        y of shape (batch_size,).
    """
    z = x / gammas.sqrt().unsqueeze(0)
    hermite_polynomials = get_hermite_polynomials() if target_hermite else None
    y = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    for coefficient, monomial in monomials:
        term = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        for variable_index, power in monomial:
            if target_hermite:
                # Normalized Hermite: He_k(z) / sqrt(k!) so that E[term^2] = 1
                term = term * hermite_polynomials[power](z[:, variable_index]) / math.sqrt(math.factorial(power))
            else:
                term = term * z[:, variable_index] ** power
        y = y + coefficient * term

    return y


# === Data generation ===

def generate_covariance_eigenvalues(d: int, powerlaw_alpha: float) -> torch.Tensor:
    """Generate normalized power-law eigenvalues for the covariance matrix.

    gamma_i = i^{-alpha} for i = 1, ..., d, then divided by sum so that
    sum(gamma) = 1, meaning E[||x||^2] = 1.

    Args:
        d: Number of dimensions.
        powerlaw_alpha: Exponent for power-law decay.

    Returns:
        Tensor of shape (d,) with normalized eigenvalues.
    """
    indices = torch.arange(1, d + 1, dtype=torch.float64)
    gammas = indices ** (-powerlaw_alpha)
    gammas = gammas / gammas.sum()
    return gammas
