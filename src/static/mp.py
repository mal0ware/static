"""The Marchenko-Pastur eigenvalue density of a pure-noise correlation matrix.

Notation follows Laloux, Cizeau, Bouchaud and Potters (1999), Eq. (3): ``N`` assets,
``T`` observations, ``Q = T / N >= 1`` and ``sigma2`` the variance of the matrix entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Edges of the density: ``sigma2 * (1 + 1/Q -+ 2 sqrt(1/Q))``. Laloux Eq. (3)."""
    if q < 1.0:
        raise ValueError("Q = T / N must be at least 1")
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be positive")
    root = np.sqrt(1.0 / q)
    return float(sigma2 * (1.0 - root) ** 2), float(sigma2 * (1.0 + root) ** 2)


def mp_pdf(lam: ArrayLike, q: float, sigma2: float = 1.0) -> FloatArray:
    """Density ``Q / (2 pi sigma2) * sqrt((l_max - l)(l - l_min)) / l``. Laloux Eq. (3).

    Zero outside ``[l_min, l_max]``.
    """
    lo, hi = mp_bounds(q, sigma2)
    x = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    out = np.zeros_like(x)
    inside = (x > lo) & (x < hi)
    xi = x[inside]
    out[inside] = q / (2.0 * np.pi * sigma2) * np.sqrt((hi - xi) * (xi - lo)) / xi
    return out


def mp_cdf(lam: ArrayLike, q: float, sigma2: float = 1.0, n_grid: int = 20001) -> FloatArray:
    """Cumulative distribution of the density, by quadrature.

    The substitution ``l = mid + half * sin(theta)`` removes the square-root behaviour
    at both edges, so a plain trapezoid rule on ``theta`` converges quickly.
    """
    lo, hi = mp_bounds(q, sigma2)
    mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_grid)
    grid = mid + half * np.sin(theta)
    # sqrt((hi - l)(l - lo)) = half * cos(theta) and dl = half * cos(theta) dtheta
    if lo == 0.0:
        # Q = 1: mid = half, and cos^2 / (1 + sin) = 1 - sin has no singularity
        integrand = q / (2.0 * np.pi * sigma2) * half * (1.0 - np.sin(theta))
    else:
        integrand = q / (2.0 * np.pi * sigma2) * (half * np.cos(theta)) ** 2 / grid
    steps = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(theta)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    cumulative /= cumulative[-1]
    x = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    return np.asarray(np.interp(x, grid, cumulative, left=0.0, right=1.0), dtype=np.float64)


def ks_distance(eigenvalues: ArrayLike, q: float, sigma2: float = 1.0) -> float:
    """Kolmogorov distance between an empirical spectrum and the density."""
    ev = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    n = ev.size
    if n == 0:
        raise ValueError("no eigenvalues")
    theory = mp_cdf(ev, q, sigma2)
    upper = np.arange(1, n + 1) / n - theory
    lower = theory - np.arange(0, n) / n
    return float(max(upper.max(), lower.max()))


@dataclass(frozen=True)
class NoiseFit:
    """Result of fitting the noise variance to the bulk of a spectrum."""

    sigma2: float
    lambda_min: float
    lambda_max: float
    n_signal: int
    iterations: int


def fit_noise_variance(eigenvalues: ArrayLike, q: float, max_iter: int = 1000) -> NoiseFit:
    """Self-consistent noise variance of the bulk.

    Start from ``sigma2 = 1``. Call every eigenvalue above the upper edge signal, set
    ``sigma2`` to the mean of the remaining eigenvalues, recompute the edge, and repeat
    until the split stops changing. With one signal eigenvalue and large ``N`` the first
    step is Laloux's ``sigma2 = 1 - lambda_1 / N``.

    Laloux et al. fitted ``sigma2`` by eye ("the best fit is obtained for 0.74") and do
    not give a procedure. This one is deterministic and has no tuning constant.
    """
    ev = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    n = ev.size
    if n < 2:
        raise ValueError("need at least two eigenvalues")
    sigma2 = 1.0
    n_signal = -1
    for iteration in range(1, max_iter + 1):
        _, hi = mp_bounds(q, sigma2)
        new_signal = int(np.sum(ev > hi))
        new_signal = min(new_signal, n - 1)
        if new_signal == n_signal:
            lo, hi = mp_bounds(q, sigma2)
            return NoiseFit(sigma2, lo, hi, n_signal, iteration)
        n_signal = new_signal
        sigma2 = float(ev[n_signal:].mean())
    raise RuntimeError("noise-variance fit did not converge")
