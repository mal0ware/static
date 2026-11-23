"""Minimum-variance portfolios and the cost of estimating them from finite samples.

Follows Pafka and Kondor (2002), "Noisy covariance matrices and portfolio optimization
II", and Kondor, Pafka and Nagy (2007).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from static.cleaning import sample_covariance
from static.mp import mp_bounds

FloatArray = NDArray[np.float64]
Cleaner = Callable[[FloatArray, float], FloatArray]


def min_variance_weights(cov: FloatArray) -> FloatArray:
    """Fully invested minimum-variance weights. Pafka and Kondor Eq. (1).

    ``w_i = sum_j inv(cov)_ij / sum_jk inv(cov)_jk``. Short positions are allowed.
    """
    ones = np.ones(cov.shape[0])
    raw = np.linalg.solve(cov, ones)
    return np.asarray(raw / raw.sum(), dtype=np.float64)


def portfolio_variance(weights: FloatArray, cov: FloatArray) -> float:
    """``w' cov w``."""
    return float(weights @ cov @ weights)


def q0_theory(ratio: float) -> float:
    """True risk of the estimated portfolio over the true optimum: ``1 / sqrt(1 - N/T)``.

    Kondor, Pafka and Nagy (2007) Eq. (7); Pafka and Kondor (2002), text after Eq. (4).
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("N/T must be in [0, 1)")
    return float(1.0 / np.sqrt(1.0 - ratio))


def q1_theory(ratio: float) -> float:
    """In-sample (predicted) risk over the true optimum: ``sqrt(1 - N/T)``.

    Pafka and Kondor (2002), text after Eq. (4).
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("N/T must be in [0, 1)")
    return float(np.sqrt(1.0 - ratio))


def q0_from_density(ratio: float, n_grid: int = 200001) -> float:
    """``q0`` from the eigenvalue density, Pafka and Kondor (2002) Eq. (4), by quadrature.

    ``q0 = sqrt(int rho/l^2 dl) / int rho/l dl`` with ``rho`` their Eq. (3). Used to
    check the closed form.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError("N/T must be in (0, 1)")
    lo, hi = mp_bounds(1.0 / ratio)
    mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_grid)
    lam = mid + half * np.sin(theta)
    weight = (half * np.cos(theta)) ** 2 / (2.0 * np.pi * ratio * lam)  # rho(l) dl/dtheta
    d_theta = float(theta[1] - theta[0])

    def integrate(values: FloatArray) -> float:
        return float((values[1:] + values[:-1]).sum() * 0.5 * d_theta)

    return float(np.sqrt(integrate(weight / lam**2)) / integrate(weight / lam))


def market_mode_covariance(n: int, ratio: float = 25.0) -> FloatArray:
    """Pafka and Kondor's Model II: identity plus one large eigenvalue along (1, ..., 1).

    The large eigenvalue is ``ratio`` times the others.
    """
    return np.asarray(np.eye(n) + (ratio - 1.0) * np.ones((n, n)) / n, dtype=np.float64)


@dataclass(frozen=True)
class NoiseRatios:
    """Risk of the estimated minimum-variance portfolio in units of the true optimum.

    ``q0`` true risk, ``q1`` predicted (in-sample) risk, ``q2`` realised risk in a second
    independent sample of the same length. Pafka and Kondor (2002), section 2.
    """

    q0: float
    q1: float
    q2: float


def simulate_noise_ratios(
    n: int,
    t: int,
    rng: np.random.Generator,
    true_cov: FloatArray | None = None,
    cleaner: Cleaner | None = None,
) -> NoiseRatios:
    """One draw of ``(q0, q1, q2)`` for ``n`` assets and ``t`` observations.

    Returns are Gaussian with covariance ``true_cov`` (identity when ``None``). The
    covariance is estimated without subtracting the mean, as in Pafka and Kondor Eq. (2).
    ``cleaner(cov, t / n)`` is applied to the estimate before optimising when given;
    ``q1`` is then measured with the cleaned matrix.
    """
    if true_cov is None:
        sigma0 = np.eye(n)
        chol = None
    else:
        sigma0 = np.asarray(true_cov, dtype=np.float64)
        chol = np.linalg.cholesky(sigma0)

    def draw() -> FloatArray:
        z = rng.standard_normal((t, n))
        return z if chol is None else z @ chol.T

    sigma1 = sample_covariance(draw(), demean=False)
    sigma2 = sample_covariance(draw(), demean=False)
    used = sigma1 if cleaner is None else cleaner(sigma1, t / n)
    w = min_variance_weights(used)
    w_true = min_variance_weights(sigma0)
    base = portfolio_variance(w_true, sigma0)
    return NoiseRatios(
        q0=float(np.sqrt(portfolio_variance(w, sigma0) / base)),
        q1=float(np.sqrt(portfolio_variance(w, used) / base)),
        q2=float(np.sqrt(portfolio_variance(w, sigma2) / base)),
    )
