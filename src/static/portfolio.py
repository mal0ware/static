"""Minimum-variance portfolios and the cost of estimating them from finite samples.

Follows Pafka and Kondor (2002), "Noisy covariance matrices and portfolio optimization
II", and Kondor, Pafka and Nagy (2007).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, stats

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


def _f_law(n: int, t: int) -> tuple[float, int, int]:
    if n < 2 or t < n:
        raise ValueError("need N >= 2 and T >= N")
    return (n - 1) / (t - n + 2), n - 1, t - n + 2


def q0_exact_rms(n: int, t: int) -> float:
    """Root mean square of ``q0`` at finite ``N`` and ``T``: ``sqrt((T - 1) / (T - N))``.

    For Gaussian returns with a known zero mean the sample covariance is Wishart with
    ``T`` degrees of freedom, and for any true covariance

    ``q0^2 = 1 + (N - 1) / (T - N + 2) * F``, with ``F ~ F(N - 1, T - N + 2)``.

    This is a property of the Wishart distribution and is not taken from the sources;
    the tests check it against simulation. The mean of ``q0^2`` is finite for
    ``T > N`` and its variance is finite only for ``T > N + 2``.
    """
    if t <= n:
        raise ValueError("the mean of q0^2 is infinite unless T > N")
    return float(np.sqrt((t - 1) / (t - n)))


def q0_exact_quantile(n: int, t: int, prob: float = 0.5) -> float:
    """Quantile of ``q0`` at finite ``N`` and ``T``. See :func:`q0_exact_rms`."""
    scale, d1, d2 = _f_law(n, t)
    return float(np.sqrt(1.0 + scale * stats.f(d1, d2).ppf(prob)))


def q0_exact_mean(n: int, t: int) -> float:
    """Mean of ``q0`` at finite ``N`` and ``T``, by quadrature. See :func:`q0_exact_rms`."""
    scale, d1, d2 = _f_law(n, t)
    law = stats.f(d1, d2)

    def integrand(u: float) -> float:
        # substitute F = u / (1 - u) so the heavy right tail maps to a finite interval
        x = u / (1.0 - u)
        return float(np.sqrt(1.0 + scale * x) * law.pdf(x) / (1.0 - u) ** 2)

    value, _ = integrate.quad(integrand, 0.0, 1.0, limit=400)
    return float(value)


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


def paired_q0(
    n: int,
    t: int,
    rng: np.random.Generator,
    estimators: Mapping[str, Callable[[FloatArray], FloatArray]],
    true_cov: FloatArray | None = None,
) -> dict[str, float]:
    """``q0`` of several covariance estimators on one and the same sample.

    Each estimator maps the ``T x N`` return sample to a covariance matrix. Drawing the
    sample once makes the difference between two estimators a paired comparison.
    """
    sigma0 = np.eye(n) if true_cov is None else np.asarray(true_cov, dtype=np.float64)
    z = rng.standard_normal((t, n))
    sample = z if true_cov is None else z @ np.linalg.cholesky(sigma0).T
    base = portfolio_variance(min_variance_weights(sigma0), sigma0)
    out = {}
    for name, estimator in estimators.items():
        w = min_variance_weights(estimator(sample))
        out[name] = float(np.sqrt(portfolio_variance(w, sigma0) / base))
    return out
