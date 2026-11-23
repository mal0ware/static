"""The absorption ratio of Kritzman, Li, Page and Rigobon (2011)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def exponential_weights(window: int, half_life: float | None) -> FloatArray:
    """Weights for a window of observations, oldest first, summing to one.

    The weight halves every ``half_life`` observations going back in time.
    ``half_life=None`` gives equal weights.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if half_life is None:
        return np.full(window, 1.0 / window)
    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    age = np.arange(window - 1, -1, -1, dtype=np.float64)
    weights = 0.5 ** (age / half_life)
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def weighted_covariance(returns: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted covariance of a ``T x N`` return matrix about its weighted mean."""
    x = np.asarray(returns, dtype=np.float64)
    if x.shape[0] != weights.size:
        raise ValueError("one weight per observation is required")
    centred = x - weights @ x
    return np.asarray((centred * weights[:, None]).T @ centred, dtype=np.float64)


def absorption_ratio(cov: FloatArray, n_components: int) -> float:
    """Share of total variance absorbed by the top eigenvectors. Kritzman et al. Eq. (1).

    ``AR = sum_{i<=n} var(E_i) / sum_{j<=N} var(A_j)``. The variance of the i-th
    eigenportfolio is the i-th eigenvalue and the asset variances sum to the trace.
    """
    n_assets = cov.shape[0]
    if not 1 <= n_components <= n_assets:
        raise ValueError("n_components must be between 1 and the number of assets")
    values = np.linalg.eigvalsh(cov)[::-1]
    return float(values[:n_components].sum() / np.trace(cov))


def default_components(n_assets: int) -> int:
    """About one fifth of the assets, as in Kritzman et al. (51 industries, 10 vectors)."""
    return max(1, round(n_assets / 5))


def rolling_absorption_ratio(
    returns: FloatArray,
    window: int = 500,
    half_life: float | None = 250.0,
    n_components: int | None = None,
) -> FloatArray:
    """Absorption ratio on trailing windows.

    Defaults are the paper's: 500 days, exponential weights with a 250-day half-life, and
    one fifth of the assets. Entry ``t`` uses rows ``t - window + 1 .. t``; the first
    ``window - 1`` entries are NaN.
    """
    x = np.asarray(returns, dtype=np.float64)
    t_total, n_assets = x.shape
    k = default_components(n_assets) if n_components is None else n_components
    weights = exponential_weights(window, half_life)
    out = np.full(t_total, np.nan)
    for end in range(window, t_total + 1):
        cov = weighted_covariance(x[end - window : end], weights)
        out[end - 1] = absorption_ratio(cov, k)
    return out


def _rolling(values: FloatArray, window: int, use_std: bool) -> FloatArray:
    out = np.full(values.size, np.nan)
    if values.size < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, window)
    stat = view.std(axis=1, ddof=1) if use_std else view.mean(axis=1)
    out[window - 1 :] = stat
    return out


def standardized_shift(ar: FloatArray, short: int = 15, long: int = 252) -> FloatArray:
    """Standardised shift of the absorption ratio. Kritzman et al. Eq. (2).

    ``(AR_15day - AR_1year) / sigma`` with 15-day and one-year moving averages and
    ``sigma`` the standard deviation of the ratio over the same year. The paper does not
    give the number of days in a year; 252 is used. Zero ``sigma`` gives NaN.
    """
    values = np.asarray(ar, dtype=np.float64)
    short_ma = _rolling(values, short, use_std=False)
    long_ma = _rolling(values, long, use_std=False)
    long_sd = _rolling(values, long, use_std=True)
    flat = ~(long_sd > 1e-12)
    shift = (short_ma - long_ma) / np.where(flat, 1.0, long_sd)
    shift[flat] = np.nan
    return np.asarray(shift, dtype=np.float64)
