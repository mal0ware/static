"""Whether portfolio optimisation under maximal loss or expected shortfall has a solution.

Kondor, Pafka and Nagy (2007) for the linear programmes and the maximal-loss
probability; Ciliberti, Kondor and Mezard (2007) for the expected-shortfall boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, linprog
from scipy.stats import binom, norm

FloatArray = NDArray[np.float64]


def ml_feasible_probability(n: int, t: int) -> float:
    """Probability that the maximal-loss problem is bounded. Kondor et al. Eq. (26).

    ``p = 2^-(T-1) * sum_{k=N-1}^{T-1} C(T-1, k)``, the chance that a fair coin tossed
    ``T - 1`` times shows at least ``N - 1`` heads.
    """
    if n < 1 or t < 1:
        raise ValueError("n and t must be positive")
    return float(binom.sf(n - 2, t - 1, 0.5))


def ml_feasible_probability_normal(n: int, t: int) -> float:
    """Large-size approximation ``1 - Phi(2 (N/T - 1/2) sqrt(T))``. Kondor Eqs. (27)-(29)."""
    z = 2.0 * (n / t - 0.5) * np.sqrt(t)
    return float(1.0 - norm.cdf(z))


@dataclass(frozen=True)
class LpResult:
    """Outcome of a risk-minimising linear programme."""

    bounded: bool
    value: float
    weights: FloatArray | None


Bounds = list[tuple[float | None, float | None]]


def _solve(
    c: FloatArray, a_ub: FloatArray, b_ub: FloatArray, a_eq: FloatArray, bounds: Bounds, n: int
) -> LpResult:
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if res.status == 0:
        return LpResult(True, float(res.fun), np.asarray(res.x[:n], dtype=np.float64))
    if res.status in (3, 4):
        # The programme always has a feasible point (equal weights), so "unbounded" and
        # "infeasible or unbounded" both mean unbounded here.
        return LpResult(False, float("-inf"), None)
    raise RuntimeError(f"linear programme failed: {res.message}")


def minimise_maximal_loss(returns: FloatArray) -> LpResult:
    """Minimise the worst loss in the sample. Kondor et al. Eqs. (21)-(23).

    ``min u`` subject to ``u >= -sum_i w_i x_it`` for every ``t`` and ``sum_i w_i = 1``.
    ``returns`` is ``T x N``.
    """
    x = np.asarray(returns, dtype=np.float64)
    t, n = x.shape
    c = np.concatenate((np.zeros(n), [1.0]))
    a_ub = np.hstack((-x, -np.ones((t, 1))))
    a_eq = np.concatenate((np.ones(n), [0.0]))[None, :]
    bounds: Bounds = [(None, None)] * (n + 1)
    return _solve(c, a_ub, np.zeros(t), a_eq, bounds, n)


def minimise_expected_shortfall(returns: FloatArray, beta: float) -> LpResult:
    """Minimise expected shortfall at level ``beta``. Kondor et al. Eqs. (16)-(19).

    ``min v + sum_t u_t / ((1 - beta) T)`` subject to ``u_t >= -v - sum_i w_i x_it``,
    ``u_t >= 0`` and ``sum_i w_i = 1``. This is the Rockafellar-Uryasev programme.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    x = np.asarray(returns, dtype=np.float64)
    t, n = x.shape
    c = np.concatenate((np.zeros(n), [1.0], np.full(t, 1.0 / ((1.0 - beta) * t))))
    a_ub = np.hstack((-x, -np.ones((t, 1)), -np.eye(t)))
    a_eq = np.concatenate((np.ones(n), [0.0], np.zeros(t)))[None, :]
    free: Bounds = [(None, None)] * (n + 1)
    bounds = free + [(0.0, None)] * t
    return _solve(c, a_ub, np.zeros(t), a_eq, bounds, n)


def _saddle(v: float, s: float) -> tuple[float, float, float]:
    """``(beta, t, lhs24)`` implied by ``v`` and ``s = sqrt(q0)`` in Ciliberti et al.

    With ``X = v + s Z`` and ``Z`` standard normal, their integrals
    ``pi^-1/2 int ds e^{-s^2} f(v + s sqrt(2 q0))`` are ``E f(X)``. Eq. (21) then reads
    ``1 - beta + E[g'(X)] / 2 = 0`` and gives ``beta``. Eq. (22), after integrating by
    parts, reads ``t P(-1 < X < 0) = 1`` and gives ``t``. The third value is the left
    side of Eq. (24), ``t (1 - beta) v - q0 / 2 + (t / 2) E[g(X)]``, with ``g`` from
    Eq. (20).
    """
    a1, a0 = (-1.0 - v) / s, -v / s
    p1 = float(norm.cdf(a1))
    f1, f0 = float(norm.pdf(a1)), float(norm.pdf(a0))
    # P(-1 < X < 0), from the upper tail when both limits are positive to keep precision
    band = float(norm.sf(a1) - norm.sf(a0)) if a1 > 0.0 else float(norm.cdf(a0)) - p1
    mean_band = v * band + s * (f1 - f0)  # E[X; -1 <= X < 0]
    square_band = v * v * band + 2.0 * v * s * (f1 - f0) + s * s * (band + a1 * f1 - a0 * f0)
    tail = -(2.0 * v + 1.0) * p1 + 2.0 * s * f1  # E[-2X - 1; X < -1]
    beta = 1.0 + mean_band - p1
    t = 1.0 / band
    lhs24 = t * (1.0 - beta) * v - 0.5 * s * s + 0.5 * t * (square_band + tail)
    return beta, t, lhs24


_S_MIN, _S_MAX = 0.2, 200.0


def _boundary_point(s: float) -> tuple[float, float]:
    """``(beta, N/T)`` on the phase boundary for a given ``s = sqrt(q0)``."""
    v = float(brentq(lambda a: _saddle(a, s)[2], -(5.0 * s + 1.0), 2.0 * s, xtol=1e-14))
    beta, t, _ = _saddle(v, s)
    return beta, 1.0 / t


def es_critical_ratio(beta: float) -> float:
    """Critical ``N/T`` above which expected-shortfall optimisation has no solution.

    Solves Ciliberti, Kondor and Mezard (2007) Eqs. (21), (22) and (24) with equality,
    for Gaussian returns in the limit of large ``N`` and ``T``. Tends to 1/2 as ``beta``
    tends to 1 (their Eq. (25)); above ``beta`` of about 0.93 the value returned is within
    1e-7 of 1/2 and is not resolved further.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    beta_hi, ratio_hi = _boundary_point(_S_MIN)
    if beta >= beta_hi:
        return ratio_hi
    beta_lo, _ = _boundary_point(_S_MAX)
    if beta <= beta_lo:
        raise ValueError(f"beta below {beta_lo:.4g} is not supported")
    log_s = brentq(
        lambda u: _boundary_point(float(np.exp(u)))[0] - beta,
        np.log(_S_MIN),
        np.log(_S_MAX),
        xtol=1e-12,
    )
    return _boundary_point(float(np.exp(log_s)))[1]
