"""Feasibility of maximal-loss and expected-shortfall optimisation."""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from static.feasibility import (
    _saddle,
    es_critical_ratio,
    minimise_expected_shortfall,
    minimise_maximal_loss,
    ml_feasible_probability,
    ml_feasible_probability_normal,
)


def test_two_assets_two_days_is_a_coin_toss() -> None:
    # Kondor, Pafka and Nagy work this case out by hand: probability 1/2
    assert ml_feasible_probability(2, 2) == pytest.approx(0.5)


@pytest.mark.parametrize(("n", "t"), [(2, 2), (3, 7), (10, 25), (40, 80), (5, 4)])
def test_probability_matches_the_binomial_sum(n: int, t: int) -> None:
    direct = sum(comb(t - 1, k) for k in range(n - 1, t)) / 2 ** (t - 1)
    assert ml_feasible_probability(n, t) == pytest.approx(direct)


def test_probability_limits() -> None:
    assert ml_feasible_probability(1, 10) == 1.0  # one asset: nothing to optimise
    assert ml_feasible_probability(11, 10) == 0.0  # fewer days than assets
    assert ml_feasible_probability(10, 10) == pytest.approx(2.0**-9)


def test_probability_is_half_on_the_critical_line() -> None:
    # with T - 1 = 2 (N - 1) - 1 tosses, "at least N - 1 heads" is exactly half the outcomes
    for n in (2, 5, 50):
        assert ml_feasible_probability(n, 2 * n - 2) == pytest.approx(0.5)


def test_probability_sharpens_into_a_step_at_one_half() -> None:
    assert ml_feasible_probability(400, 1000) > 0.999  # N/T = 0.4
    assert ml_feasible_probability(600, 1000) < 0.001  # N/T = 0.6


def test_normal_approximation_is_close_for_large_sizes() -> None:
    # Eq. (27) drops a shift of about 1 / sqrt(T) in z, so it sits a little below the
    # exact sum: 0.500 against 0.556 at N = 100, T = 200
    for n, t in [(100, 200), (95, 200), (110, 200), (480, 1000)]:
        exact = ml_feasible_probability(n, t)
        approx = ml_feasible_probability_normal(n, t)
        assert approx == pytest.approx(exact, abs=0.06)
        assert approx <= exact
    assert ml_feasible_probability_normal(4000, 8000) == pytest.approx(
        ml_feasible_probability(4000, 8000), abs=0.01
    )


def test_maximal_loss_lp_on_a_solvable_case() -> None:
    # two assets, two days, slopes of opposite sign: a solution exists
    x = np.array([[1.0, -1.0], [-1.0, 1.0]])
    res = minimise_maximal_loss(x)
    assert res.bounded
    assert res.weights is not None
    assert np.allclose(res.weights, [0.5, 0.5])
    assert res.value == pytest.approx(0.0, abs=1e-9)


def test_maximal_loss_lp_on_an_unbounded_case() -> None:
    # asset 0 beats asset 1 on both days: going long 0 and short 1 without limit
    x = np.array([[1.0, 0.0], [2.0, 1.0]])
    assert not minimise_maximal_loss(x).bounded


def test_maximal_loss_lp_value_is_the_worst_loss_of_the_weights() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal((30, 4))
        res = minimise_maximal_loss(x)
        if res.bounded:
            assert res.weights is not None
            assert res.weights.sum() == pytest.approx(1.0)
            assert res.value == pytest.approx(np.max(-x @ res.weights), abs=1e-8)
            assert res.value <= np.max(-x @ np.full(4, 0.25)) + 1e-9


def test_simulated_feasibility_matches_the_closed_form() -> None:
    rng = np.random.default_rng(1)
    n, reps = 8, 400
    for t in (12, 15, 20):
        hits = sum(minimise_maximal_loss(rng.standard_normal((t, n))).bounded for _ in range(reps))
        p = ml_feasible_probability(n, t)
        sd = np.sqrt(p * (1.0 - p) / reps)
        assert abs(hits / reps - p) < 4.0 * sd


def test_expected_shortfall_equals_maximal_loss_when_only_the_worst_day_counts() -> None:
    rng = np.random.default_rng(2)
    t = 20
    beta = 1.0 - 0.5 / t  # (1 - beta) T <= 1
    seen = 0
    for _ in range(30):
        x = rng.standard_normal((t, 4))
        ml = minimise_maximal_loss(x)
        es = minimise_expected_shortfall(x, beta)
        assert ml.bounded == es.bounded
        if ml.bounded:
            seen += 1
            assert es.value == pytest.approx(ml.value, abs=1e-7)
    assert seen > 5


def test_expected_shortfall_lp_value_is_the_tail_mean() -> None:
    rng = np.random.default_rng(3)
    t, beta = 100, 0.9
    x = rng.standard_normal((t, 3))
    res = minimise_expected_shortfall(x, beta)
    assert res.bounded
    assert res.weights is not None
    losses = np.sort(-x @ res.weights)[::-1]
    assert res.value == pytest.approx(losses[:10].mean(), abs=1e-7)


def test_saddle_closed_forms_match_quadrature() -> None:
    def g(x: float) -> float:
        if x >= 0.0:
            return 0.0
        return x * x if x >= -1.0 else -2.0 * x - 1.0

    def g_prime(x: float) -> float:
        if x >= 0.0:
            return 0.0
        return 2.0 * x if x >= -1.0 else -2.0

    def expect(f: Callable[[float, float], float], v: float, s: float) -> float:
        breaks = [-v / s, (-1.0 - v) / s]
        return float(quad(lambda z: f(v + s * z, z) * norm.pdf(z), -12, 12, points=breaks)[0])

    for v, s in [(-0.3, 0.7), (0.5, 2.0), (-2.0, 1.5)]:
        beta, t, lhs = _saddle(v, s)
        # Eq. (21): 1 - beta + E[g'] / 2 = 0
        assert 1.0 - beta + 0.5 * expect(lambda x, z: g_prime(x), v, s) == pytest.approx(
            0.0, abs=1e-9
        )
        # Eq. (22) in its printed form: -1 + t / (2 sqrt(q0)) E[Z g'(X)] = 0
        stein = expect(lambda x, z: z * g_prime(x), v, s)
        assert -1.0 + t / (2.0 * s) * stein == pytest.approx(0.0, abs=1e-8)
        # Eq. (24) left side
        direct = t * (1.0 - beta) * v - 0.5 * s * s + 0.5 * t * expect(lambda x, z: g(x), v, s)
        assert lhs == pytest.approx(direct, abs=1e-8)


def test_es_boundary_tends_to_one_half() -> None:
    # Ciliberti et al. Eq. (25): 1/t* -> 1/2 as beta -> 1, the maximal-loss value
    assert es_critical_ratio(0.99) == pytest.approx(0.5, abs=1e-6)
    assert es_critical_ratio(0.95) == pytest.approx(0.5, abs=1e-6)


def test_es_boundary_is_increasing_and_below_one_half() -> None:
    betas = np.linspace(0.05, 0.99, 40)
    ratios = np.array([es_critical_ratio(b) for b in betas])
    assert np.all(np.diff(ratios) >= -1e-12)
    assert np.all(ratios > 0.0)
    assert np.all(ratios <= 0.5)


def test_es_boundary_goes_to_the_origin() -> None:
    assert es_critical_ratio(0.02) < 0.01


def test_simulated_es_feasibility_straddles_the_boundary() -> None:
    rng = np.random.default_rng(4)
    beta, n = 0.7, 40
    critical = es_critical_ratio(beta)

    def frequency(ratio: float) -> float:
        t = round(n / ratio)
        return float(
            np.mean(
                [
                    minimise_expected_shortfall(rng.standard_normal((t, n)), beta).bounded
                    for _ in range(30)
                ]
            )
        )

    assert frequency(critical - 0.12) > 0.85
    assert frequency(critical + 0.12) < 0.15
