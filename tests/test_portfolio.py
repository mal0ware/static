"""Minimum-variance weights and the divergence of out-of-sample risk."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from static.cleaning import clip_covariance, sample_covariance
from static.portfolio import (
    market_mode_covariance,
    min_variance_weights,
    paired_q0,
    portfolio_variance,
    q0_exact_mean,
    q0_exact_quantile,
    q0_exact_rms,
    q0_from_density,
    q0_theory,
    q1_theory,
    simulate_noise_ratios,
)


def test_weights_sum_to_one_and_equal_for_identity() -> None:
    w = min_variance_weights(np.eye(7))
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 1.0 / 7.0)


def test_weights_inverse_variance_for_diagonal() -> None:
    variances = np.array([1.0, 2.0, 4.0])
    w = min_variance_weights(np.diag(variances))
    assert np.allclose(w, (1.0 / variances) / (1.0 / variances).sum())


def test_two_asset_closed_form() -> None:
    s1, s2, rho = 0.2, 0.3, 0.4
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    assert np.allclose(min_variance_weights(cov), [w1, 1.0 - w1])


def test_minimum_variance_is_one_over_sum_of_inverse() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 6))
    cov = a.T @ a / 30
    w = min_variance_weights(cov)
    assert portfolio_variance(w, cov) == pytest.approx(1.0 / np.linalg.inv(cov).sum())
    # no other fully invested portfolio does better
    for _ in range(20):
        other = w + 0.1 * rng.standard_normal(6)
        other /= other.sum()
        assert portfolio_variance(other, cov) >= portfolio_variance(w, cov)


def test_theory_limits() -> None:
    assert q0_theory(0.0) == 1.0
    assert q1_theory(0.0) == 1.0
    assert q0_theory(0.75) == pytest.approx(2.0)
    assert q0_theory(0.9999) > 99.0
    assert q0_theory(0.3) * q1_theory(0.3) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        q0_theory(1.0)


@pytest.mark.parametrize("ratio", [0.1, 1.0 / 3.0, 0.5, 0.8])
def test_closed_form_matches_the_density_integral(ratio: float) -> None:
    # Pafka and Kondor Eq. (4) integrated numerically against "simple integration yields
    # q0 = 1 / sqrt(1 - r)"
    assert q0_from_density(ratio) == pytest.approx(q0_theory(ratio), rel=1e-6)


@pytest.mark.parametrize(
    ("n", "t", "q0", "q1"),
    [
        (100, 600, 1.09, 0.92),
        (500, 3000, 1.09, 0.91),
        (500, 1500, 1.22, 0.81),
        (500, 750, 1.73, 0.57),
    ],
)
def test_theory_against_published_table(n: int, t: int, q0: float, q1: float) -> None:
    # Pafka and Kondor (2002) Table 1, Model I, simulated means printed to two decimals
    assert q0_theory(n / t) == pytest.approx(q0, abs=0.01)
    assert q1_theory(n / t) == pytest.approx(q1, abs=0.011)


def test_simulated_divergence_sits_where_the_theory_puts_it() -> None:
    rng = np.random.default_rng(11)
    n = 200
    for ratio in (0.25, 0.5, 0.8):
        t = round(n / ratio)
        draws = [simulate_noise_ratios(n, t, rng) for _ in range(20)]
        q0 = np.mean([d.q0 for d in draws])
        q1 = np.mean([d.q1 for d in draws])
        q2 = np.mean([d.q2 for d in draws])
        assert q0 == pytest.approx(q0_theory(n / t), rel=0.03)
        assert q1 == pytest.approx(q1_theory(n / t), rel=0.03)
        assert q2 == pytest.approx(q0, rel=0.03)


def test_q0_is_never_below_one() -> None:
    rng = np.random.default_rng(12)
    draws = [simulate_noise_ratios(20, 60, rng) for _ in range(50)]
    assert min(d.q0 for d in draws) >= 1.0
    assert np.mean([d.q1 for d in draws]) < 1.0


def test_market_mode_covariance_spectrum() -> None:
    cov = market_mode_covariance(40, 25.0)
    values = np.linalg.eigvalsh(cov)
    assert values[-1] == pytest.approx(25.0)
    assert np.allclose(values[:-1], 1.0)
    assert np.allclose(cov @ np.ones(40), 25.0 * np.ones(40))


def test_model_two_gives_the_same_ratios_as_model_one() -> None:
    # Pafka and Kondor Table 1: Model II rows match Model I at the same n and T
    rng = np.random.default_rng(13)
    n, t = 200, 300
    cov = market_mode_covariance(n)
    q0 = np.mean([simulate_noise_ratios(n, t, rng, true_cov=cov).q0 for _ in range(20)])
    assert q0 == pytest.approx(q0_theory(n / t), rel=0.05)


def test_clipping_removes_the_divergence_when_the_truth_is_one_factor() -> None:
    rng = np.random.default_rng(14)
    n, t = 100, 125
    cov = market_mode_covariance(n)

    def cleaner(sigma: np.ndarray, q: float) -> np.ndarray:
        return clip_covariance(sigma, q)

    raw = np.mean([simulate_noise_ratios(n, t, rng, true_cov=cov).q0 for _ in range(10)])
    cleaned = np.mean(
        [simulate_noise_ratios(n, t, rng, true_cov=cov, cleaner=cleaner).q0 for _ in range(10)]
    )
    assert raw > 2.0
    assert cleaned < 1.25


@pytest.mark.parametrize(("n", "t"), [(10, 13), (20, 22), (5, 40)])
def test_exact_law_of_q0_against_simulation(n: int, t: int) -> None:
    """q0^2 = 1 + (N - 1) / (T - N + 2) * F(N - 1, T - N + 2)."""
    rng = np.random.default_rng(n * 1000 + t)
    q0 = np.array([simulate_noise_ratios(n, t, rng).q0 for _ in range(4000)])
    scaled = (q0**2 - 1.0) * (t - n + 2) / (n - 1)
    assert stats.kstest(scaled, stats.f(n - 1, t - n + 2).cdf).pvalue > 0.001
    assert np.median(q0) == pytest.approx(q0_exact_quantile(n, t), rel=0.03)


def test_exact_law_holds_for_a_market_mode_too() -> None:
    rng = np.random.default_rng(8)
    n, t = 12, 16
    cov = market_mode_covariance(n)
    q0 = np.array([simulate_noise_ratios(n, t, rng, true_cov=cov).q0 for _ in range(4000)])
    scaled = (q0**2 - 1.0) * (t - n + 2) / (n - 1)
    assert stats.kstest(scaled, stats.f(n - 1, t - n + 2).cdf).pvalue > 0.001


def test_exact_moments_are_ordered_and_reach_the_limit() -> None:
    assert q0_exact_rms(100, 102) == pytest.approx(np.sqrt(101 / 2))
    for n, t in [(100, 102), (400, 408), (50, 100)]:
        assert q0_exact_quantile(n, t) < q0_exact_mean(n, t) < q0_exact_rms(n, t)
    # the mean climbs towards 1 / sqrt(1 - N/T) as N grows at fixed N/T
    means = [q0_exact_mean(n, round(n / 0.98)) for n in (100, 200, 400, 3200)]
    assert means == sorted(means)
    assert means[-1] == pytest.approx(q0_theory(0.98), rel=0.01)
    assert q0_exact_mean(200, 2000) == pytest.approx(q0_theory(0.1), rel=0.002)
    with pytest.raises(ValueError, match="infinite"):
        q0_exact_rms(10, 10)
    with pytest.raises(ValueError, match="N >= 2"):
        q0_exact_quantile(10, 9)


def test_exact_mean_against_simulation() -> None:
    rng = np.random.default_rng(9)
    q0 = np.array([simulate_noise_ratios(30, 34, rng).q0 for _ in range(4000)])
    se = q0.std(ddof=1) / np.sqrt(q0.size)
    assert abs(q0.mean() - q0_exact_mean(30, 34)) < 4 * se


def test_paired_q0_scores_every_estimator_on_one_sample() -> None:
    estimators = {
        "raw": lambda x: sample_covariance(x, demean=False),
        "again": lambda x: sample_covariance(x, demean=False),
        "identity": lambda x: np.eye(x.shape[1]),
    }
    out = paired_q0(15, 40, np.random.default_rng(10), estimators)
    assert out["raw"] == out["again"] > 1.0
    assert out["identity"] == pytest.approx(1.0)
    same_stream = simulate_noise_ratios(15, 40, np.random.default_rng(10)).q0
    assert out["raw"] == pytest.approx(same_stream)
