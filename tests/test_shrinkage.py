"""Ledoit-Wolf shrinkage: the formula, its limits and what it buys."""

from __future__ import annotations

import numpy as np
import pytest

from static.cleaning import sample_covariance
from static.portfolio import market_mode_covariance
from static.shrinkage import ledoit_wolf


def brute_force(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Lemmas 3.2 to 3.4 and Eq. (14) written out with the T outer products."""
    t, n = x.shape
    sample = x.T @ x / t
    m = np.trace(sample) / n
    d2 = np.sum((sample - m * np.eye(n)) ** 2) / n
    b2_bar = sum(np.sum((np.outer(row, row) - sample) ** 2) / n for row in x) / t**2
    b2 = min(b2_bar, d2)
    return b2 / d2 * m * np.eye(n) + (1.0 - b2 / d2) * sample, b2 / d2


@pytest.mark.parametrize(("t", "n"), [(40, 5), (30, 30), (20, 50)])
def test_matches_the_formula_written_out(t: int, n: int) -> None:
    rng = np.random.default_rng(t * 100 + n)
    x = rng.standard_normal((t, n)) * rng.uniform(0.5, 2.0, size=n)
    expected, intensity = brute_force(x)
    result = ledoit_wolf(x, demean=False)
    assert result.intensity == pytest.approx(intensity, rel=1e-10)
    assert np.allclose(result.covariance, expected, rtol=1e-10, atol=1e-12)


def test_demeaning_is_the_same_as_removing_the_mean_first() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 8)) + 3.0
    first = ledoit_wolf(x, demean=True).covariance
    second = ledoit_wolf(x - x.mean(axis=0), demean=False).covariance
    assert np.allclose(first, second)


def test_trace_is_preserved_and_intensity_is_a_weight() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 20))
    result = ledoit_wolf(x)
    assert 0.0 <= result.intensity <= 1.0
    assert np.trace(result.covariance) == pytest.approx(np.trace(sample_covariance(x)))
    assert result.mean_variance == pytest.approx(np.trace(sample_covariance(x)) / 20)


def test_shrinks_fully_when_the_truth_is_the_target() -> None:
    rng = np.random.default_rng(3)
    intensity = [ledoit_wolf(rng.standard_normal((100, 50))).intensity for _ in range(20)]
    assert np.mean(intensity) > 0.9


def test_barely_shrinks_with_many_observations_of_a_structured_matrix() -> None:
    rng = np.random.default_rng(4)
    cov = market_mode_covariance(10)
    x = rng.standard_normal((20000, 10)) @ np.linalg.cholesky(cov).T
    assert ledoit_wolf(x).intensity < 0.01


def test_invertible_with_fewer_observations_than_assets() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((20, 60))
    result = ledoit_wolf(x)
    assert np.linalg.eigvalsh(sample_covariance(x)).min() < 1e-10
    assert np.linalg.eigvalsh(result.covariance).min() > 0.1


def test_closer_to_the_truth_than_the_sample_matrix_on_average() -> None:
    """The guarantee is about expected loss, so the mean over samples is compared."""
    rng = np.random.default_rng(6)
    n, t = 40, 60
    cov = market_mode_covariance(n, ratio=10.0)
    chol = np.linalg.cholesky(cov)
    shrunk, raw = [], []
    for _ in range(200):
        x = rng.standard_normal((t, n)) @ chol.T
        shrunk.append(np.sum((ledoit_wolf(x, demean=False).covariance - cov) ** 2))
        raw.append(np.sum((sample_covariance(x, demean=False) - cov) ** 2))
    assert np.mean(shrunk) < np.mean(raw)


def test_intensity_tracks_the_oracle() -> None:
    """Theorem 2.1: the best fixed weight on the target is beta^2 / delta^2."""
    rng = np.random.default_rng(7)
    n, t = 30, 90
    cov = np.diag(np.linspace(0.5, 2.0, n))
    mu = np.trace(cov) / n
    samples = [rng.standard_normal((t, n)) * np.sqrt(np.diag(cov)) for _ in range(400)]
    covs = [sample_covariance(x, demean=False) for x in samples]
    beta2 = np.mean([np.sum((s - cov) ** 2) / n for s in covs])
    delta2 = np.mean([np.sum((s - mu * np.eye(n)) ** 2) / n for s in covs])
    measured = np.mean([ledoit_wolf(x, demean=False).intensity for x in samples])
    assert measured == pytest.approx(beta2 / delta2, abs=0.03)


def test_degenerate_and_bad_input() -> None:
    with pytest.raises(ValueError, match="T x N"):
        ledoit_wolf(np.zeros(5))
    with pytest.raises(ValueError, match="two observations"):
        ledoit_wolf(np.zeros((1, 3)))
    zero = ledoit_wolf(np.zeros((5, 3)))
    assert zero.intensity == 0.0
    assert np.all(zero.covariance == 0.0)
