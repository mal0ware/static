"""Eigenvalue clipping: invariants and recovery of a planted factor."""

from __future__ import annotations

import numpy as np
import pytest

from static.cleaning import (
    clip_correlation,
    clip_covariance,
    correlation_from_covariance,
    count_signal,
    sample_correlation,
    sample_covariance,
    sorted_eigh,
)


def planted(n: int, t: int, loading: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns with one common factor, and their true correlation matrix."""
    rng = np.random.default_rng(seed)
    x = loading * rng.standard_normal((t, 1)) + rng.standard_normal((t, n))
    rho = loading**2 / (1.0 + loading**2)
    truth = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    return x, truth


def test_sample_covariance_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 4))
    assert np.allclose(sample_covariance(x), np.cov(x, rowvar=False, bias=True))
    assert np.allclose(sample_covariance(x, demean=False), x.T @ x / 50)


def test_correlation_has_unit_diagonal_and_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((80, 5)) * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    corr = sample_correlation(x)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, np.corrcoef(x, rowvar=False))


def test_sorted_eigh_is_descending_and_reconstructs() -> None:
    rng = np.random.default_rng(2)
    corr = sample_correlation(rng.standard_normal((60, 6)))
    values, vectors = sorted_eigh(corr)
    assert np.all(np.diff(values) <= 0)
    assert np.allclose((vectors * values) @ vectors.T, corr)


def test_clipping_preserves_trace_and_top_eigenpairs() -> None:
    x, _ = planted(60, 240, 0.6, 3)
    corr = sample_correlation(x)
    cleaned = clip_correlation(corr, 4.0, unit_diagonal=False)
    assert np.trace(cleaned) == pytest.approx(np.trace(corr))
    values, vectors = sorted_eigh(corr)
    k = count_signal(values, 4.0)
    assert k == 1
    c_values, c_vectors = sorted_eigh(cleaned)
    assert c_values[0] == pytest.approx(values[0])
    assert abs(c_vectors[:, 0] @ vectors[:, 0]) == pytest.approx(1.0)
    # every noise eigenvalue is replaced by the same number, their average
    assert np.allclose(c_values[k:], values[k:].mean())


def test_clipping_with_all_kept_is_identity_operation() -> None:
    x, _ = planted(20, 100, 0.5, 4)
    corr = sample_correlation(x)
    assert np.allclose(clip_correlation(corr, 5.0, n_keep=20), corr)


def test_clipping_with_none_kept_gives_identity_matrix() -> None:
    x, _ = planted(20, 100, 0.5, 5)
    assert np.allclose(clip_correlation(sample_correlation(x), 5.0, n_keep=0), np.eye(20))


def test_planted_factor_is_recovered_after_cleaning() -> None:
    n, t = 100, 300
    x, truth = planted(n, t, 0.6, 6)
    corr = sample_correlation(x)
    cleaned = clip_correlation(corr, t / n)
    assert np.allclose(np.diag(cleaned), 1.0)
    values, vectors = sorted_eigh(cleaned)
    market = np.ones(n) / np.sqrt(n)
    assert abs(vectors[:, 0] @ market) > 0.98
    # exactly one eigenvalue stands above the rest
    assert count_signal(sorted_eigh(corr)[0], t / n) == 1
    assert values[1] < 0.5 * values[0]
    # the cleaned matrix is closer to the truth than the raw one
    raw_error = np.linalg.norm(corr - truth)
    clean_error = np.linalg.norm(cleaned - truth)
    assert clean_error < 0.5 * raw_error


def test_clipping_makes_a_singular_matrix_invertible() -> None:
    n, t = 60, 40  # more assets than observations
    x, _ = planted(n, t, 0.6, 7)
    cov = sample_covariance(x)
    assert np.linalg.matrix_rank(cov) < n
    cleaned = clip_covariance(cov, t / n)
    assert np.linalg.eigvalsh(cleaned).min() > 1e-6
    assert np.allclose(np.diag(cleaned), np.diag(cov))


def test_clip_covariance_keeps_sample_variances() -> None:
    rng = np.random.default_rng(8)
    x = rng.standard_normal((200, 30)) * np.linspace(0.5, 3.0, 30)
    cov = sample_covariance(x)
    cleaned = clip_covariance(cov, 200 / 30)
    assert np.allclose(np.diag(cleaned), np.diag(cov))
    assert np.allclose(np.diag(correlation_from_covariance(cleaned)), 1.0)


def test_fitted_edge_never_keeps_fewer_than_the_plain_edge() -> None:
    x, _ = planted(80, 400, 0.8, 9)
    values, _ = sorted_eigh(sample_correlation(x))
    assert count_signal(values, 5.0, fit=True) >= count_signal(values, 5.0)
