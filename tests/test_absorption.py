"""Absorption ratio: closed forms and weighting."""

from __future__ import annotations

import numpy as np
import pytest

from static.absorption import (
    absorption_ratio,
    default_components,
    exponential_weights,
    rolling_absorption_ratio,
    standardized_shift,
    weighted_covariance,
)


def test_weights_sum_to_one_and_halve_at_the_half_life() -> None:
    w = exponential_weights(500, 250.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[-1] / w[-1 - 250] == pytest.approx(2.0)
    assert np.all(np.diff(w) > 0)  # newest observation weighs most


def test_equal_weights_when_no_half_life() -> None:
    assert np.allclose(exponential_weights(10, None), 0.1)


def test_weighted_covariance_with_equal_weights_is_sample_covariance() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 5))
    cov = weighted_covariance(x, exponential_weights(120, None))
    assert np.allclose(cov, np.cov(x, rowvar=False, bias=True))


def test_long_half_life_tends_to_equal_weights() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((100, 4))
    slow = weighted_covariance(x, exponential_weights(100, 1e9))
    assert np.allclose(slow, np.cov(x, rowvar=False, bias=True), atol=1e-6)


def test_absorption_ratio_of_identity_is_n_over_N() -> None:  # noqa: N802
    assert absorption_ratio(np.eye(50), 10) == pytest.approx(0.2)


def test_absorption_ratio_with_all_components_is_one() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((40, 8))
    assert absorption_ratio(a.T @ a, 8) == pytest.approx(1.0)


def test_one_factor_closed_form() -> None:
    # equicorrelation rho: eigenvalues 1 + (N - 1) rho once and 1 - rho otherwise
    n, rho, k = 49, 0.4, 10
    cov = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    expected = (1.0 + (n - 1) * rho + (k - 1) * (1.0 - rho)) / n
    assert absorption_ratio(cov, k) == pytest.approx(expected)


def test_absorption_ratio_rises_with_correlation() -> None:
    n = 30
    values = []
    for rho in (0.0, 0.2, 0.5, 0.8):
        cov = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
        values.append(absorption_ratio(cov, 6))
    assert np.all(np.diff(values) > 0)


def test_absorption_ratio_is_scale_invariant() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((60, 10))
    cov = a.T @ a / 60
    assert absorption_ratio(7.0 * cov, 2) == pytest.approx(absorption_ratio(cov, 2))


def test_default_components_matches_the_paper() -> None:
    assert default_components(51) == 10  # Kritzman et al.: 51 industries, 10 eigenvectors
    assert default_components(49) == 10
    assert default_components(2) == 1


def test_rolling_matches_direct_calculation() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((80, 10))
    ar = rolling_absorption_ratio(x, window=50, half_life=25.0, n_components=2)
    assert np.all(np.isnan(ar[:49]))
    assert np.all(np.isfinite(ar[49:]))
    direct = absorption_ratio(weighted_covariance(x[30:80], exponential_weights(50, 25.0)), 2)
    assert ar[-1] == pytest.approx(direct)


def test_rolling_uses_only_past_data() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((120, 8))
    full = rolling_absorption_ratio(x, window=40, half_life=20.0)
    changed = x.copy()
    changed[100:] *= 10.0
    again = rolling_absorption_ratio(changed, window=40, half_life=20.0)
    assert np.allclose(again[:100], full[:100], equal_nan=True)
    assert not np.allclose(again[100:], full[100:])


def test_standardized_shift_definition() -> None:
    rng = np.random.default_rng(6)
    ar = 0.8 + 0.01 * rng.standard_normal(400).cumsum()
    shift = standardized_shift(ar, short=15, long=252)
    assert np.all(np.isnan(shift[:251]))
    expected = (ar[-15:].mean() - ar[-252:].mean()) / ar[-252:].std(ddof=1)
    assert shift[-1] == pytest.approx(expected)


def test_standardized_shift_of_a_constant_is_nan() -> None:
    assert np.all(np.isnan(standardized_shift(np.full(300, 0.7))))
