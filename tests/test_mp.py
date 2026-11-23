"""Marchenko-Pastur density: closed forms, limits, and pure-noise spectra."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from static.cleaning import sample_correlation, sorted_eigh
from static.mp import fit_noise_variance, ks_distance, mp_bounds, mp_cdf, mp_pdf


def test_bounds_closed_form() -> None:
    lo, hi = mp_bounds(4.0)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(2.25)


def test_bounds_scale_with_sigma2() -> None:
    lo, hi = mp_bounds(3.22, 0.74)
    lo1, hi1 = mp_bounds(3.22)
    assert lo == pytest.approx(0.74 * lo1)
    assert hi == pytest.approx(0.74 * hi1)


def test_lower_edge_is_zero_at_q_one() -> None:
    lo, hi = mp_bounds(1.0)
    assert lo == 0.0
    assert hi == pytest.approx(4.0)


def test_bounds_collapse_as_q_grows() -> None:
    lo, hi = mp_bounds(1e8)
    assert lo == pytest.approx(1.0, abs=1e-3)
    assert hi == pytest.approx(1.0, abs=1e-3)


def test_rejects_q_below_one() -> None:
    with pytest.raises(ValueError):
        mp_bounds(0.5)


@pytest.mark.parametrize(("q", "sigma2"), [(1.5, 1.0), (3.22, 0.74), (10.0, 2.0)])
def test_density_moments(q: float, sigma2: float) -> None:
    lo, hi = mp_bounds(q, sigma2)

    def moment(k: int) -> float:
        return quad(lambda x: x**k * mp_pdf(x, q, sigma2)[0], lo, hi, limit=200)[0]

    assert moment(0) == pytest.approx(1.0, abs=1e-7)
    assert moment(1) == pytest.approx(sigma2, abs=1e-7)
    # second moment of the Marchenko-Pastur law: sigma^4 (1 + 1/Q)
    assert moment(2) == pytest.approx(sigma2**2 * (1.0 + 1.0 / q), abs=1e-6)


def test_density_is_zero_outside_the_edges() -> None:
    lo, hi = mp_bounds(2.0)
    assert np.all(mp_pdf([lo - 0.01, hi + 0.01, -1.0, 100.0], 2.0) == 0.0)


def test_wigner_limit_at_q_one() -> None:
    # At Q = 1 the density is sqrt(4 - l) / (2 pi sqrt(l)), the square of a semicircle
    # variable, and diverges as 1 / sqrt(l) at the origin.
    x = np.array([0.01, 0.5, 2.0, 3.9])
    expected = np.sqrt((4.0 - x) / x) / (2.0 * np.pi)
    assert np.allclose(mp_pdf(x, 1.0), expected)


@pytest.mark.parametrize("q", [1.0, 2.0, 5.0])
def test_cdf_matches_quadrature(q: float) -> None:
    lo, hi = mp_bounds(q)
    points = np.linspace(lo, hi, 9)[1:-1]
    direct = [quad(lambda x: mp_pdf(x, q)[0], lo, p, limit=200)[0] for p in points]
    assert np.allclose(mp_cdf(points, q), direct, atol=1e-6)
    assert mp_cdf([lo - 1.0], q)[0] == 0.0
    assert mp_cdf([hi + 1.0], q)[0] == 1.0


def test_pure_noise_eigenvalues_fall_inside_the_bounds() -> None:
    rng = np.random.default_rng(1)
    n, t = 400, 1600
    values, _ = sorted_eigh(sample_correlation(rng.standard_normal((t, n))))
    lo, hi = mp_bounds(t / n)
    # finite-size smoothing of the edges is of order N^(-2/3)
    slack = 3.0 * n ** (-2.0 / 3.0)
    assert values.max() < hi + slack
    assert values.min() > lo - slack
    assert np.mean((values < lo) | (values > hi)) < 0.02


def test_pure_noise_spectrum_matches_the_density() -> None:
    rng = np.random.default_rng(2)
    n, t = 500, 1500
    values, _ = sorted_eigh(sample_correlation(rng.standard_normal((t, n))))
    assert ks_distance(values, t / n) < 0.02


def test_ks_distance_detects_wrong_q() -> None:
    rng = np.random.default_rng(3)
    n, t = 300, 900
    values, _ = sorted_eigh(sample_correlation(rng.standard_normal((t, n))))
    assert ks_distance(values, 12.0) > 5 * ks_distance(values, 3.0)


def test_fit_on_pure_noise_leaves_sigma2_near_one() -> None:
    rng = np.random.default_rng(4)
    n, t = 300, 1200
    values, _ = sorted_eigh(sample_correlation(rng.standard_normal((t, n))))
    fit = fit_noise_variance(values, t / n)
    assert fit.n_signal <= 2
    assert fit.sigma2 == pytest.approx(1.0, abs=0.02)


def test_fit_with_one_market_mode_gives_laloux_first_step() -> None:
    rng = np.random.default_rng(5)
    n, t = 200, 800
    market = rng.standard_normal((t, 1))
    x = 0.5 * market + rng.standard_normal((t, n))
    values, _ = sorted_eigh(sample_correlation(x))
    fit = fit_noise_variance(values, t / n)
    assert fit.n_signal == 1
    # Laloux: sigma2 = 1 - lambda_1 / N; ours divides by N - 1 instead of N
    assert fit.sigma2 == pytest.approx((n - values[0]) / (n - 1))
    assert fit.sigma2 == pytest.approx(1.0 - values[0] / n, rel=0.01)
