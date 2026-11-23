"""The experiment tables: shape, sanity, and identical output from run to run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from static import experiments as ex

FAST = ex.Config(fast=True)


def test_simulation_tables_are_identical_from_run_to_run() -> None:
    for build in (ex.divergence_table, ex.mp_edge_table, ex.clipping_sim_table):
        pd.testing.assert_frame_equal(build(FAST), build(FAST), check_exact=True)


def test_feasibility_tables_are_identical_from_run_to_run() -> None:
    pd.testing.assert_frame_equal(
        ex.ml_feasibility_table(FAST), ex.ml_feasibility_table(FAST), check_exact=True
    )
    pd.testing.assert_frame_equal(
        ex.es_feasibility_table(FAST), ex.es_feasibility_table(FAST), check_exact=True
    )


def test_tables_do_not_depend_on_run_order() -> None:
    first = ex.clipping_sim_table(FAST)
    ex.divergence_table(FAST)
    pd.testing.assert_frame_equal(first, ex.clipping_sim_table(FAST), check_exact=True)


def test_a_different_seed_changes_the_draws() -> None:
    other = ex.Config(fast=True, seed=1)
    assert not ex.divergence_table(FAST)["q0_mean"].equals(ex.divergence_table(other)["q0_mean"])


def test_divergence_table_tracks_the_theory() -> None:
    table = ex.divergence_table(FAST)
    assert np.all(table["q0_mean"] >= 1.0)
    low = table[table["N/T"] <= 0.5]
    assert np.allclose(low["q0_mean"], low["q0_theory"], rtol=0.05)
    assert np.allclose(low["q1_mean"], low["q1_theory"], rtol=0.05)


def synthetic_universe(days: int, assets: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("1990-01-01", periods=days, name="date")
    market = 0.01 * rng.standard_normal((days, 1))
    data = market + 0.01 * rng.standard_normal((days, assets))
    return pd.DataFrame(data, index=index, columns=[f"a{i}" for i in range(assets)])


def test_clipping_real_table_on_synthetic_returns() -> None:
    frame = synthetic_universe(700, 12, 0)
    table = ex.clipping_real_table(
        {"toy": frame}, windows=(8, 60, 250), hold=21, start="1990-01-01"
    )
    assert set(table["method"]) == set(ex.CLIP_METHODS)
    short = table[table["T"] == 8].set_index("method")
    assert short.loc["raw", "rebalances"] == 0  # fewer days than assets: no inverse
    assert short.loc["clipped, fitted edge", "rebalances"] == 0
    assert short.loc["clipped", "rebalances"] > 0
    scored = table[table["rebalances"] > 0]
    assert scored["rebalances"].nunique() == 1  # every window is scored on the same days
    equal = scored[scored["method"] == "equal weight"]
    assert equal["gross_leverage"].eq(1.0).all()
    assert equal["realised_vol_pct"].nunique() == 1


def test_absorption_tables_on_synthetic_returns() -> None:
    frame = synthetic_universe(9600, 10, 1)
    frame.index = pd.bdate_range("1990-01-01", periods=len(frame), name="date")
    series = ex.absorption_series(frame, start="1990-01-01")
    assert series["absorption_ratio"].iloc[:499].isna().all()
    valid = series["absorption_ratio"].dropna()
    assert valid.between(0.2, 1.0).all()
    market = frame.mean(axis=1)
    events = ex.absorption_events_table(series, market)
    assert list(events["episode"]) == ["2008", "2020", "2022", "all days"]
    after = ex.absorption_returns_table(series, market)
    assert len(after) == 9
    assert (after["days_shift_above_1"] > 0).all()


def test_spectrum_summary_of_noise() -> None:
    rng = np.random.default_rng(2)
    summary = ex.spectrum_summary(rng.standard_normal((900, 300)))
    assert summary["Q"] == pytest.approx(3.0)
    assert summary["lambda_1/lambda_max"] == pytest.approx(1.0, abs=0.05)
    assert summary["sigma2_fitted"] == pytest.approx(1.0, abs=0.03)
    assert summary["share_of_eigenvalues_in_fitted_band"] > 0.97


def test_es_half_crossings_interpolates() -> None:
    table = pd.DataFrame(
        {
            "N": [10, 10, 10],
            "N/T": [0.3, 0.4, 0.5],
            "simulated": [1.0, 0.75, 0.25],
            "critical_N/T": [0.46] * 3,
        }
    )
    assert ex.es_half_crossings(table)["N/T_at_half"].iloc[0] == pytest.approx(0.45)
