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


def test_min_variance_table_on_synthetic_returns() -> None:
    frame = synthetic_universe(700, 12, 0)
    table = ex.min_variance_table(
        {"toy": frame}, windows=(8, 60, 250), hold=21, start="1990-01-01", n_boot=50
    )
    assert set(table["method"]) == set(ex.CLIP_METHODS)
    short = table[table["T"] == 8].set_index("method")
    assert short.loc["raw", "rebalances"] == 0  # fewer days than assets: no inverse
    assert short.loc["clipped, fitted edge", "rebalances"] == 0
    assert short.loc["clipped", "rebalances"] > 0
    assert short.loc["ledoit-wolf", "rebalances"] > 0
    assert 0.0 < short.loc["ledoit-wolf", "shrinkage_mean"] <= 1.0
    assert (table["N_min"] == 12).all() and (table["N_max"] == 12).all()
    ratios = table.dropna(subset=["vol_ratio_to_ledoit_wolf"])
    assert (ratios["ratio_lo"] <= ratios["vol_ratio_to_ledoit_wolf"]).all()
    assert (ratios["vol_ratio_to_ledoit_wolf"] <= ratios["ratio_hi"]).all()
    assert table[table["method"] == "ledoit-wolf"]["vol_ratio_to_ledoit_wolf"].isna().all()
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
    assert (after["runs_shift_above_1"] <= after["days_shift_above_1"]).all()
    assert (after["difference_lo_pct"] < after["difference_hi_pct"]).all()
    assert "naive_t_of_difference" not in after.columns
    again = ex.absorption_returns_table(series, market)
    pd.testing.assert_frame_equal(after, again, check_exact=True)


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


def test_universe_is_chosen_from_what_was_known_at_each_rebalance() -> None:
    frame = synthetic_universe(700, 6, 3)
    late = frame.copy()
    late.iloc[:150, 0] = np.nan  # asset 0 has no history at first
    table = ex.min_variance_table(
        {"toy": late}, windows=(20, 100), hold=21, start="1990-01-01", n_boot=20
    )
    assert table["N_min"].iloc[0] == 5 and table["N_max"].iloc[0] == 6

    # a gap that lies in the future of every rebalance but the last few must not
    # change what was held before it
    gap = frame.copy()
    gap.iloc[650:660, 1] = np.nan
    cut = 600
    early_full = ex.min_variance_table(
        {"toy": frame.iloc[:cut]}, windows=(20, 100), hold=21, start="1990-01-01", n_boot=20
    )
    early_gap = ex.min_variance_table(
        {"toy": gap.iloc[:cut]}, windows=(20, 100), hold=21, start="1990-01-01", n_boot=20
    )
    pd.testing.assert_frame_equal(early_full, early_gap, check_exact=True)
    with_gap = ex.min_variance_table(
        {"toy": gap}, windows=(20, 100), hold=21, start="1990-01-01", n_boot=20
    )
    assert with_gap["N_min"].iloc[0] == 5 and with_gap["N_max"].iloc[0] == 6


def test_membership_flags_limit_the_universe() -> None:
    frame = synthetic_universe(400, 6, 4)
    member = pd.DataFrame(True, index=frame.index, columns=frame.columns)
    member.iloc[:, 5] = False
    member.iloc[300:, 4] = False
    table = ex.min_variance_table(
        {"toy": frame},
        windows=(30, 90),
        hold=21,
        start="1990-01-01",
        members={"toy": member},
        n_boot=20,
    )
    assert table["N_max"].iloc[0] == 5 and table["N_min"].iloc[0] == 4


def test_a_missing_return_while_held_counts_as_zero() -> None:
    frame = synthetic_universe(200, 4, 5)
    holed = frame.copy()
    holed.iloc[150:171, 2] = np.nan
    zeroed = frame.copy()
    zeroed.iloc[150:171, 2] = 0.0
    kwargs = {"windows": (30, 150), "hold": 21, "start": "1990-01-01", "n_boot": 20}
    a = ex.min_variance_table({"toy": holed.iloc[:171]}, **kwargs)
    b = ex.min_variance_table({"toy": zeroed.iloc[:171]}, **kwargs)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_clipping_sim_table_is_paired() -> None:
    table = ex.clipping_sim_table(FAST)
    assert np.allclose(table["clipped_minus_raw"], table["q0_clipped"] - table["q0_raw"])
    assert (table["clipped_minus_raw_se"] > 0).all()
    assert (table["q0_ledoit_wolf"] >= 1.0).all()


def test_divergence_table_reports_the_finite_size_law() -> None:
    table = ex.divergence_table(FAST)
    assert (table["q0_median_lo"] <= table["q0_median"]).all()
    assert (table["q0_median"] <= table["q0_median_hi"]).all()
    assert (table["q0_median_exact"] < table["q0_rms_exact"]).all()
    assert (table["z_mean_vs_exact"].abs() < 4.0).all()


def test_stock_windows_and_spectra_on_synthetic_returns() -> None:
    frame = synthetic_universe(1000, 30, 6)
    frame.iloc[:400, 0] = np.nan
    member = pd.DataFrame(True, index=frame.index, columns=frame.columns)
    member.iloc[:, 1] = False
    member.iloc[450:, 2] = False
    windows = ex.stock_windows(frame, member, length=300)
    assert [len(w) for w, _ in windows] == [300, 300, 300]
    assert windows[0][0].index[0] == frame.index[100]
    assert windows[-1][0].index[-1] == frame.index[-1]
    # first window ends on day 399: asset 0 has gaps, asset 1 is not a member
    assert windows[0][0].shape[1] == 28 and windows[0][1] == 29
    assert windows[-1][0].shape[1] == 28 and windows[-1][1] == 28
    table = ex.stock_spectrum_table(frame, member, length=300)
    assert len(table) == 3
    assert (table["n_above_first_step_edge"] >= 1).all()
    assert table["share_of_eigenvalues_in_first_step_band"].between(0.5, 1.0).all()


def test_first_step_band_holds_a_one_factor_spectrum() -> None:
    rng = np.random.default_rng(12)
    returns = rng.standard_normal((1200, 300)) + 0.6 * rng.standard_normal((1200, 1))
    summary = ex.spectrum_summary(returns)
    assert summary["n_above_first_step_edge"] == 1
    assert summary["share_of_eigenvalues_in_first_step_band"] > 0.97


def test_ml_recheck_reruns_the_farthest_cell_with_more_samples() -> None:
    table = ex.ml_feasibility_table(FAST)
    recheck = ex.ml_recheck_table(FAST, table)
    assert len(recheck) >= 1
    assert (recheck["samples"] == 10 * recheck["first_samples"]).all()
    worst = table.loc[table["z_score"].abs().idxmax()]
    assert recheck["first_z_score"].abs().max() == pytest.approx(abs(worst["z_score"]))
    forced = ex.ml_recheck_table(FAST, table, threshold=0.0)
    assert len(forced) == len(table)


def test_the_cell_that_was_an_outlier_in_the_first_seeded_run() -> None:
    """N = 20, T = 44 came out 3.5 standard errors high in the first 400 samples."""
    from static.feasibility import minimise_maximal_loss, ml_feasible_probability

    rng = np.random.default_rng(44)
    reps = 2000
    hits = sum(minimise_maximal_loss(rng.standard_normal((44, 20))).bounded for _ in range(reps))
    exact = ml_feasible_probability(20, 44)
    assert abs(hits / reps - exact) < 3.0 * np.sqrt(exact * (1 - exact) / reps)


def test_widest_band_holds_all_of_a_noise_spectrum_and_bounds_the_other_shares() -> None:
    rng = np.random.default_rng(13)
    noise = ex.spectrum_summary(rng.standard_normal((900, 300)))
    assert noise["largest_share_in_any_band"] > 0.99
    assert noise["sigma2_of_that_band"] == pytest.approx(1.0, abs=0.05)
    factor = ex.spectrum_summary(
        rng.standard_normal((900, 300)) * np.linspace(0.2, 3.0, 300) + rng.standard_normal((900, 1))
    )
    assert (
        factor["largest_share_in_any_band"] >= factor["share_of_eigenvalues_in_fitted_band"] - 0.02
    )
    assert (
        factor["largest_share_in_any_band"]
        >= factor["share_of_eigenvalues_in_first_step_band"] - 0.02
    )
