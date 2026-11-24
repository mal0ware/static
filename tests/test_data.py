"""Parsing of French library files, on a small inline sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from static.data import (
    complete_window,
    load_stocks,
    parse_french_csv,
    stock_coverage,
    vintage,
)

SAMPLE = """This file was created using the 202609 CRSP database.
It contains value- and equal-weighted returns for 3 portfolios.

Missing data are indicated by -99.99 or -999.

  Average Value Weighted Returns -- Daily
,Agric,Food ,Soda
19260701,   0.56,  -0.07, -99.99
19260702,   0.29,   0.06, -99.99
19260706,  -0.33,   0.18,   1.00

  Average Equal Weighted Returns -- Daily
,Agric,Food ,Soda
19260701,   9.00,   9.00,   9.00

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

FACTORS = """This file was created by using the 202609 CRSP database.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
19260702,    0.45,   -0.34,   -0.03,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_parses_the_named_section_only() -> None:
    frame = parse_french_csv(SAMPLE, "Average Value Weighted Returns -- Daily")
    assert list(frame.columns) == ["Agric", "Food", "Soda"]
    assert frame.shape == (3, 3)
    assert frame.index[2].strftime("%Y-%m-%d") == "1926-07-06"
    assert frame.loc["1926-07-01", "Agric"] == pytest.approx(0.0056)


def test_missing_codes_become_nan() -> None:
    frame = parse_french_csv(SAMPLE, "Average Value Weighted Returns -- Daily")
    assert np.isnan(frame.loc["1926-07-01", "Soda"])
    assert frame.loc["1926-07-06", "Soda"] == pytest.approx(0.01)


def test_first_table_when_no_section_is_given() -> None:
    frame = parse_french_csv(FACTORS)
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert frame.shape == (2, 4)
    assert frame.loc["1926-07-02", "Mkt-RF"] == pytest.approx(0.0045)


def test_vintage_line() -> None:
    assert "202609 CRSP" in vintage(SAMPLE)


def test_complete_window_drops_columns_with_gaps() -> None:
    frame = parse_french_csv(SAMPLE, "Average Value Weighted Returns -- Daily")
    assert list(complete_window(frame, "1926-07-01", "1926-07-06").columns) == ["Agric", "Food"]
    assert list(complete_window(frame, "1926-07-06", "1926-07-06").columns) == [
        "Agric",
        "Food",
        "Soda",
    ]


def test_load_stocks_turns_log_returns_into_simple_returns(tmp_path: Path) -> None:
    folder = tmp_path / "stocks"
    folder.mkdir()
    index = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"])
    log_returns = pd.DataFrame(
        {"AAA": [0.0, np.log(1.1), np.log(0.5)], "BBB": [np.nan, np.nan, 0.0]}, index=index
    )
    member = pd.DataFrame({"AAA": [True, True, False], "BBB": [False, False, True]}, index=index)
    log_returns.to_parquet(folder / "logret.parquet")
    member.to_parquet(folder / "membership_daily.parquet")
    returns, flags = load_stocks(tmp_path)
    assert list(returns.index.strftime("%Y-%m-%d")) == ["2000-01-04", "2000-01-05"]
    assert returns["AAA"].tolist() == pytest.approx([0.1, -0.5])
    assert np.isnan(returns["BBB"].iloc[0])
    assert flags["BBB"].tolist() == [False, True]

    member.iloc[:, ::-1].to_parquet(folder / "membership_daily.parquet")
    with pytest.raises(ValueError, match="line up"):
        load_stocks(tmp_path)


def test_stock_files_must_be_fetched_first(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_data"):
        load_stocks(tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_data"):
        stock_coverage(tmp_path)
