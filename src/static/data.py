"""Readers for the Kenneth French data library files used here."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

#: name -> (zip file on the server, csv inside the zip)
FILES: dict[str, tuple[str, str]] = {
    "industries49": ("49_Industry_Portfolios_daily_CSV.zip", "49_Industry_Portfolios_Daily.csv"),
    "size_bm_100": ("100_Portfolios_10x10_Daily_CSV.zip", "100_Portfolios_10x10_Daily.csv"),
    "factors": ("F-F_Research_Data_Factors_daily_CSV.zip", "F-F_Research_Data_Factors_daily.csv"),
}

VALUE_WEIGHTED = "Average Value Weighted Returns -- Daily"


def parse_french_csv(text: str, section: str | None = None) -> pd.DataFrame:
    """Parse one table of a French library CSV into decimal daily returns.

    ``section`` is the title line above the table; ``None`` takes the first table.
    Values are in percent in the file. The missing codes -99.99 and -999 become NaN.
    """
    lines = text.splitlines()
    if section is None:
        start = next(i for i, line in enumerate(lines) if line.startswith(","))
    else:
        title = next(i for i, line in enumerate(lines) if section in line)
        start = title + 1
    end = start + 1
    while end < len(lines) and lines[end].strip() and lines[end].lstrip()[:1].isdigit():
        end += 1
    frame = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str).str.strip(), format="%Y%m%d")
    frame.index.name = "date"
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.astype(np.float64)
    frame = frame.where(frame > -99.0)
    return frame / 100.0


def vintage(text: str) -> str:
    """The first line of a French file, which names the CRSP database it was built from."""
    return text.splitlines()[0].strip()


def load(name: str, data_dir: Path) -> pd.DataFrame:
    """Load one of :data:`FILES` from ``data_dir`` (after ``scripts/fetch_data.py``)."""
    _, csv_name = FILES[name]
    path = data_dir / csv_name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/fetch_data.py first")
    text = path.read_text(encoding="latin-1")
    section = None if name == "factors" else VALUE_WEIGHTED
    return parse_french_csv(text, section)


def complete_window(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows from ``start`` to ``end``, keeping only columns with no missing value there."""
    window = frame.loc[start:end]
    return window.loc[:, window.notna().all(axis=0)]
