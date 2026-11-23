"""Download the public data into ``data/``.

Three files from the Kenneth R. French data library: daily returns on 49 industry
portfolios, daily returns on 100 portfolios sorted on size and book-to-market, and the
daily research factors (for the market return). No key or account is needed.

    uv run python scripts/fetch_data.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from static.data import BASE_URL, FILES, vintage

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "static-research-script (python urllib)"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--force", action="store_true", help="download even if present")
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, dict[str, str]] = {}
    for name, (zip_name, csv_name) in FILES.items():
        target = data_dir / csv_name
        url = BASE_URL + zip_name
        if target.exists() and not args.force:
            print(f"{name}: already present ({target.name})")
        else:
            print(f"{name}: downloading {url}")
            with zipfile.ZipFile(io.BytesIO(fetch(url))) as archive:
                target.write_bytes(archive.read(csv_name))
        text = target.read_text(encoding="latin-1")
        record[name] = {"url": url, "file": csv_name, "vintage": vintage(text)}
        print(f"  {record[name]['vintage']}")
    record["_fetched"] = {"utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (data_dir / "vintage.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
