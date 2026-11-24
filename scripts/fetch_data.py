"""Download the public data into ``data/``.

Three files from the Kenneth R. French data library: daily returns on 49 industry
portfolios, daily returns on 100 portfolios sorted on size and book-to-market, and the
daily research factors (for the market return).

Four files from G. Lim's Zenodo deposit 10.5281/zenodo.23933497 (processed data under
CC BY 4.0): daily log returns of 628 stocks that were in the S&P 500 at some time since
2000, their daily index membership, and two tables on which members the deposit could
not recover. The archive is 525 MB; only the four files, about 30 MB, are read from it
with range requests.

No key or account is needed for either source.

    uv run python scripts/fetch_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from static.data import BASE_URL, FILES, STOCK_FILES, STOCKS_DOI, STOCKS_PREFIX, STOCKS_URL, vintage
from static.remote import RangeReader, read_members

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "static-research-script (python urllib)"


def fetch(url: str, byte_range: tuple[int, int] | None = None, attempts: int = 4) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=300) as response:
                if byte_range is not None and response.status != 206:
                    raise OSError("the server ignored the range request")
                return bytes(response.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == attempts - 1:
                raise
            print(f"  retrying after: {error}")
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Range"].split("/")[1])


def fetch_stocks(data_dir: Path, force: bool) -> dict[str, str]:
    """Read the stock files out of the Zenodo archive. Returns their SHA-256 digests."""
    folder = data_dir / "stocks"
    folder.mkdir(parents=True, exist_ok=True)
    missing = [name for name in STOCK_FILES if force or not (folder / name).exists()]
    if missing:
        print(f"stocks: reading {len(missing)} files from {STOCKS_URL}")
        reader = RangeReader(lambda a, b: fetch(STOCKS_URL, (a, b)), remote_size(STOCKS_URL))
        members = read_members(reader, [STOCKS_PREFIX + STOCK_FILES[name] for name in missing])
        for name in missing:
            (folder / name).write_bytes(members[STOCKS_PREFIX + STOCK_FILES[name]])
        print(f"  {reader.requests} range requests")
    else:
        print("stocks: already present")
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in STOCK_FILES}


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
    record["stocks"] = {"doi": STOCKS_DOI, "url": STOCKS_URL, **fetch_stocks(data_dir, args.force)}
    record["_fetched"] = {"utc": datetime.now(UTC).isoformat(timespec="seconds")}
    (data_dir / "vintage.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
