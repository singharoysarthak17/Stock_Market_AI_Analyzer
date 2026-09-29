"""Phase 9 - Financial news headline collection.

Headlines are collected from the **GDELT Document API** (free, no API key),
which stores real, historically dated news article headlines.

    https://api.gdeltproject.org/api/v2/doc/doc

For each of the three instruments we search an exact phrase that identifies it:

    RELIANCE.NS ->  "Reliance Industries"
    TCS.NS       ->  "Tata Consultancy Services"
    ^NSEI        ->  "Nifty 50"

Results are cached on disk (``data/raw/``) so the API is only hit once, and are
merged into ``data/news_data.csv``.  GDELT throttles aggressively (HTTP 429),
hence the exponential backoff.

Usage:
    python -m src.collect_news            # incremental (cached rows reused)
    python -m src.collect_news --force    # refetch everything
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
NEWS_CSV = PROJECT_ROOT / "data" / "news_data.csv"

API = "https://api.gdeltproject.org/api/v2/doc/doc"

QUERIES = {
    "RELIANCE.NS": '"Reliance Industries"',
    "TCS.NS": '"Tata Consultancy Services"',
    "^NSEI": '"Nifty 50"',
}

START = datetime(2018, 1, 1)

# Days per API request, per symbol.  Longer windows = fewer requests = less
# time lost to GDELT's HTTP 429 throttling.  Only symbols that have not been
# collected yet are given a larger window, so existing caches stay valid.
WINDOW_DAYS: dict[str, int] = {
    "^NSEI": 730,
}
MAX_RECORDS = 100        # smaller pages are far less likely to be throttled
SLEEP_SECONDS = 5.0     # GDELT asks for at most one request every 5 seconds
MAX_RETRIES = 4
BASE_DELAY = 15.0
MAX_DELAY = 60.0


# --------------------------------------------------------------------------
def _get(url: str) -> bytes:
    """GET with exponential backoff on throttling / transient errors."""
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 502, 503, 504) and attempt < MAX_RETRIES:
                wait = delay
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, float(retry_after))
                wait = min(wait, MAX_DELAY)
                print(f"    HTTP {exc.code} - retry {attempt}/{MAX_RETRIES} in {wait:.0f}s")
                time.sleep(wait)
                delay = min(delay * 2, MAX_DELAY)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < MAX_RETRIES:
                print(f"    network error {exc} - retry {attempt}/{MAX_RETRIES} in {delay:.0f}s")
                time.sleep(min(delay, MAX_DELAY))
                delay = min(delay * 2, MAX_DELAY)
                continue
            raise
    raise RuntimeError("unreachable")


def fetch_window(symbol: str, start: datetime, end: datetime) -> list[dict]:
    """Fetch one time window for one symbol."""
    query = urllib.parse.quote(f'{QUERIES[symbol]} sourcelang:english')
    url = (
        f"{API}?query={query}&mode=artlist&format=json&sort=dateasc"
        f"&maxrecords={MAX_RECORDS}"
        f"&startdatetime={start:%Y%m%d}000000"
        f"&enddatetime={end:%Y%m%d}235959"
    )
    payload = _get(url)
    try:
        doc = json.loads(payload.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return []
    rows = []
    for art in doc.get("articles", []):
        seen = art.get("seendate", "")
        if len(seen) < 8:
            continue
        rows.append(
            {
                "Symbol": symbol,
                "Date": datetime.strptime(seen[:8], "%Y%m%d").date().isoformat(),
                "Headline": (art.get("title") or "").strip(),
                "Domain": art.get("domain", ""),
                "SourceCountry": art.get("sourcecountry", ""),
                "URL": art.get("url", ""),
                "CollectedAt": datetime.now().isoformat(timespec="seconds"),
            }
        )
    return rows


def collect(force: bool = False, end: datetime | None = None,
            symbols: list[str] | None = None) -> pd.DataFrame:
    """Collect every window for every symbol (cached per window).

    Resilient: if a single window keeps failing after the backoff it is skipped
    (and left un-cached so a later run can retry it) instead of aborting the run.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    end = end or datetime.now()
    all_rows: list[dict] = []
    failures: list[str] = []

    for symbol in (symbols or list(QUERIES)):
        if symbol not in QUERIES:
            print(f"!! unknown symbol {symbol} (choose from {list(QUERIES)})")
            continue
        cache_file = RAW_DIR / f"gdelt_{symbol.replace('^', 'IDX_').replace('.', '_')}.json"
        cached: dict[str, list[dict]] = {}
        if cache_file.exists() and not force:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            n = sum(len(v) for v in cached.values())
            print(f"{symbol}: reusing cached windows ({n} rows)")

        start = START
        while start < end:
            # ^NSEI has no cached windows yet, so it is collected in two-year
            # chunks instead of one-year chunks: half the API requests means
            # half the exposure to GDELT's rate limiter.  Symbols that already
            # have one-year windows cached keep their original window size so
            # the cache stays valid.
            window_days = WINDOW_DAYS.get(symbol, 364)
            stop = min(start + timedelta(days=window_days), end)
            key = f"{start:%Y%m%d}_{stop:%Y%m%d}"
            if key not in cached:
                print(f"  {symbol} {key} ...", end=" ")
                sys.stdout.flush()
                try:
                    rows = fetch_window(symbol, start, stop)
                except Exception as exc:          # noqa: BLE001 - keep going
                    print(f"FAILED ({exc}); skipping this window")
                    failures.append(f"{symbol} {key}")
                    start = stop + timedelta(days=1)
                    continue
                cached[key] = rows
                print(f"{len(rows)} headlines")
                cache_file.write_text(json.dumps(cached), encoding="utf-8")
                time.sleep(SLEEP_SECONDS)
            all_rows.extend(cached.get(key, []))
            start = stop + timedelta(days=1)

    if failures:
        print(f"\n{len(failures)} window(s) failed and will be retried on the next run: "
              f"{', '.join(failures)}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    # GDELT sometimes returns a record with an empty title - that is not a
    # headline, and it would poison both the dataset and the VADER averages.
    df["Headline"] = df["Headline"].astype(str).str.strip()
    df = df[df["Headline"].ne("") & df["Headline"].str.lower().ne("nan")]
    if "SourceCountry" in df.columns:
        df["SourceCountry"] = df["SourceCountry"].fillna("Unknown")
    df = (
        df.drop_duplicates(subset=["Symbol", "Date", "Headline"])
        .sort_values(["Symbol", "Date"])
        .reset_index(drop=True)
    )
    return df


def build(force: bool = False, symbols: list[str] | None = None) -> pd.DataFrame:
    """Collect and write ``data/news_data.csv`` (merges with what is already there)."""
    df = collect(force=force, symbols=symbols)
    if df.empty:
        raise RuntimeError("No headlines collected - check network access.")

    NEWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if NEWS_CSV.exists() and not force:
        existing = pd.read_csv(NEWS_CSV, parse_dates=["Date"])
        fresh_symbols = set(df["Symbol"].unique())
        existing = existing[~existing["Symbol"].isin(fresh_symbols)]
        df = pd.concat([existing, df], ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df = (df.drop_duplicates(subset=["Symbol", "Date", "Headline"])
                .sort_values(["Symbol", "Date"]).reset_index(drop=True))

    df.to_csv(NEWS_CSV, index=False)
    print(f"\nWrote {len(df)} headlines -> {NEWS_CSV}")
    print(df.groupby("Symbol")["Date"].agg(["count", "min", "max"]).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore the local cache")
    ap.add_argument("--symbol", action="append", choices=list(QUERIES),
                    help="only collect this symbol (repeatable)")
    args = ap.parse_args()
    build(force=args.force, symbols=args.symbol)
