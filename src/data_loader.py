"""Phase 1 - Data Collection and Preparation.

Downloads (or reloads) historical OHLCV data for the selected instruments and
performs the cleaning steps required by the assignment:

    1. Load the data using Pandas
    2. Convert the Date column into datetime
    3. Sort all records chronologically
    4. Check for missing values
    5. Check for duplicate records
    6. Handle missing / invalid values
    7. Display basic statistical information
    8. Calculate the percentage daily return for each stock

Dataset source: Yahoo Finance via the ``yfinance`` package.
    * ^NSEI      -> NIFTY 50 index   (market index)
    * RELIANCE.NS -> Reliance Industries Ltd (NSE)
    * TCS.NS      -> Tata Consultancy Services Ltd (NSE)
The raw download is cached in ``data/stock_data.csv`` so the project can be
re-run offline once the cache exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TICKERS = {
    "^NSEI": "NIFTY 50",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
}

START_DATE = "2018-01-01"
# end = today; prices are only useful up to the last completed session
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "stock_data.csv"
PROCESSED_CSV = DATA_DIR / "processed" / "features.csv"

# Columns we keep for modelling
PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------
# 1-3. Load, convert dates, sort
# --------------------------------------------------------------------------
def download_ohlcv(
    tickers: dict[str, str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and cache it to ``data/stock_data.csv``."""
    import yfinance as yf

    frames = []
    for symbol in tickers:
        print(f"Downloading {symbol} ...")
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,  # adjusted prices (splits/dividends handled)
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            print(f"  !! no data returned for {symbol}")
            continue
        # yfinance returns MultiIndex columns for a single ticker in recent versions
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index()
        df["Symbol"] = symbol
        df["Name"] = tickers[symbol]
        frames.append(df)

    if not frames:
        raise RuntimeError("No data downloaded - check network connection.")

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out = out.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return out


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Load the raw dataset, downloading it only when no cache is present."""
    if force_download or not RAW_CSV.exists():
        df = download_ohlcv()
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV, index=False)
        print(f"Cached raw data -> {RAW_CSV}")
    else:
        df = pd.read_csv(RAW_CSV, parse_dates=["Date"])
        print(f"Loaded cached data <- {RAW_CSV}")
    return df


# --------------------------------------------------------------------------
# 4-7. Cleaning
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the required cleaning steps and return (cleaned df, report dict)."""
    report: dict = {}

    # 2. datetime + 3. chronological order (per symbol)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # 4. missing values
    missing_before = df.isna().sum().to_dict()
    report["missing_before"] = {k: int(v) for k, v in missing_before.items() if v}

    # 5. duplicates (same symbol + date)
    dup_mask = df.duplicated(subset=["Symbol", "Date"], keep="first")
    report["duplicate_rows"] = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)

    # 6a. numeric coercion - invalid values become NaN
    for col in PRICE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6b. obviously invalid prices -> NaN
    price_like = ["Open", "High", "Low", "Close"]
    for col in price_like:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan
    if "Volume" in df.columns:
        # zero-volume sessions (halted / non-trading days) are not usable either
        df.loc[df["Volume"] <= 0, "Volume"] = np.nan

    # 6c. fill: prices forward-fill within a symbol (then backfill the head),
    #     volume forward-fill.  Documented in README.md.
    for col in price_like:
        if col in df.columns:
            df[col] = df.groupby("Symbol")[col].transform(lambda s: s.ffill().bfill())
    if "Volume" in df.columns:
        df["Volume"] = df.groupby("Symbol")["Volume"].transform(lambda s: s.ffill().bfill())

    report["missing_after"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    report["rows"] = int(len(df))
    report["symbols"] = df["Symbol"].unique().tolist()
    report["date_min"] = str(df["Date"].min().date())
    report["date_max"] = str(df["Date"].max().date())

    return df, report


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """7. Basic statistical information about the dataset."""
    return (
        df.groupby("Symbol")[PRICE_COLS]
        .describe()
        .round(2)
    )


# --------------------------------------------------------------------------
# 8. Daily percentage return
# --------------------------------------------------------------------------
def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """8. Percentage daily return per stock."""
    df = df.copy()
    df["Daily_Return"] = df.groupby("Symbol")["Close"].pct_change(fill_method=None) * 100.0
    return df


def prepare() -> pd.DataFrame:
    """Full Phase-1 pipeline: load -> clean -> daily return -> cache processed copy."""
    raw = load_raw()
    clean_df, report = clean(raw)
    clean_df = add_daily_return(clean_df)

    print("\n--- Phase 1 cleaning report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return clean_df


if __name__ == "__main__":
    df = prepare()
    print(basic_stats(df))
    print(df.head())
