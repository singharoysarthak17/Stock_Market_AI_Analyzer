"""Phase 4 (Feature Engineering), Phase 5 (Target) and Phase 6 (Train/Test split).

Design rules that prevent data leakage (Phase 14):

* Every feature at row ``t`` is computed from rows ``<= t`` only. Rolling and
  diff operations are strictly backward looking, and no column contains
  tomorrow's close or volume.
* The target ``Next_Day_Direction`` is created **after** the features by
  shifting the close one day forward; the final (unknown-target) row of every
  symbol is then dropped. The target itself is never used as a feature.
* Sentiment (Phase 9) is joined with ``shift(1)`` so day ``t`` only sees news
  published on day ``t-1`` or earlier.
* The train/test split is chronological (first 80% / last 20%), never shuffled,
  and the scaler is fit on the training window only (see ``modeling.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators

# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------
BASE_FEATURES = [
    # raw market features (as listed in Phase 4)
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA_5",
    "MA_20",
    "RSI",
    "MACD",
    "Volatility_10",
    "Volume_Change",
    "Previous_Day_Return",
    # engineered extras
    "Close_vs_MA5",
    "Close_vs_MA20",
    "High_Low_Range",
    "Close_Open_Chg",
    "Return_Lag_2",
    "Return_Lag_3",
    "Return_Mean_5",
    "RSI_Distance",
    "MACD_Hist",
    "Volume_Z",
    "DayOfWeek",
    "Month",
]

TARGET = "Next_Day_Direction"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the Phase-4 feature set. Input must already contain indicators."""
    df = df.sort_values(["Symbol", "Date"]).copy()
    g = df.groupby("Symbol", sort=False)

    # ---- A. Why feature engineering? Raw OHLCV alone is not informative to a
    #      tree/linear model: relative and derived measures carry the signal.
    df["Close_vs_MA5"] = df["Close"] / df["MA_5"] - 1.0
    df["Close_vs_MA20"] = df["Close"] / df["MA_20"] - 1.0
    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Close_Open_Chg"] = df["Close"] / df["Open"] - 1.0

    df["Previous_Day_Return"] = g["Daily_Return"].shift(1)
    df["Return_Lag_2"] = g["Daily_Return"].shift(2)
    df["Return_Lag_3"] = g["Daily_Return"].shift(3)
    df["Return_Mean_5"] = g["Daily_Return"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=5).mean()
    )

    df["RSI_Distance"] = df["RSI"] - 50.0
    df["Volume_Z"] = g["Volume"].transform(
        lambda s: (s - s.rolling(60, min_periods=20).mean())
        / s.rolling(60, min_periods=20).std()
    )

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 5 - ``Next_Day_Direction``: 1 if tomorrow's close > today's close else 0."""
    df = df.sort_values(["Symbol", "Date"]).copy()
    future_close = df.groupby("Symbol")["Close"].shift(-1)
    df[TARGET] = (future_close > df["Close"]).astype("float")
    df.loc[future_close.isna(), TARGET] = np.nan  # unknown target -> dropped later
    return df


def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be used: unknown target or warm-up NaNs in features."""
    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    usable = df.dropna(subset=[c for c in BASE_FEATURES if c in df.columns])
    print(f"Feature matrix: {before} rows -> {len(usable)} usable rows "
          f"(warm-up/target rows dropped: {before - len(usable)})")
    return usable.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 6 - chronological split
# ---------------------------------------------------------------------------
def make_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oldest ``train_ratio`` of the sample -> train, the rest -> test.

    One **global date boundary** is used for every instrument, so by
    construction ``train['Date'].max() < test['Date'].min()``: no training row
    can ever be dated after a test row. (Splitting each symbol at its own 80 %
    mark would leave a window in which symbol A is trained on days that symbol
    B is still being tested on.) Rows are never shuffled, and all rows sharing
    the boundary date go to training so the two windows stay disjoint in time.
    """
    if df.empty:
        return df.copy(), df.copy()

    ordered = df.sort_values(["Date", "Symbol"], kind="stable").reset_index(drop=True)
    cut = int(len(ordered) * train_ratio)
    cut = min(max(cut, 1), len(ordered) - 1)          # keep both sides non-empty
    boundary = ordered["Date"].iloc[cut - 1]

    train = ordered[ordered["Date"] <= boundary]
    test = ordered[ordered["Date"] > boundary]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_dataset(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One-shot pipeline: clean -> indicators -> features -> target -> drop NaNs."""
    if df is None:
        from .data_loader import prepare

        df = prepare()
    df = add_indicators(df)
    df = engineer_features(df)
    df = add_target(df)
    return drop_incomplete(df)


FEATURES_USED = BASE_FEATURES


if __name__ == "__main__":
    full = build_dataset()
    tr, te = make_split(full)
    print(f"features ({len(BASE_FEATURES)}): {BASE_FEATURES}")
    print(f"train: {len(tr)} rows  {tr['Date'].min().date()} -> {tr['Date'].max().date()}")
    print(f"test : {len(te)} rows  {te['Date'].min().date()} -> {te['Date'].max().date()}")
    print("target balance (full):", full[TARGET].value_counts(normalize=True).round(3).to_dict())
    assert not full[BASE_FEATURES].isna().any().any(), "NaNs left in feature matrix"
