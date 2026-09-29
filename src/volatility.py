"""Phase 12 - Risk / Volatility Indicator.

Rule (explicit, reproducible)
-----------------------------
1. Compute the **10-day rolling annualised volatility** of daily returns::

       Volatility_10[t] = std(return[t-9..t]) * sqrt(252)

2. Fix two thresholds from the **training window only** (33rd and 67th
   percentile of ``Volatility_10`` over the training period).  Using training
   percentiles keeps the rule free of look-ahead bias - the test period can
   never influence the definition of "high".

3. Classify the latest value::

       vol <= p33           -> LOW VOLATILITY
       p33   <  vol <= p67  -> MEDIUM VOLATILITY
       vol   >  p67         -> HIGH VOLATILITY

Typical values are annualised: a calm index runs ~8-12% while a stressed
market prints 25%+.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import TRADING_DAYS

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"


def fit_thresholds(train_df: pd.DataFrame, col: str = "Volatility_10") -> dict:
    """Percentile thresholds from the training window (no future information)."""
    q1, q2 = np.nanpercentile(train_df[col].dropna(), [33, 67])
    return {"low_max": float(q1), "medium_max": float(q2),
            "median": float(np.nanmedian(train_df[col].dropna()))}


def classify(vol: float, thresholds: dict) -> str:
    if not np.isfinite(vol):
        return "N/A"
    if vol <= thresholds["low_max"]:
        return LOW
    if vol <= thresholds["medium_max"]:
        return MEDIUM
    return HIGH


def volatility_label(history: pd.DataFrame, thresholds: dict | None = None,
                      window: int = 10) -> dict:
    """Latest volatility level for one instrument's price history.

    ``history`` needs a ``Close`` column sorted chronologically; if a
    ``Volatility_10`` column is absent it is recomputed here.
    """
    hist = history.sort_values("Date") if "Date" in history.columns else history
    close = hist["Close"]
    ret = close.pct_change(fill_method=None)
    vol = float(ret.rolling(window, min_periods=window).std().iloc[-1] * np.sqrt(TRADING_DAYS))
    if thresholds is None:
        thresholds = {"low_max": 0.12, "medium_max": 0.20}
    return {
        "volatility_10": vol,
        "volatility_pct": round(vol * 100, 2),
        "level": classify(vol, thresholds),
        "thresholds": thresholds,
    }


def label_series(df: pd.DataFrame, thresholds: dict, col: str = "Volatility_10") -> pd.Series:
    return df[col].map(lambda v: classify(v, thresholds))


def volatility_report(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      thresholds: dict, symbol: str = "") -> pd.DataFrame:
    """Distribution of volatility levels in train vs. test (for the report)."""
    rows = []
    for split, frame in (("train", train_df), ("test", test_df)):
        labels = label_series(frame, thresholds)
        counts = labels.value_counts(normalize=True).reindex([LOW, MEDIUM, HIGH]).fillna(0)
        rows.append({"Split": split, "Symbol": symbol,
                     **{f"{k} %": round(100 * v, 1) for k, v in counts.items()},
                     "Mean vol %": round(100 * frame["Volatility_10"].mean(), 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .features import build_dataset, make_split

    full = build_dataset()
    train, test = make_split(full)
    th = fit_thresholds(train)
    print("thresholds:", th)
    print(volatility_report(train, test, th).to_string(index=False))
