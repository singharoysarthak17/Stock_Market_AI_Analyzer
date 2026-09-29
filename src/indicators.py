"""Phase 3 - Technical Indicator Calculation.

Every indicator is computed **per symbol** and uses only information available
on or before the row's date (pure backward-looking rolling windows), which is
the first line of defence against look-ahead bias.

What each indicator is and why it is useful in an ML model:

MA_5 (5-day Moving Average)
    Simple average of the last 5 closing prices. Short-horizon trend; the
    distance between price and MA_5 is a momentum/trend cue.

MA_20 (20-day Moving Average)
    Simple average of the last 20 closing prices. Medium-horizon trend and a
    common support/resistance proxy; price-vs-MA_20 captures trend regime.

Daily_Return (%)
    Percentage change of the close versus the previous close. Raw one-day
    momentum; the model sees the recent return history.

RSI (Relative Strength Index, 14 periods)
    100 - 100/(1 + RS), RS = avg gain / avg loss over 14 days (Wilder's
    smoothing). Oscillates 0-100 and flags overbought (>70) / oversold (<30)
    conditions, i.e. mean-reversion information.

MACD (12/26/9)
    MACD line = EMA12 - EMA26; signal line = EMA9 of the MACD line;
    histogram = MACD - signal. Trend/momentum change detection; the histogram
    sign is a classic trend-shift signal.

Volatility_10 (10-day Rolling Volatility)
    Standard deviation of the last 10 daily returns, annualised (x sqrt(252)).
    Calm vs. turbulent regime; also drives the Phase-12 risk indicator.

Volume_Change (%)
    Percentage change of volume versus the previous session. Sudden volume
    spikes often precede/confirm large price moves (news, results).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # Indian markets: ~250-252 sessions per year

INDICATOR_COLS = [
    "MA_5",
    "MA_20",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Volatility_10",
    "Volume_Change",
]


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-smoothed Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's EMA smoothing (alpha = 1/window)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # when there is no loss at all RSI is 100; no gain at all RSI is 0
    out = out.where(avg_loss != 0, 100.0)
    out = out.where((avg_gain != 0) | (avg_loss != 0), np.nan)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all Phase-3 indicators added per symbol."""
    df = df.sort_values(["Symbol", "Date"]).copy()
    g = df.groupby("Symbol", sort=False)

    # A-B. Moving averages
    df["MA_5"] = g["Close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    df["MA_20"] = g["Close"].transform(lambda s: s.rolling(20, min_periods=20).mean())

    # C. Daily return (in fraction, not %) - stable for z-scoring later
    df["Daily_Return"] = g["Close"].transform(lambda s: s.pct_change(fill_method=None))

    # D. RSI
    df["RSI"] = g["Close"].transform(lambda s: rsi(s, 14))

    # E. MACD
    df["MACD"] = g["Close"].transform(lambda s: macd(s)[0])
    df["MACD_Signal"] = g["Close"].transform(lambda s: macd(s)[1])
    df["MACD_Hist"] = g["Close"].transform(lambda s: macd(s)[2])

    # F. 10-day rolling volatility (annualised)
    df["Volatility_10"] = (
        g["Daily_Return"].transform(lambda s: s.rolling(10, min_periods=10).std())
        * np.sqrt(TRADING_DAYS)
    )

    # G. Volume change (%)
    df["Volume_Change"] = g["Volume"].transform(lambda s: s.pct_change(fill_method=None))

    return df


def indicator_descriptions() -> pd.DataFrame:
    """Small helper used by the report / app to print what each indicator is."""
    rows = [
        ("MA_5", "5-day simple moving average of close", "Short-term trend level"),
        ("MA_20", "20-day simple moving average of close", "Medium-term trend level"),
        ("Daily_Return", "(Close[t] - Close[t-1]) / Close[t-1]", "One-day momentum"),
        ("RSI", "100 - 100/(1 + avg_gain/avg_loss), 14 days", "Overbought/oversold (0-100)"),
        ("MACD", "EMA12(Close) - EMA26(Close)", "Trend/momentum line"),
        ("MACD_Signal", "EMA9(MACD)", "MACD trigger line"),
        ("MACD_Hist", "MACD - MACD_Signal", "Trend-shift histogram"),
        ("Volatility_10", "std(ret,10) * sqrt(252)", "Annualised 10-day risk"),
        ("Volume_Change", "(Vol[t] - Vol[t-1]) / Vol[t-1]", "Volume shock detector"),
    ]
    return pd.DataFrame(rows, columns=["Indicator", "Calculation", "Why it helps ML"])


if __name__ == "__main__":
    from src.data_loader import prepare

    df = add_indicators(prepare())
    print(indicator_descriptions().to_string(index=False))
    print(df[["Date", "Symbol", "Close"] + INDICATOR_COLS].tail(10).to_string(index=False))
