"""Helpers used by the Streamlit app to produce a live prediction.

The whole point of this module is *reproducibility*: features for the app are
built with the exact same functions used during training
(``indicators.add_indicators`` + ``features.engineer_features``), so the app can
never drift away from the model's expected inputs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import BASE_FEATURES, engineer_features
from .indicators import add_indicators
from .volatility import classify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = PROJECT_ROOT / "data" / "live"
NEWS_CSV = PROJECT_ROOT / "data" / "news_data.csv"

DISPLAY = {
    "^NSEI": "NIFTY 50 Index",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
}


def _cache_path(symbol: str) -> Path:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    return LIVE_DIR / f"{symbol.replace('^', 'IDX_').replace('.', '_')}.csv"


def get_history(symbol: str, force_refresh: bool = False, days: int = 550) -> pd.DataFrame:
    """Recent OHLCV history, downloaded once per day and cached on disk."""
    path = _cache_path(symbol)

    if path.exists() and not force_refresh:
        cached = pd.read_csv(path, parse_dates=["Date"])
        if not cached.empty:
            written = datetime.fromtimestamp(path.stat().st_mtime).date()
            latest = pd.Timestamp(cached["Date"].max()).date()
            # reuse the cache when the file was written today or already holds today's bar
            if written == datetime.now().date() or latest == datetime.now().date():
                return cached

    try:
        import yfinance as yf

        raw = yf.download(
            symbol, period=f"{max(days, 400)}d", auto_adjust=True,
            progress=False, threads=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError("empty response")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["Symbol"] = symbol
        df["Name"] = DISPLAY.get(symbol, symbol)
        df.to_csv(path, index=False)
        return df
    except Exception as exc:  # offline / rate-limited -> fall back to the cache
        if path.exists():
            return pd.read_csv(path, parse_dates=["Date"])
        raise RuntimeError(
            f"Could not download {symbol} ({exc}) and no local cache exists."
        )


# sentiment columns a "with news" model expects as extra features
NEWS_FEATURES = ["Headlines", "News_Sentiment_Compound", "News_Sentiment_Mean_Value",
                 "News_Sentiment"]


def _add_news_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Attach the lagged rolling sentiment columns a with-news model expects.

    Uses the very same ``sentiment.add_sentiment_to_features`` code path as
    training (5-day trailing window, 1-day lag), so live rows can never drift
    away from what the model learned.  Missing/unreadable news -> zeros
    (VADER-neutral), never an exception.
    """
    from .sentiment import NEWS_CSV, add_sentiment_to_features, load_news

    for col in NEWS_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    if not NEWS_CSV.exists():
        return df
    try:
        news = load_news(NEWS_CSV)
    except Exception:                      # unreadable / empty -> stay neutral
        return df
    if news.empty or "Symbol" not in news.columns:
        return df
    sym_news = news[news["Symbol"] == symbol]
    if sym_news.empty:
        return df
    try:
        return add_sentiment_to_features(df, sym_news, window_days=5, lag=1)
    except Exception:
        return df


def latest_features(
    history: pd.DataFrame,
    features: list[str] | None = None,
    symbol: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full feature frame, last-row frame ready for the model).

    ``features`` is the model's feature list: when it asks for news-sentiment
    columns they are computed here with the training code path.
    """
    df = history.sort_values("Date").copy()
    if "Symbol" not in df.columns:
        df["Symbol"] = symbol or "UNKNOWN"
    df = add_indicators(df)
    df = engineer_features(df)

    wanted = [f for f in (features or []) if f not in df.columns]
    if wanted:
        df = _add_news_features(df, symbol or str(df["Symbol"].iloc[-1]))

    # ±inf can appear when a rolling window has zero variance (e.g. a flat
    # volume stretch); sklearn rejects them, so treat them like missing data
    frame = df.replace([np.inf, -np.inf], np.nan).dropna(subset=BASE_FEATURES)
    return df, frame.tail(1)


def predict_direction(model, features: list[str], row: pd.DataFrame) -> dict:
    """Direction + confidence from the trained pipeline."""
    proba = float(model.predict_proba(row[features])[0, 1])
    pred = int(proba >= 0.5)
    return {
        "direction": "UP" if pred == 1 else "DOWN",
        "direction_value": pred,
        "probability_up": proba,
        "confidence": round(100.0 * (proba if pred == 1 else 1 - proba), 1),
    }


def latest_news(symbol: str, n: int = 5) -> tuple[pd.DataFrame, dict]:
    """Most recent headlines for a symbol plus a VADER sentiment summary."""
    from .sentiment import POS_THRESHOLD, NEG_THRESHOLD, score_dataframe

    if not NEWS_CSV.exists():
        return pd.DataFrame(), {"label": "N/A", "compound": 0.0, "headlines": 0}

    try:
        news = pd.read_csv(NEWS_CSV, parse_dates=["Date"])
        if news.empty or not {"Symbol", "Date", "Headline"}.issubset(news.columns):
            return pd.DataFrame(), {"label": "N/A", "compound": 0.0, "headlines": 0}
        news = news[news["Symbol"] == symbol].sort_values("Date", ascending=False)
    except Exception:                       # missing / corrupt file -> no news
        return pd.DataFrame(), {"label": "N/A", "compound": 0.0, "headlines": 0}
    if news.empty:
        return pd.DataFrame(), {"label": "N/A", "compound": 0.0, "headlines": 0}

    recent = news.head(30).copy()          # last 30 available headlines
    scored = score_dataframe(recent)
    compound = float(scored["compound"].mean())
    label = ("Positive" if compound >= POS_THRESHOLD else
             "Negative" if compound <= NEG_THRESHOLD else "Neutral")
    return scored.head(n)[["Date", "Headline", "label", "compound"]], {
        "label": label,
        "compound": round(compound, 3),
        "headlines": int(len(recent)),
        "last_date": str(recent["Date"].max().date()),
    }


def volatility_from_history(history: pd.DataFrame, thresholds: dict, window: int = 10) -> dict:
    """Latest annualised volatility level; ``N/A`` when there is too little data."""
    if history is None or len(history) < window + 1:
        return {"level": "N/A", "value": float("nan")}
    close = history.sort_values("Date")["Close"]
    ret = close.pct_change(fill_method=None)
    if ret.notna().sum() < window:
        return {"level": "N/A", "value": float("nan")}
    vol = float(ret.rolling(window, min_periods=window).std().iloc[-1] * np.sqrt(252))
    if not np.isfinite(vol):
        return {"level": "N/A", "value": float("nan")}
    return {"level": classify(vol, thresholds), "value": round(vol * 100, 2)}


def price_snapshot(history: pd.DataFrame) -> dict:
    """Latest bar + one-day change; safe for 0- and 1-row histories."""
    empty = {"date": "n/a", "close": float("nan"), "change": 0.0, "change_pct": 0.0,
             "volume": 0.0, "open": float("nan"), "high": float("nan"),
             "low": float("nan")}
    if history is None or history.empty or "Close" not in history.columns:
        return empty

    df = history.sort_values("Date")
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    change = float(last["Close"]) - float(prev["Close"])
    prev_close = float(prev["Close"])
    pct = change / prev_close * 100.0 if prev_close else 0.0
    vol = last.get("Volume", 0)
    return {
        "date": str(pd.to_datetime(last["Date"]).date()) if "Date" in df.columns else "n/a",
        "close": float(last["Close"]),
        "change": change,
        "change_pct": pct,
        "volume": float(vol) if pd.notna(vol) else 0.0,
        "open": float(last.get("Open", last["Close"])),
        "high": float(last.get("High", last["Close"])),
        "low": float(last.get("Low", last["Close"])),
    }
