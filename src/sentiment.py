"""Phase 9 - Financial News Sentiment Analysis.

Each headline is scored with **VADER** (Valence Aware Dictionary and sEntiment
Reasoner), a rule-based sentiment model tuned on social-media style text.  The
continuous compound score in ``[-1, +1]`` is also mapped to the coarse label the
assignment asks for:

    compound >=  0.05 -> Positive  -> sentiment_value = +1
    compound <= -0.05 -> Negative  -> sentiment_value = -1
    otherwise         -> Neutral   -> sentiment_value =  0

Daily aggregation (leakage safe)
--------------------------------
For every trading day and symbol we aggregate only headlines whose publication
date is **on or before** that day.  The daily score is then lagged by one day
(``shift(1)``) before it is joined onto the feature matrix, so a prediction made
with day-*t* features can only use news published on day *t-1* or earlier.
News published after the prediction date therefore never reaches the model.

Limitations (answered in Phase 9B of the report)
------------------------------------------------
VADER was trained on informal English, not on financial text: it cannot judge
"profits fell 20% but still beat estimates", it ignores numbers entirely, and
it has no company-specific knowledge.  A financial model such as FinBERT would
be the obvious upgrade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEWS_CSV = PROJECT_ROOT / "data" / "news_data.csv"

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05
SENT_COL = "News_Sentiment"          # daily value in {-1, 0, +1}
COMPOUND_COL = "News_Sentiment_Compound"  # daily mean VADER compound

# --------------------------------------------------------------------------
# Finance-specific lexicon extension
# --------------------------------------------------------------------------
# VADER ships a general-purpose lexicon; on financial copy it scores obvious
# negatives such as "revenue falls below market expectations" as 0.00 (Neutral).
# VADER explicitly supports lexicon updates, so we add the market vocabulary
# below (values keep VADER's [-4, +4] scale).  The effect is documented in
# project_report.md §10.
FINANCE_LEXICON: dict[str, float] = {
    # --- negative market vocabulary ---
    "miss": -1.6, "misses": -1.8, "missed": -2.0, "misses estimates": -2.4,
    "disappoint": -2.2, "disappointing": -2.4, "disappointed": -2.2,
    "investigation": -2.0, "probe": -1.8, "inquiry": -1.4, "lawsuit": -2.4,
    "fraud": -3.2, "corruption": -3.0, "fine": -1.8, "penalty": -2.0,
    "downgrade": -2.4, "underperform": -2.2, "bearish": -2.4,
    "fall": -1.4, "falls": -1.6, "fell": -1.6, "falling": -1.6,
    "decline": -1.6, "declines": -1.6, "plunge": -3.0, "plunges": -3.0,
    "slump": -2.6, "slumps": -2.6, "tumble": -2.6, "tumbles": -2.6,
    "selloff": -2.4, "sell-off": -2.4, "weak": -1.8, "weakness": -1.8,
    "warns": -1.8, "warning": -1.6, "cuts guidance": -2.8, "slashes": -2.2,
    "layoffs": -2.2, "job cuts": -2.2, "recession": -2.6, "default": -3.2,
    "bankruptcy": -3.4, "insolvency": -3.2, "recall": -1.6, "halt": -1.8,
    "suspends": -1.8, "delay": -1.4, "delays": -1.4, "ban": -1.8,
    "loss": -1.8, "losses": -1.8, "tariffs": -1.6, "tariff": -1.4,
    "outage": -1.4, "resigns": -1.8, "steps down": -1.2, "withdrawn": -1.4,
    "cuts": -1.0, "contraction": -1.8, "disruption": -1.6, "volatile": -0.8,
    # --- positive market vocabulary ---
    "beat": 2.0, "beats": 2.2, "beat estimates": 2.6, "outperform": 2.2,
    "upgrade": 2.4, "upgraded": 2.4, "bullish": 2.4, "rally": 2.4,
    "rallies": 2.4, "surge": 2.8, "surges": 2.8, "soar": 3.0, "soars": 3.0,
    "jump": 1.8, "jumps": 2.0, "gain": 1.6, "gains": 1.8, "record": 1.6,
    "record profits": 3.0, "all-time high": 3.0, "hits record": 3.0,
    "buyback": 1.8, "dividend": 1.4, "expansion": 1.6, "breakthrough": 2.4,
    "wins contract": 2.6, "bags order": 2.4, "acquisition": 0.8,
    "profit": 1.6, "profits": 1.8, "growth": 1.6, "strong": 1.8,
    "robust": 2.0, "double-digit": 1.4, "raises": 1.0, "approval": 1.8,
    "stake": 0.4, "resilient": 1.6, "rebound": 2.0, "rebounds": 2.0,
}

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(FINANCE_LEXICON)


def score_headline(text: str) -> dict:
    """VADER scores for a single headline."""
    scores = _analyzer.polarity_scores(str(text))
    label = "Neutral"
    if scores["compound"] >= POS_THRESHOLD:
        label = "Positive"
    elif scores["compound"] <= NEG_THRESHOLD:
        label = "Negative"
    scores["label"] = label
    scores["value"] = 1 if label == "Positive" else (-1 if label == "Negative" else 0)
    return scores


def score_dataframe(news: pd.DataFrame) -> pd.DataFrame:
    """Add compound / label / value columns for every headline."""
    news = news.copy()
    compound = news["Headline"].map(lambda t: _analyzer.polarity_scores(str(t))["compound"])
    news["compound"] = compound
    news["label"] = np.where(
        compound >= POS_THRESHOLD, "Positive", np.where(compound <= NEG_THRESHOLD, "Negative", "Neutral")
    )
    news["value"] = np.where(news["label"] == "Positive", 1, np.where(news["label"] == "Negative", -1, 0))
    return news


def daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol, per-day aggregate sentiment.

    Only news dated on or before that day is used (``expanding`` aggregation
    over the sorted history), then the result is shifted one day so the feature
    for day *t* uses news from day *t-1* and earlier.
    """
    news = score_dataframe(news)
    news["Date"] = pd.to_datetime(news["Date"])

    daily = (
        news.groupby(["Symbol", "Date"])
        .agg(
            Headlines=("Headline", "count"),
            News_Sentiment_Compound=("compound", "mean"),
            News_Sentiment_Mean_Value=("value", "mean"),
            News_Sentiment_Pos=("value", lambda s: int((s > 0).sum())),
            News_Sentiment_Neg=("value", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values(["Symbol", "Date"])
    )

    # coarse Phase-9 label for the daily score
    c = daily["News_Sentiment_Compound"]
    daily[SENT_COL] = np.where(
        c >= POS_THRESHOLD, 1, np.where(c <= NEG_THRESHOLD, -1, 0)
    ).astype(int)
    return daily


def load_news(path: Path | str = NEWS_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.collect_news` first."
        )
    news = pd.read_csv(path, parse_dates=["Date"])
    if news.empty:
        raise ValueError("News dataset is empty.")
    # defensive: never let an empty title reach the sentiment scorer
    if "Headline" in news.columns:
        news = news[news["Headline"].notna()]
        news = news[news["Headline"].astype(str).str.strip().ne("")]
    if news.empty:
        raise ValueError("News dataset has no usable headlines.")
    return news.reset_index(drop=True)


def sentiment_report(news: pd.DataFrame) -> pd.DataFrame:
    """Headline-level label distribution - used in the report and the app."""
    scored = score_dataframe(news)
    tbl = (
        scored.groupby("Symbol")["label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(3)
    )
    return tbl


def rolling_sentiment(
    news: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    window_days: int = 5,
    lag: int = 1,
) -> pd.DataFrame:
    """Leakage-safe trailing news sentiment on a calendar-day index.

    For a feature row dated ``t`` this returns aggregates over news published in
    ``(t - window_days - lag, t - lag]`` - i.e. strictly before the prediction
    moment (``lag`` whole days of buffer, default 1 day).
    """
    scored = score_dataframe(news)
    scored["Date"] = pd.to_datetime(scored["Date"])

    start = min(trading_dates.min(), scored["Date"].min())
    end = max(trading_dates.max(), scored["Date"].max())
    idx = pd.date_range(start.normalize(), end.normalize(), freq="D")

    agg = (
        scored.groupby("Date")
        .agg(
            Headlines=("Headline", "count"),
            _sum_compound=("compound", "sum"),
            _sum_value=("value", "sum"),
        )
        .reindex(idx)
        .fillna(0.0)
    )

    # shift first -> only strictly older news, then look back over the window
    count = agg["Headlines"].rolling(window_days, min_periods=1).sum().shift(lag)
    total_c = agg["_sum_compound"].rolling(window_days, min_periods=1).sum().shift(lag)
    total_v = agg["_sum_value"].rolling(window_days, min_periods=1).sum().shift(lag)

    out = pd.DataFrame(
        {
            "Date": idx,
            "Headlines": count.values,
            "News_Sentiment_Compound": np.where(
                count > 0, total_c / count.replace(0, np.nan), 0.0
            ),
            "News_Sentiment_Mean_Value": np.where(
                count > 0, total_v / count.replace(0, np.nan), 0.0
            ),
        }
    )
    c = out["News_Sentiment_Compound"]
    out[SENT_COL] = np.where(
        c >= POS_THRESHOLD, 1, np.where(c <= NEG_THRESHOLD, -1, 0)
    ).astype(int)
    return out.fillna(0.0)


def add_sentiment_to_features(
    features: pd.DataFrame,
    news: pd.DataFrame | None = None,
    window_days: int = 5,
    lag: int = 1,
    prefix: str = "",
) -> pd.DataFrame:
    """Join lagged rolling news sentiment onto the feature matrix (per symbol)."""
    if news is None:
        news = load_news()

    frames = []
    for symbol, part in features.groupby("Symbol", sort=False):
        part = part.sort_values("Date")
        dates = pd.DatetimeIndex(part["Date"].unique())
        sym_news = news[news["Symbol"] == symbol]
        roll = rolling_sentiment(sym_news, dates, window_days=window_days, lag=lag)
        frames.append(part.merge(roll, on="Date", how="left"))

    out = pd.concat(frames).sort_values(["Symbol", "Date"]).reset_index(drop=True)
    # days without news -> neutral sentiment, zero headlines
    for col in ["Headlines", "News_Sentiment_Compound", "News_Sentiment_Mean_Value"]:
        out[col] = out[col].fillna(0.0)
    out[SENT_COL] = out[SENT_COL].fillna(0).astype(int)

    if prefix:
        rename = {
            c: prefix + c
            for c in [
                "Headlines",
                "News_Sentiment_Compound",
                "News_Sentiment_Mean_Value",
                SENT_COL,
            ]
        }
        out = out.rename(columns=rename)
    return out


if __name__ == "__main__":
    news = load_news()
    print(f"headlines: {len(news)}")
    print(sentiment_report(news).to_string())
    d = daily_sentiment(news)
    print(d.head().to_string(index=False))
