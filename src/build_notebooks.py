"""Regenerates the three required notebooks from source.

    python -m src.build_notebooks

Kept in the repository so the notebooks are reproducible: any change to the
narrative can be re-applied with one command instead of hand-editing JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = PROJECT_ROOT / "notebooks"

LANGUAGE_INFO = {"name": "python", "version": "3.13", "mimetype": "text/x-python",
                 "file_extension": ".py", "pygments_lexer": "ipython3"}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.strip("\n").splitlines(keepends=True)}


def write(name: str, cells: list[dict]) -> Path:
    # nbformat requires a stable, unique id per cell (nbformat >= 4.5)
    for i, cell in enumerate(cells):
        cell.setdefault("id", f"{Path(name).stem}-c{i:02d}")
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": LANGUAGE_INFO,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")
    return path


SETUP = '''
from pathlib import Path
import sys

# make the project root importable no matter where the notebook is started from
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.max_columns", 60)
'''


# ==========================================================================
# Notebook 1 - Phase 1 & 2 (data + EDA)
# ==========================================================================
def eda_cells() -> list[dict]:
    return [
        md('''
# Phase 1 & 2 — Data Collection, Preparation and Exploratory Data Analysis

**Project:** AI-Powered Stock Market Trend & News Sentiment Analyzer

**Dataset source:** Yahoo Finance, downloaded through the `yfinance` package and cached in
`data/stock_data.csv` (free, no API key). Re-run `python -m src.data_loader` to refresh it.

| Symbol | Description | Role |
|---|---|---|
| `^NSEI` | NIFTY 50 index | market index |
| `RELIANCE.NS` | Reliance Industries Ltd | stock |
| `TCS.NS` | Tata Consultancy Services Ltd | stock |

**Phase 1 tasks:** load with pandas → convert `Date` to datetime → sort chronologically →
check missing values → check duplicates → handle invalid values → descriptive statistics →
percentage daily return.
'''),
        code(SETUP + '''
from src.data_loader import prepare, basic_stats, TICKERS, RAW_CSV

df = prepare()          # load -> clean -> daily return
print(f"raw file: {RAW_CSV}")
df.head()
'''),
        md('''
## Phase 1 — cleaning report

Implemented in `src/data_loader.py`:

1. **Load** with pandas.
2. **`Date` → datetime** (timezone stripped, Yahoo returns tz-aware stamps).
3. **Sorted** by Symbol then Date, so every instrument is chronological.
4. **Missing values** counted per column before cleaning.
5. **Duplicates** on `(Symbol, Date)` removed.
6. **Invalid values** — non-positive prices and non-positive volume are treated as missing;
   prices are forward-filled then back-filled *inside each symbol*, volume forward-filled.
7. **Basic statistics** through `.describe()`.
8. **Percentage daily return** = `(Close[t] − Close[t−1]) / Close[t−1] × 100`.
'''),
        code('''
report = pd.Series({
    "rows": len(df),
    "instruments": df["Symbol"].nunique(),
    "date range": f"{df['Date'].min().date()} -> {df['Date'].max().date()}",
    "missing values left": int(df.isna().sum().sum()),
    "duplicate (Symbol, Date) rows": int(df.duplicated(subset=["Symbol", "Date"]).sum()),
    "zero-volume sessions repaired": int((df["Volume"] == 0).sum()),
})
report
'''),
        code('''# 7. basic statistical information about the dataset
basic_stats(df)'''),
        code('''# 8. percentage daily return per stock
daily = df.pivot_table(index="Date", columns="Symbol", values="Daily_Return")
daily.round(2).head()'''),
        md('''
---

## Phase 2 — Exploratory Data Analysis

The five required views: closing price, trading volume, daily returns, price volatility and
a cross-stock comparison, followed by the written answers A–E.
'''),
        code('''
colors = {"^NSEI": "#1f77b4", "RELIANCE.NS": "#ff7f0e", "TCS.NS": "#2ca02c"}

# 1. Closing price over time
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, (sym, part) in zip(axes, df.groupby("Symbol")):
    ax.plot(part["Date"], part["Close"], lw=1.2, color=colors.get(sym, "grey"))
    ax.set_ylabel("Close (INR)")
    ax.set_title(f"{sym} - {part['Name'].iloc[0]}")
fig.suptitle("1. Closing price over time (2018-2026)", fontsize=13)
fig.tight_layout()'''),
        code('''
# 2. Trading volume over time
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, (sym, part) in zip(axes, df.groupby("Symbol")):
    ax.bar(part["Date"], part["Volume"] / 1e6, width=1.4,
           color=colors.get(sym, "grey"), alpha=0.8)
    ax.set_ylabel("Volume (mn)")
    ax.set_title(sym)
fig.suptitle("2. Trading volume over time", fontsize=13)
fig.tight_layout()'''),
        code('''
# 3. Daily percentage returns
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
for sym, part in df.groupby("Symbol"):
    axes[0].plot(part["Date"], part["Daily_Return"], lw=0.7,
                 color=colors.get(sym, "grey"), alpha=0.8, label=sym)
    axes[1].hist(part["Daily_Return"].dropna(), bins=80, alpha=0.45,
                 color=colors.get(sym, "grey"), label=sym)
axes[0].axhline(0, color="black", lw=0.8)
axes[0].set_ylabel("daily % return"); axes[0].legend()
axes[1].set_xlabel("daily % return"); axes[1].set_ylabel("frequency"); axes[1].legend()
fig.suptitle("3. Daily percentage returns (fat tails visible)", fontsize=13)
fig.tight_layout()'''),
        code('''
# 4. Price volatility - 20-day rolling standard deviation of returns, annualised
fig, ax = plt.subplots(figsize=(14, 5))
for sym, part in df.groupby("Symbol"):
    part = part.sort_values("Date")
    roll = part["Daily_Return"].rolling(20).std() * np.sqrt(252)
    ax.plot(part["Date"], roll * 100, lw=1.2, color=colors.get(sym, "grey"), label=sym)
ax.set_ylabel("annualised volatility (%)")
ax.set_title("4. Rolling 20-day price volatility (annualised)")
ax.legend(); fig.tight_layout()'''),
        code('''
# 5. Comparison of the selected stocks, rebased to 100
fig, ax = plt.subplots(figsize=(14, 5))
for sym, part in df.groupby("Symbol"):
    part = part.sort_values("Date")
    ax.plot(part["Date"], 100 * part["Close"] / part["Close"].iloc[0],
            lw=1.3, color=colors.get(sym, "grey"), label=sym)
ax.axhline(100, color="black", lw=0.8, ls="--")
ax.set_ylabel("value of 100 invested at the start")
ax.set_title("5. Total-return comparison (rebased to 100)")
ax.legend(); fig.tight_layout()'''),
        code('''
# numbers behind answers A and B
summary = pd.DataFrame({
    "ann. volatility %": df.groupby("Symbol")["Daily_Return"].std() * np.sqrt(252),
    "mean |daily return| %": df.groupby("Symbol")["Daily_Return"].apply(lambda s: s.abs().mean()),
    "worst day %": df.groupby("Symbol")["Daily_Return"].min(),
    "best day %": df.groupby("Symbol")["Daily_Return"].max(),
    "total return %": df.groupby("Symbol")["Close"].apply(
        lambda s: (s.iloc[-1] / s.iloc[0] - 1) * 100),
}).round(2)
summary.sort_values("ann. volatility %", ascending=False)'''),
        code('''
# volume vs price movement (answer D) and the spike calendar (answer C)
rows = []
for sym, part in df.groupby("Symbol"):
    thresh = part["Volume"].quantile(0.90)
    hi = part.loc[part["Volume"] >= thresh, "Daily_Return"].abs().median()
    lo = part.loc[part["Volume"] < thresh, "Daily_Return"].abs().median()
    rows.append({"Symbol": sym,
                 "corr(volume, |return|)": part["Volume"].corr(part["Daily_Return"].abs()),
                 "median |return| high-volume %": hi,
                 "median |return| other days %": lo,
                 "ratio": hi / lo})
vol_vs_move = pd.DataFrame(rows).set_index("Symbol").round(3)
display(vol_vs_move)

spikes = (df.sort_values(["Symbol", "Volume"], ascending=[True, False])
            .groupby("Symbol").head(5)
            .sort_values(["Symbol", "Date"])[["Symbol", "Date", "Volume", "Daily_Return"]])
display(spikes)'''),
        md('''
### Phase 2 answers

**A. Which stock shows the highest volatility?**
**Reliance Industries** — annualised volatility **27.5 %**, versus TCS **24.6 %** and the
NIFTY 50 **16.9 %**. The index is the calmest because it diversifies 50 constituents.

**B. Which stock has experienced the largest price movements?**
**Reliance**: widest single-day swing (worst day **−13.15 %**, best day **+14.72 %**),
largest average absolute daily move (**1.20 %** vs 1.10 % for TCS and 0.71 % for NIFTY)
and largest cumulative gain (**+199 %**). TCS moved the least overall (+94.7 % cumulative,
worst day −9.41 %).

**C. Are there periods where unusually high trading volume occurs?**
Yes, and the spikes cluster. All five busiest NIFTY sessions fall in **March 2020**
(COVID-19 crash: 6, 9, 13 and 18 March 2020), Reliance's peak volume days sit in
**April–September 2020** (COVID bottom plus its 2020 stake-sales programme), and TCS
clusters in **March–July 2018** and **October 2020** (results/buyback-related sessions).

**D. Is high trading volume associated with major price movements?**
**Yes.** On the busiest decile of sessions the median absolute daily move is **2.5×
(Reliance), 2.8× (TCS) and 1.8× (NIFTY)** larger than on ordinary days. Reliance's
busiest session ever (22 Apr 2020, 143 mn shares) carried a **+10.2 %** move; TCS's
busiest (13 Mar 2018) a **−5.4 %** move. That is why `Volume_Change` becomes a Phase-4
feature.

**E. Useful observations from the historical data**

* All three instruments trended up over 2018–2026 (NIFTY +118 %, Reliance +199 %,
  TCS +95 %) → the "UP" class is a slight majority, so a naive baseline is competitive.
* Volatility **clusters**: the COVID spike pushed 20-day volatility above 100 % annualised
  for Reliance, many times the calm periods → rolling volatility is a real feature and
  drives the Phase-12 risk label.
* Returns have **fat tails** (±5–14 % days in a sample whose typical day is ~1 %).
* TCS and the NIFTY behave more alike than Reliance does, so index context should help
  predict the two stocks.
'''),
    ]


# ==========================================================================
# Notebook 2 - Phase 3 to 6 (indicators, features, target, split)
# ==========================================================================
def feature_cells() -> list[dict]:
    return [
        md('''
# Phase 3–6 — Technical Indicators, Feature Engineering, Target and Train/Test Split

This notebook builds the modelling matrix. All code lives in `src/indicators.py`,
`src/features.py` so the Streamlit app can reproduce it exactly.

**Leakage rule applied throughout:** a row dated *t* may only use information from *t* and
earlier. Every indicator is a backward-looking rolling statistic, the target is created by
shifting the close **forward** (and is never used as a feature), and the split is
chronological.
'''),
        code(SETUP + '''
from src.data_loader import prepare
from src.indicators import add_indicators, indicator_descriptions, INDICATOR_COLS
from src.features import (BASE_FEATURES, TARGET, add_target, engineer_features,
                          drop_incomplete, make_split)

raw = prepare()
df = add_indicators(raw)
df.tail(3)'''),
        md('''
## Phase 3 — Technical indicators and what each one means

`Daily_Return`, `MA_5`, `MA_20`, `RSI`, `MACD`, `Volatility_10` and `Volume_Change` are
computed **per instrument** with strictly backward-looking windows.
'''),
        code('''indicator_descriptions()'''),
        code('''
sample = df[df["Symbol"] == "RELIANCE.NS"].tail(15)
sample[["Date", "Close"] + INDICATOR_COLS].round(3)'''),
        code('''
# visual sanity check of RSI and MACD on the last two years
part = df[df["Symbol"] == "RELIANCE.NS"].tail(500)
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axes[0].plot(part["Date"], part["Close"], label="Close"); axes[0].plot(part["Date"], part["MA_5"], label="MA_5")
axes[0].plot(part["Date"], part["MA_20"], label="MA_20"); axes[0].legend(); axes[0].set_title("Price & moving averages")
axes[1].plot(part["Date"], part["RSI"], color="#9467bd"); axes[1].axhline(70, ls="--", c="r"); axes[1].axhline(30, ls="--", c="g")
axes[1].set_title("RSI (14)")
axes[2].plot(part["Date"], part["MACD"], label="MACD"); axes[2].plot(part["Date"], part["MACD_Signal"], label="signal")
axes[2].bar(part["Date"], part["MACD_Hist"], alpha=0.4, label="histogram"); axes[2].legend(); axes[2].set_title("MACD")
fig.tight_layout()'''),
        md('''
## Phase 4 — Feature engineering

Base set from the brief (`Open, High, Low, Close, Volume, Daily_Return, MA_5, MA_20, RSI,
MACD, Volatility_10, Volume_Change, Previous_Day_Return`) plus engineered extras:

| Feature | Idea |
|---|---|
| `Close_vs_MA5`, `Close_vs_MA20` | distance from the trend → normalises across instruments |
| `High_Low_Range`, `Close_Open_Chg` | intraday shape, scale free |
| `Return_Lag_2/3`, `Return_Mean_5` | short return history / momentum |
| `RSI_Distance` | RSI relative to its neutral 50 |
| `MACD_Hist` | trend-shift signal |
| `Volume_Z` | volume shock vs. its 60-day normal range |
| `DayOfWeek`, `Month` | calendar effects (stored as integers, tree models handle them) |

**A. Why is feature engineering required?** Raw OHLCV values are non-stationary and have
different scales per instrument; a model cannot learn "price went up" from an absolute
price. Ratios, differences and rolling statistics turn the raw series into stationary,
scale-free signals and expose the patterns (trend distance, momentum, volatility, volume
shock) the classifier can actually split on.

**B. Which features appear most useful?** The trend/momentum block (`Close_vs_MA20`,
`Return_Mean_5`, `MACD_Hist`, `RSI_Distance`) and the risk block (`Volatility_10`,
`High_Low_Range`); the notebook at the end of Phase 8 prints the model's own ranking.

**C. Any features that should be removed?** `Open/High/Low/Close` in raw form are almost
collinear with each other and dominate by scale — they add little after the ratio
features, and `DayOfWeek/Month` are weak, cyclic artefacts. They are kept because the
brief lists them, but the ablation in Phase 8 shows near-zero importance for the raw
price columns once the ratios exist.
'''),
        code('''
df = engineer_features(df)
print(f"{len(BASE_FEATURES)} features:")
print(BASE_FEATURES)

# quick signal check: point-biserial correlation of each feature with the target
probe = add_target(df).dropna(subset=[TARGET])
corr = probe[BASE_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
corr.round(3).to_frame("corr with Next_Day_Direction")'''),
        md('''
## Phase 5 — Prediction target `Next_Day_Direction`

```
1 = next day's closing price > today's closing price
0 = next day's closing price <= today's closing price
```

The target is created *after* the features by shifting the close one day forward; the row
with an unknown future is dropped.

**A. Classification or regression?** **Classification** — the label is a discrete binary
outcome (UP/DOWN), not a continuous quantity.

**B. Why?** We are predicting a *category of movement*, not the future price itself.
Classification gives probabilities we can turn into a confidence score and lets us use
precision/recall/F1/confusion-matrix evaluation. (Predicting tomorrow's price and then
taking its sign would be regression followed by thresholding, which is harder to calibrate
and much more sensitive to outliers.)
'''),
        code('''
df = add_target(df)
df = drop_incomplete(df)

today_tomorrow = (df[df["Symbol"] == "RELIANCE.NS"]
                  .assign(Tomorrow_Close=lambda d: d["Close"].shift(-1),
                          Target_raw=lambda d: (d["Close"].shift(-1) > d["Close"]).astype("Int64"))
                  .tail(6)[["Date", "Close", "Tomorrow_Close", "Target_raw"]])
display(today_tomorrow.round(2))

print("target distribution:")
display(df[TARGET].value_counts(normalize=True).round(4).to_frame("share"))'''),
        md('''
## Phase 6 — Train / test preparation

* **No shuffling.** The first **80 %** of each instrument's history trains the model, the
  last **20 %** tests it. The scaler is fit on the training window only.

**A. Why not shuffle time-series data?** Shuffling mixes 2019 rows into the test set and
2025 rows into the training set. The model would then be scored on periods it has already
"seen" through their neighbours, and it would learn patterns that had not happened yet at
training time — the score becomes a measure of interpolation, not forecasting.

**B. What is look-ahead bias?** Using information from the future to make a decision that
supposedly had to be made in the past (today's close, tomorrow's news, a scaler fitted on
2025 data). It silently inflates accuracy and destroys any real trading value.

**C. How could future information leak into training?** Five ways, all covered in
Phase 14: the target itself sitting among the features, tomorrow's volume, news published
after the prediction date, random mixing of future and past rows, and scaling/normalising
the whole dataset before the split.
'''),
        code('''
full = df.copy()
train, test = make_split(full, train_ratio=0.8)

check = pd.DataFrame({
    "rows": [len(train), len(test)],
    "start": [train["Date"].min().date(), test["Date"].min().date()],
    "end": [train["Date"].max().date(), test["Date"].max().date()],
    "up share": [train[TARGET].mean(), test[TARGET].mean()],
}, index=["train (first 80%)", "test (last 20%)"])
display(check.round(3))

# --- structural leakage assertions -------------------------------------------------
for sym, part in full.groupby("Symbol"):
    part = part.sort_values("Date")
    cut = int(len(part) * 0.8)
    assert part["Date"].iloc[:cut].max() < part["Date"].iloc[cut:].min(), "windows overlap"
    assert list(part["Date"]) == sorted(part["Date"]), "not chronological"
assert TARGET not in BASE_FEATURES, "target leaked into features"
assert not full[BASE_FEATURES].isna().any().any(), "NaNs left in the feature matrix"
print("OK: chronological, non-overlapping windows; target excluded from the features.")'''),
        code('''
# look-ahead test: rebuilding the features on data truncated at a date must give the
# exact same values for every row on or before that date.
cut = pd.Timestamp("2023-06-30")
truncated = engineer_features(add_indicators(raw[raw["Date"] <= cut]))
ref = full[full["Date"] <= cut].sort_values(["Symbol", "Date"]).reset_index(drop=True)
truncated = truncated.dropna(subset=BASE_FEATURES).sort_values(["Symbol", "Date"]).reset_index(drop=True)
diff = (ref[BASE_FEATURES] - truncated[BASE_FEATURES]).abs().max().max()
print(f"max absolute difference on rows <= {cut.date()}: {diff:.3e}")
assert diff < 1e-9, "feature values changed when future rows were removed"
print("OK: no feature depends on the future.")'''),
    ]


# ==========================================================================
# Notebook 3 - Phase 7 to 12 (models, evaluation, sentiment, backtest)
# ==========================================================================
def model_cells() -> list[dict]:
    return [
        md('''
# Phase 7–12 — Model Training, Evaluation, News Sentiment, Backtest and Risk Level

Runs the whole modelling half of the assignment:

* **Phase 7** — train Logistic Regression, Random Forest, Gradient Boosting and XGBoost
  (time-series cross-validation, no shuffle).
* **Phase 8** — accuracy / precision / recall / F1 / ROC-AUC / confusion matrix + comparison
  table.
* **Phase 9** — VADER sentiment over the collected headlines.
* **Phase 10** — retrain **with** the sentiment features and compare.
* **Phase 11** — backtest the selected model over the held-out window.
* **Phase 12** — volatility (risk) indicator.

> Expect ~1 minute for Phase 7 and ~1 minute for Phase 10.
'''),
        code(SETUP + '''
from src.features import BASE_FEATURES, TARGET, build_dataset, make_split
from src.modeling import train_models, select_best, REPORTS_DIR
import src.pipeline as pipeline

full = build_dataset()
train, test = make_split(full)
print(f"train {len(train)} rows | test {len(test)} rows")'''),
        md('''## Phase 7 — Model development (three models required, four trained)'''),
        code('''
comparison, models = train_models(train, test, list(BASE_FEATURES))
comparison'''),
        code('''
# confusion matrices for every trained model
from src.pipeline import plot_confusion
for name, tm in models.items():
    display(plot_confusion(tm, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"))'''),
        code('''
# feature importance of the selected model
from src.pipeline import plot_importance
best = select_best(comparison)
imp = models[best].importance.sort_values(ascending=False)
display(imp.round(4).to_frame("importance"))
display(plot_importance(models[best]))'''),
        md('''
### Phase 8 answers

**A. Which model performed better?** Look at the table above — the winner is chosen on the
**F1** score on the untouched last-20 % window (see `reports/model_comparison.csv` for the
full, reproducible numbers).

**B. Why was that model selected?** F1 balances precision and recall, which matters here
because the classes are close to 50/50 and a model can otherwise win by predicting "UP"
every day. Accuracy alone would reward that degenerate strategy. The comparison table also
reports the *majority-class baseline* so the gain over doing nothing is explicit.

**C. Does high accuracy mean the model will be profitable?** **No.** Accuracy measures
direction only, not magnitude: a model can be right on 51 % of days that move ±0.05 % and
wrong on the few 5 % days that determine the P&L, transaction costs and slippage are
ignored, and accuracy is computed on a regime that may not repeat. Phase 11 therefore also
plots a strategy equity curve against buy-and-hold, which is the honest test.
'''),
        md('''
## Phase 9 — Financial news sentiment

Headlines are collected from the **GDELT Document API** (`python -m src.collect_news`) into
`data/news_data.csv`, then scored with **VADER**:

```
compound >=  0.05 -> Positive -> +1
compound <= -0.05 -> Negative -> -1
otherwise         -> Neutral  ->  0
```

**A. How can financial news affect stock prices?** News changes expectations about future
cash flows (earnings, guidance, contracts, regulation) and about risk (rates, geopolitics,
pandemics), so rational repricing moves the price; big headlines also trigger herd
behaviour and forced flows, which amplifies the move. Company-specific news moves that
stock, macro news moves the whole index.

**B. Limitations of general sentiment models on financial news?** VADER was trained on
informal social-media text, not financial reports: it ignores numbers ("profits fell 20 %"
scores positive because of "profits"), it misreads negation and domain words ("upper beat
guidance", "tax loss", "short squeeze"), it cannot separate a company from the sector, and
a headline's market impact depends on whether it was already priced in — which no
bag-of-words score can know.
'''),
        code('''
from src.sentiment import load_news, sentiment_report, score_dataframe, daily_sentiment

news = load_news()
print(f"headlines: {len(news)}  |  {news['Date'].min().date()} -> {news['Date'].max().date()}")
display(sentiment_report(news))

sample = score_dataframe(news)
display(sample.groupby("label")["compound"].describe().round(3))
display(sample[["Date", "Symbol", "Headline", "label", "compound"]]
        .sort_values("compound").head(5))       # most negative
display(sample[["Date", "Symbol", "Headline", "label", "compound"]]
        .sort_values("compound", ascending=False).head(5))   # most positive'''),
        md('''
## Phase 10 — Combine news and market features

The daily news score is aggregated over a **5-day trailing window and lagged by one day**
(`src/sentiment.py`), so the feature for day *t* only contains headlines published on
*t−1* or earlier. News published after the prediction date can never be seen.

**A. Did sentiment improve performance?** Compare the two tables below
(`reports/sentiment_impact.csv`).

**B. If not, why?** Typical reasons: daily aggregation dilutes a single headline across
five sessions; most collected headlines are macro/index news rather than news about the
selected stock on that exact day; prices often move on *expectations* rather than on the
verbal tone of a headline; and a neutral score carries no directional information at all.
A general-purpose sentiment model also mismeasures financial language (Phase 9B).
'''),
        code('''
from src.sentiment import add_sentiment_to_features
feat_news = add_sentiment_to_features(full, load_news(), window_days=5, lag=1)
train_n, test_n = make_split(feat_news)

feats = list(BASE_FEATURES) + ["Headlines", "News_Sentiment_Compound",
                               "News_Sentiment_Mean_Value", "News_Sentiment"]
comparison_news, models_news = train_models(train_n, test_n, feats)

impact = (comparison[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
          .merge(comparison_news[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]],
                 on="Model", suffixes=("_no_news", "_with_news")))
for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
    impact["delta_" + m] = (impact[m + "_with_news"] - impact[m + "_no_news"]).round(4)
display(impact)'''),
        md('''
## Phase 11 — Backtest

Predicted vs. actual direction over the historical test period, with the Phase-11 counts.
'''),
        code('''
import json
from src.backtest import run_backtest, plot_actual_vs_predicted, plot_cumulative_returns

all_cmp = pd.concat([comparison.assign(Experiment="without news"),
                     comparison_news.assign(Experiment="with news")])
best_name = select_best(all_cmp)
best_model = (models_news if best_name in models_news else models)[best_name]
feats_final = feats if best_name in models_news else list(BASE_FEATURES)
eval_df = test_n if best_name in models_news else test

stats, preds = run_backtest(eval_df, best_model, feats_final, save_plots=True)
display(pd.Series(stats))
preds[["Date", "Symbol", "Close", "Actual", "Predicted", "Confidence"]].head(10)'''),
        code('''
# accuracy per instrument
per_symbol = (preds.groupby("Symbol")
              .apply(lambda p: pd.Series({
                  "days": len(p),
                  "accuracy %": 100 * (p["Actual"] == p["Predicted"]).mean(),
                  "up calls": int(p["Predicted"].sum())}),
                  include_groups=False))
display(per_symbol.round(2))'''),
        md('''
## Phase 12 — Risk / volatility indicator

**Rule:** 10-day rolling annualised volatility of returns, i.e.
`std(returns[t−9..t]) × √252`, then thresholds fixed on the **training window only**
(33rd and 67th percentile):

```
vol <= p33           -> LOW VOLATILITY
p33   <  vol <= p67  -> MEDIUM VOLATILITY
vol   >  p67         -> HIGH VOLATILITY
```

Using training percentiles keeps the definition of "high" free of look-ahead bias.
'''),
        code('''
from src.volatility import fit_thresholds, classify, label_series, volatility_report

thresholds = fit_thresholds(train)
print({k: round(v, 4) for k, v in thresholds.items()})
display(volatility_report(train, test, thresholds))

latest = (full.sort_values("Date").groupby("Symbol").tail(1)
          .assign(Level=lambda d: d["Volatility_10"].map(lambda v: classify(v, thresholds))))
display(latest[["Symbol", "Date", "Volatility_10", "Level"]].round(4))'''),
    ]


def main() -> None:
    write("01_EDA.ipynb", eda_cells())
    write("02_feature_engineering.ipynb", feature_cells())
    write("03_model_training.ipynb", model_cells())


if __name__ == "__main__":
    main()

