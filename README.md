# Stock Market AI Analyzer

**AI-powered stock-market trend & news-sentiment analyzer** — predicts whether an Indian
instrument (NIFTY 50 index, Reliance Industries, TCS) will close **UP** or **DOWN** on the
next trading day from historical OHLCV data, technical indicators and financial-news
sentiment, then serves the result in an interactive **Streamlit** app.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E)
![Phases](https://img.shields.io/badge/brief-15%20phases%20%2F%2016%20deliverables-brightgreen)

> **Educational / research use only — this is not investment advice.** Held-out directional
> accuracy is **49.3 %** versus a **47.1 %** naive baseline, i.e. only marginally better
> than a coin flip. See [Results](#3-results) and [Limitations](#9-limitations).

---

## Contents

1. [What the project does](#1-what-the-project-does)
2. [How it works](#2-how-it-works)
3. [Results](#3-results)
4. [Repository structure](#4-repository-structure)
5. [Brief → deliverables map](#5-brief--deliverables-map)
6. [Getting started](#6-getting-started)
7. [Testing](#7-testing)
8. [Data leakage guards](#8-data-leakage-guards)
9. [Limitations](#9-limitations)
10. [Project report](#10-project-report)

---

## 1. What the project does

* Collects and cleans **8 years of NSE price history** (2018 → 2026) for three instruments.
* Collects **2,979 real, dated financial headlines** from the **GDELT Document API** and
  scores them with a **finance-extended VADER** lexicon.
* Engineers technical indicators (MA-5/20, RSI-14, MACD, volatility, volume change) and
  leakage-safe features, then trains and compares **four classifiers** plus a naive
  baseline: Logistic Regression, Random Forest, Gradient Boosting, XGBoost.
* Backtests the selected model over the untouched hold-out window and classifies market
  volatility as **LOW / MEDIUM / HIGH** from training-window percentiles.
* Ships a Streamlit app that shows the selected instrument, latest price, daily change,
  price chart with moving averages, volume, MA-5, MA-20, RSI, MACD, news sentiment,
  predicted direction, model confidence, volatility level and the educational-use
  disclaimer.

---

## 2. How it works

```
Yahoo Finance (yfinance) ──► data/stock_data.csv ─┐
                                                  ├─► clean ─► indicators ─► features ─► 80/20 chronological split
GDELT Document API ────────► data/news_data.csv ──┘                                                      │
        │                                                                                                 ├─► 4 models + baseline
        └──► VADER (+90 finance terms) ─► lagged 5-day sentiment ─► features_with_sentiment.csv ──────────┘
                                                                                                          │
                                          model selection (best F1 on hold-out) ◄──────────────────────────┘
                                                       │
                       backtest ─► volatility rule ─► models/bundle.pkl ─► Streamlit app (app.py)
```

Every stage is a module under `src/`; `python -m src.pipeline` runs Phases 1–12 and
rewrites every artifact deterministically.

---

## 3. Results

Produced by `python -m src.pipeline` on the committed data files.

**Data**

| Dataset | Rows | Range | Source |
|---|---|---|---|
| `data/stock_data.csv` | 6,480 OHLCV bars (3 instruments) | 2018-01-01 → 2026-09-28 | Yahoo Finance via `yfinance` |
| `data/news_data.csv` | **2,979 headlines** (RELIANCE 1,796 · TCS 889 · NIFTY 294) | 2018-10-03 → 2026-07-20 | GDELT Document API |

**Split** — one global date boundary, never shuffled: train **5,138 rows**
(2018-01-29 → 2025-01-08) · test **1,282 rows** (2025-01-09 → 2026-09-25).
`max(train date) < min(test date)` holds across all instruments.

**Model comparison — held-out test window** (`models/model_comparison.csv`,
`reports/model_comparison.png`; both experiments are in the file):

| Model | Acc. (no news) | F1 (no news) | Acc. (+news) | F1 (+news) |
|---|---|---|---|---|
| Logistic Regression | 48.8 % | 0.581 | 48.9 % | 0.582 |
| **Random Forest** ★ | 49.3 % | **0.593** | 49.1 % | 0.588 |
| Gradient Boosting | **50.8 %** | 0.515 | 49.8 % | 0.497 |
| XGBoost | 50.4 % | 0.531 | 48.9 % | 0.522 |
| Majority-class baseline | 47.1 % | 0.641 | 47.1 % | 0.641 |

★ **selected model** — best F1 on the held-out window, *without* news sentiment.
News sentiment did **not** improve anything here: three of four models moved down and the
fourth moved up by less than the ±1.4-point sampling error of this test
(`reports/sentiment_impact.csv`, `project_report.md` §11).

**News sentiment (Phase 9)** — 2,979 real headlines scored with finance-extended VADER:

| Symbol | Negative | Neutral | Positive |
|---|---|---|---|
| RELIANCE.NS | 19.9 % | 31.9 % | 48.2 % |
| TCS.NS | 24.0 % | 30.6 % | 45.4 % |
| ^NSEI (NIFTY 50) | 15.0 % | 22.1 % | 62.9 % |

**Backtest (Phase 11)** — `reports/backtest_stats.json`:

| Metric | Value |
|---|---|
| Trading days tested | 1,282 |
| Correct / incorrect | 632 / 650 |
| **Directional accuracy** | **49.3 %** |
| Majority-class baseline | 47.1 % |
| Accuracy on UP calls (994) | 47.7 % |
| Accuracy on DOWN calls (288) | 54.9 % |

**Volatility rule (Phase 12)** — 10-day rolling annualised volatility, thresholds from the
training window's 33rd/67th percentiles: **LOW ≤ 14.7 %, MEDIUM ≤ 22.4 %, HIGH > 22.4 %**.

**Per-instrument backtest accuracy** — ^NSEI 52.6 % · RELIANCE.NS 49.0 % · TCS.NS 46.4 %.

---

## 4. Repository structure

```
.
├── app.py                       # Streamlit application (Phase 13)
├── data/
│   ├── stock_data.csv           # cleaned OHLCV dataset (Phase 1 deliverable)
│   ├── news_data.csv            # dated headline dataset (Phase 9 deliverable)
│   ├── processed/               # features.csv, features_with_sentiment.csv
│   ├── raw/                     # raw GDELT responses (one JSON per window)
│   └── live/                    # daily-refreshed price cache used by the app
├── notebooks/
│   ├── 01_EDA.ipynb             # Phases 1–2: cleaning, EDA, 5 charts, answers A–E
│   ├── 02_feature_engineering.ipynb  # Phases 3–6: indicators, features, target, split
│   └── 03_model_training.ipynb   # Phases 7–12: models, sentiment, backtest, risk
├── models/
│   ├── stock_model.pkl          # final selected model (deliverable #10)
│   ├── scaler.pkl               # fitted StandardScaler (deliverable #15)
│   ├── bundle.pkl               # everything the app needs in one file
│   ├── model_comparison.csv     # Phase-8 comparison table
├── reports/                     # figures, backtest stats, thresholds, sentiment tables
├── src/
│   ├── data_loader.py           # Phase 1  download, clean, returns, statistics
│   ├── indicators.py            # Phase 3  MA, RSI, MACD, volatility, volume change
│   ├── features.py              # Phases 4–6  engineered features, target, split
│   ├── modeling.py              # Phases 7–8  pipelines, time-series CV, metrics
│   ├── collect_news.py          # Phase 9  GDELT collection (backoff, resumable)
│   ├── sentiment.py             # Phases 9–10  VADER + finance lexicon, lagged join
│   ├── backtest.py              # Phase 11  counts, accuracy, charts
│   ├── volatility.py            # Phase 12  LOW/MEDIUM/HIGH rule
│   ├── live.py                  # app-side feature builder (single source of truth)
│   ├── pipeline.py              # runs Phases 1–12 and writes every artifact
│   ├── build_notebooks.py       # regenerates the three notebooks
│   └── audit.py                 # verifies the 16 mandatory deliverables
├── tests/
│   └── test_e2e.py              # end-to-end + failure-injection suite
├── requirements.txt             # runtime dependencies
├── requirements-dev.txt         # test dependencies (pytest)
├── README.md
└── project_report.md            # final report: Phases 1–15, all written answers
```

---

## 5. Brief → deliverables map

| # | Required deliverable | Where |
|---|---|---|
| 1 | Historical stock dataset | `data/stock_data.csv` |
| 2 | News / headline dataset | `data/news_data.csv` |
| 3 | Data-cleaning code | `src/data_loader.py` |
| 4 | EDA notebook (5 charts + answers A–E) | `notebooks/01_EDA.ipynb` |
| 5 | Technical-indicator calculations | `src/indicators.py` |
| 6 | Feature-engineering code | `src/features.py` |
| 7 | At least three trained ML models | `models/stock_model.pkl`, `models/scaler.pkl`, `models/bundle.pkl` |
| 8 | Model-comparison results | `models/model_comparison.csv`, `reports/model_comparison.*` |
| 9 | News sentiment analysis | `src/sentiment.py`, `reports/sentiment_distribution.csv` |
| 10 | Final selected model | `models/stock_model.pkl` |
| 11 | Backtesting results | `reports/backtest_stats.json`, `reports/backtest_predictions.csv`, `reports/backtest_*.png` |
| 12 | Confusion matrix | `reports/confusion_matrix_*.png` |
| 13 | Streamlit / Gradio application | `app.py` |
| 14 | `requirements.txt` | `requirements.txt` |
| 15 | `README.md` | `README.md` |
| 16 | Final project report | `project_report.md` |

Phases 1–15 map to the same table above in the report (§7–§16); run
`python -m src.audit` to re-verify all 16 at any time.

---

## 6. Getting started

**Requirements:** Python 3.10+ (developed on 3.13), internet only for the initial data
download — everything else runs offline from the committed CSVs.

```bash
# 1. install
pip install -r requirements.txt

# 2. (optional) refresh the price dataset                 -> data/stock_data.csv
python -m src.data_loader

# 3. (optional) refresh the headline dataset              -> data/news_data.csv
python -m src.collect_news                    # whole history
python -m src.collect_news --symbol TCS.NS    # one instrument only

# 4. train, evaluate, backtest, export artifacts          -> models/, reports/
python -m src.pipeline

# 5. launch the application                               (Phase 13)
streamlit run app.py
```

Steps 2 and 3 can be skipped once the CSVs exist; `src.pipeline` re-uses them and prints
a cleaning report. Useful entry points:

| Command | Purpose |
|---|---|
| `python -m src.pipeline` | run Phases 1–12, rewrite every artifact |
| `python -m src.pipeline --no-news` | same, but skip Phases 9–10 (works without `news_data.csv`) |
| `python -m src.collect_news --force` | ignore the GDELT window cache and refetch |
| `python -m src.data_loader` | re-download and re-clean the price data |
| `python -m src.audit` | check all 16 mandatory deliverables exist |
| `python -m src.build_notebooks` | regenerate the three notebooks from source |
| `jupyter nbconvert --to notebook --execute notebooks/*.ipynb --inplace` | re-execute them |

**Running the app**

```bash
streamlit run app.py
```

Pick an instrument in the sidebar, use **↻ Refresh data** to re-download today's bar, and
read the chart / indicators / sentiment / prediction / volatility panels plus the
model-comparison and backtest tabs. Recent prices are cached once a day under
`data/live/` with an offline fallback, and every number shown (accuracy, F1, backtest
counts) is read from `models/bundle.pkl`, so the app always displays the current model's
real performance.

---

## 7. Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

`tests/test_e2e.py` covers three layers:

| Layer | What is checked |
|---|---|
| **Structural health** | every `src` module imports, requirements are importable, no leftover draft markers, all notebooks executed without errors, `python -m src.audit` passes, CLI entry points respond |
| **Methodology** | dataset cleanliness, no target leakage, features unchanged when the future is truncated, strictly chronological split, lagged news sentiment, backtest/model artifacts mutually consistent, **report and README quote the real numbers the code produces** |
| **Failure injection** | missing/corrupt news files, 0/1/5/40-row price histories, a bundle that requires news features, a corrupt `bundle.pkl`, and the Streamlit app under all of the above (via `streamlit.testing.v1.AppTest` for every instrument) |

---

## 8. Data leakage guards

* Features at day *t* use only data from *t* and earlier — every rolling/diff operation
  looks backwards, and the target is created by shifting the close **forward**; it is
  never part of the feature set.
* Notebook 02 contains a *recompute-on-truncated-data* assertion: deleting the most recent
  150 rows must leave every earlier feature value bit-identical.
* News sentiment is aggregated over a trailing 5-day window and **lagged by one day**, so
  headlines published on or after the prediction date can never be seen.
* **One global chronological date boundary** for the 80/20 split — never shuffled, and no
  training row is dated after a test row, across instruments as well as within them.
* Hyper-parameters come from `TimeSeriesSplit` cross-validation (never random CV), and
  the scaler lives **inside** the sklearn `Pipeline`, so it is fit on training data only.
* Volatility thresholds are percentiles of the **training window only**.

`project_report.md` §14 discusses all five leakage channels in detail.

---

## 9. Limitations

* **The signal itself is weak.** Held-out accuracy ≈ 49–51 % with ROC-AUC ≈ 0.51 — a few
  points above the naive baseline — and a strategy following the signals does not beat
  buy-and-hold before costs.
* **News sentiment adds nothing usable here.** It is a coarse daily aggregate of
  general-purpose sentiment: with it, three of four models moved down and the fourth was
  flat (all deltas inside noise). VADER's lexicon is extended with ~90 market terms
  (`src/sentiment.py`) because stock VADER scores headlines such as *"Revenue falls below
  market expectations"* as Neutral — see `project_report.md` §10 for the before/after
  table. A finance-tuned model (FinBERT) with event-level alignment is the obvious
  upgrade.
* **One market, one split.** NSE instruments over 2018–2026 with a single 80/20 split; no
  walk-forward validation, no transaction costs, no position sizing or risk management.
* No macro/earnings/calendar features (results dates, RBI policy, F&O expiry).

---

## 10. Project report

The full write-up — Phase-by-Phase description, all written answers (Phases 4A–E,
5A–C, 6A–C, 8A–C, 9A–B, 10A–B, the five leakage channels of Phase 14 and the ten
Phase-15 questions), methodology reasoning and reproducibility notes — is in
**[`project_report.md`](project_report.md)**.

**Data credits:** price data from [Yahoo Finance](https://finance.yahoo.com) (via
`yfinance`), headlines from the [GDELT Document API](https://www.gdeltproject.org),
sentiment from [VADER](https://github.com/cjhutto/vaderSentiment) (extended locally).

*Educational / research use only. Not investment advice.*
