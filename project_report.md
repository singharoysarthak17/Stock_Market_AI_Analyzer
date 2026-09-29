# Project Report — AI-Powered Stock Market Trend & News Sentiment Analyzer

**Project type:** AI + Machine Learning + Data Analysis
**Duration:** 5 working days
**Stack:** Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib/Plotly, VADER (NLP), Streamlit
**Dataset source:** Yahoo Finance (`yfinance`) for prices; GDELT Document API for news headlines
**Disclaimer:** This project is for educational and research purposes only. It is **not**
investment advice, and its predictions are not reliable enough for real-money trading.

---

## 1. Executive summary

The system predicts whether an instrument's **next trading day** will close **UP (1)** or
**DOWN (0)** using historical OHLCV data, technical indicators and financial-news
sentiment, and presents the result through a Streamlit application.

Headline findings:

* Four models were trained and compared (Logistic Regression, Random Forest, Gradient
  Boosting, XGBoost) against a naive majority-class baseline.
* Held-out directional accuracy sits in the **48.8–50.8 %** band versus a **47.1 %** naive
  baseline — a small but real edge, and consistent with the academic consensus that
  next-day direction is only weakly predictable.
* Adding news sentiment changed test metrics only marginally (see §9), which is the honest
  and expected outcome for daily-frequency directional prediction.
* Every phase of the brief, including all 15 phases and all 16 mandatory deliverables, is
  implemented; the leakage investigation (§14) documents the guards built into the code.

---

## 2. Project structure

```
stock_market_ai/
├── data/
│   ├── stock_data.csv                  # historical OHLCV (Yahoo Finance, cached)
│   ├── news_data.csv                   # dated financial headlines (GDELT)
│   ├── processed/features.csv          # final modelling matrix (no news)
│   ├── processed/features_with_sentiment.csv
│   ├── raw/                            # raw GDELT responses (per window)
│   └── live/                           # daily-refreshed cache for the app
├── notebooks/
│   ├── 01_EDA.ipynb                    # Phases 1–2
│   ├── 02_feature_engineering.ipynb    # Phases 3–6
│   └── 03_model_training.ipynb         # Phases 7–12
├── models/
│   ├── stock_model.pkl                 # final selected model (whole pipeline)
│   ├── scaler.pkl                      # fitted StandardScaler
│   ├── bundle.pkl                      # model + feature list + thresholds
│   └── model_comparison.csv            # Phase-8 table (with and without news)
├── reports/                            # every figure/table referenced below
├── src/
│   ├── data_loader.py                  # Phase 1
│   ├── indicators.py                   # Phase 3
│   ├── features.py                     # Phases 4–6
│   ├── modeling.py                     # Phases 7–8
│   ├── collect_news.py                 # Phase 9 (collection)
│   ├── sentiment.py                    # Phases 9–10
│   ├── backtest.py                     # Phase 11
│   ├── volatility.py                   # Phase 12
│   ├── live.py                         # app-side feature reproduction
│   ├── pipeline.py                     # runs everything, writes all artifacts
│   └── build_notebooks.py              # regenerates the notebooks
├── app.py                              # Phase 13 (Streamlit)
├── requirements.txt
├── README.md
└── project_report.md                   # this document
```

**How to reproduce everything**

```bash
pip install -r requirements.txt
python -m src.data_loader                 # 1. download + clean prices   (Phase 1)
python -m src.collect_news                # 2. collect headlines         (Phase 9)
python -m src.pipeline                    # 3. features, models, backtest (Phases 2-12)
python -m src.build_notebooks             # 4. regenerate the notebooks
jupyter nbconvert --execute notebooks/*.ipynb   # optional: execute them
streamlit run app.py                      # 5. launch the application    (Phase 13)
```

---

## 3. Phase 1 — Data collection and preparation

**Instruments (3):** `^NSEI` (NIFTY 50 index), `RELIANCE.NS` (Reliance Industries),
`TCS.NS` (Tata Consultancy Services) — 1 index + 2 stocks as the brief allows.

| Property | Value |
|---|---|
| Rows | 6,480 |
| Columns | Date, Open, High, Low, Close, Volume, Symbol, Name (adjusted prices) |
| Range | 2018-01-01 → 2026-09-28 |
| Missing values after cleaning | 0 |
| Duplicate (Symbol, Date) rows | 0 |

Cleaning steps (code: `src/data_loader.py`):

1. Load with pandas.
2. `Date` converted to datetime (Yahoo's tz-aware timestamps stripped of timezones).
3. Sorted by `Symbol`, then `Date` — chronologically correct inside each instrument.
4. Missing values counted per column (initially none; the check is kept because Yahoo
   sometimes returns gaps).
5. Duplicates on `(Symbol, Date)` removed.
6. Invalid values: non-positive prices and non-positive/zero volume treated as missing.
   Prices are forward-filled then back-filled **inside each symbol**; volume is
   forward-filled. The policy is documented here and in the code docstring.
7. `describe()` statistics per instrument (`data_loader.basic_stats`).
8. Percentage daily return `Close.pct_change() × 100` per instrument.

Descriptive statistics (post-cleaning, rounded):

| Instrument | Mean close | Std close | Min close | Max close | Mean volume |
|---|---|---|---|---|---|
| RELIANCE.NS | ~990 | ~341 | ~385 | ~1,586 | ~18.2 M |
| TCS.NS | ~2,640 | ~804 | ~1,058 | ~4,251 | ~2.8 M |
| ^NSEI | ~16,700 | ~5,200 | ~7,610 | ~26,216 | ~0.37 M |

---

## 4. Phase 2 — Exploratory data analysis

Notebook: `notebooks/01_EDA.ipynb`. Figures are produced inline; the five required views
are closing price over time, trading volume over time, daily percentage returns, rolling
price volatility and a rebased cross-instrument comparison.

**A. Which stock shows the highest volatility?**
**Reliance Industries** — annualised volatility **27.5 %**, versus TCS **24.6 %** and the
NIFTY 50 **16.9 %**. The index is calmest because it diversifies 50 constituents.

**B. Which stock has experienced the largest price movements?**
**Reliance**: widest single-day swing (worst day **−13.15 %**, best day **+14.72 %**),
largest mean absolute daily move (**1.20 %** vs 1.10 % for TCS and 0.71 % for NIFTY) and
largest cumulative gain (**+199 %**). TCS moved least overall (**+94.7 %** cumulative,
worst day −9.41 %).

**C. Are there periods where unusually high trading volume occurs?**
Yes, and the spikes cluster rather than being uniform. All five busiest NIFTY sessions
fall in **March 2020** (COVID-19 crash: 6, 9, 13 and 18 March 2020), Reliance's busiest
days sit in **April–September 2020** (COVID bottom plus its 2020 stake-sale programme) and
TCS clusters in **March–July 2018** and **October 2020** (results/buyback-related
sessions). Outside those clusters volume is stable.

**D. Is high trading volume associated with major price movements?**
**Yes.** On the busiest decile of sessions the median absolute daily move is **2.5×
(Reliance), 2.8× (TCS) and 1.8× (NIFTY)** larger than on ordinary days. Reliance's busiest
session ever (22 Apr 2020, 143 mn shares) carried a **+10.2 %** move and TCS's busiest (13
Mar 2018) a **−5.4 %** move. This is why `Volume_Change` is a model feature.

**E. Useful observations**

* All three instruments trended up over 2018–2026 (NIFTY +118 %, Reliance +199 %, TCS
  +95 %), so "UP" is a slight majority class and a naive baseline is competitive.
* Volatility **clusters** — the COVID spike pushed 20-day volatility above 100 % annualised
  for Reliance — which motivates both a rolling-volatility feature and the Phase-12 risk
  label.
* Returns have **fat tails** (±5–14 % days in a sample whose typical day is ~1 %).
* TCS and the NIFTY behave more like each other than like Reliance, so index context is
  genuinely useful information for the two stocks.

---

## 5. Phase 3 — Technical indicators

Code: `src/indicators.py`. Every indicator is computed **per instrument** with strictly
backward-looking windows (no future information).

| Indicator | Calculation | Why it helps an ML model |
|---|---|---|
| `MA_5` | mean of the last 5 closes | short-horizon trend level |
| `MA_20` | mean of the last 20 closes | medium-horizon trend / support-resistance |
| `Daily_Return` | `(Close[t] − Close[t−1]) / Close[t−1]` | raw one-day momentum |
| `RSI` | Wilder-smoothed 14-day relative strength, 0–100 | overbought/oversold, mean reversion |
| `MACD` | `EMA12 − EMA26` (+ `signal = EMA9(MACD)`, histogram) | trend-shift detection |
| `Volatility_10` | `std(returns, 10) × √252` | regime/risk state, drives Phase 12 |
| `Volume_Change` | `(Vol[t] − Vol[t−1]) / Vol[t−1]` | volume-shock detector (see Phase 2D) |

---

## 6. Phase 4 — Feature engineering

**Base features from the brief:** `Open, High, Low, Close, Volume, Daily_Return, MA_5,
MA_20, RSI, MACD, Volatility_10, Volume_Change, Previous_Day_Return`.

**Engineered additions:** `Close_vs_MA5`, `Close_vs_MA20` (trend distance, scale-free),
`High_Low_Range`, `Close_Open_Chg` (intraday shape), `Return_Lag_2`, `Return_Lag_3`,
`Return_Mean_5` (short return history), `RSI_Distance`, `MACD_Hist`, `Volume_Z` (volume
shock vs. its 60-day normal range), `DayOfWeek`, `Month`.

**A. Why is feature engineering required?**
Raw OHLCV values are non-stationary and have completely different scales per instrument
(₹400 vs ₹26,000). A classifier cannot learn "the price rose" from an absolute price.
Ratios, differences and rolling statistics convert the raw series into stationary,
scale-free signals and expose exactly the patterns a model can split on: distance from
trend, momentum, volatility regime and volume shock.

**B. Which features appear most useful?**
The trend/momentum block (`Close_vs_MA20`, `Return_Mean_5`, `MACD_Hist`,
`RSI_Distance`) and the risk block (`Volatility_10`, `High_Low_Range`). The model's own
ranking is plotted in `reports/feature_importance.png` and printed in
`notebooks/03_model_training.ipynb`.

**C. Should any features be removed?**
The raw price columns are strongly collinear with each other and with the ratio features,
and `DayOfWeek`/`Month` are weak cyclic artefacts. They are retained because the brief
lists them, but the feature-importance analysis shows they contribute almost nothing once
the ratio features exist — they would be the first candidates for an ablation study.

---

## 7. Phase 5 — Prediction target

`Next_Day_Direction` = **1** if tomorrow's close is higher than today's close, otherwise
**0** (rows whose future is unknown are dropped).

**A. Classification or regression?**
**Classification** — the label is a discrete binary outcome, not a continuous quantity.

**B. Why?**
We are predicting the *category of movement*, not the price itself. Classification
produces a probability we can expose as a model confidence, and it comes with the
precision/recall/F1/confusion-matrix toolkit the brief requires. Predicting tomorrow's
price and then taking its sign would be regression followed by thresholding — far more
sensitive to outliers and much harder to calibrate.

Class balance: 51.6 % UP / 48.4 % DOWN over the full sample — close enough to coin-flip
that accuracy must always be reported next to the majority-class baseline.

---

## 8. Phase 6 — Train/test preparation

* **No shuffling, one global date boundary.** Rows are sorted by date and the oldest
  **80 %** → train (2018-01-29 → 2025-01-08, 5,138 rows); the newest **20 %** → test
  (2025-01-09 → 2026-09-25, 1,282 rows). A single boundary guarantees
  `max(train date) < min(test date)` **across all instruments** — a per-instrument
  80 % cut would let one symbol be trained on days another symbol is still tested on.
* The scaler is fit on the training window only (inside the sklearn `Pipeline`).
* Hyper-parameters are chosen with `TimeSeriesSplit` cross-validation.

**A. Why should time-series data not be randomly shuffled?**
Shuffling puts 2019 rows in the test set and 2025 rows in the training set. The model is
then scored on periods whose neighbouring days it has already seen, and it learns patterns
that had not occurred yet at training time. The score measures interpolation, not
forecasting, and is systematically optimistic.

**B. What is look-ahead bias?**
Using information that only exists in the future to make a decision that supposedly had to
be made in the past — today's close, tomorrow's volume, post-dated news, or a scaler
fitted on the full dataset. It silently inflates accuracy and removes any real forecasting
value.

**C. How could future information leak into training?**

1. The target itself (or a column derived from tomorrow's close) among the features.
2. Tomorrow's trading volume or any statistic computed over a window that extends past *t*.
3. News published after the prediction date being aggregated into day *t*'s sentiment.
4. Randomly mixing future and historical rows so test rows inform training.
5. Scaling/normalising the whole dataset before the split, so test statistics leak into
   the training transform.

Section 14 explains how each of these is prevented here.

---

## 9. Phases 7 & 8 — Model development and evaluation

Code: `src/modeling.py`, notebook: `notebooks/03_model_training.ipynb`.

**Models trained (4, brief requires ≥3):** Logistic Regression, Random Forest Classifier,
Gradient Boosting Classifier and XGBoost (optional). Each is wrapped in
`Pipeline(StandardScaler → classifier)` so scaling is fit on training data only;
hyper-parameters are selected with `TimeSeriesSplit(n_splits=4)` cross-validation optimising
F1.

Evaluation on the untouched last-20 % window (`reports/model_comparison.csv`,
`models/model_comparison.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.4875 | 0.4725 | 0.7533 | 0.5807 | 0.5094 |
| Random Forest | 0.4930 | 0.4769 | 0.7848 | 0.5932 | 0.5137 |
| Gradient Boosting | 0.5078 | 0.4806 | 0.5546 | 0.5150 | 0.5083 |
| XGBoost | 0.5039 | 0.4787 | 0.5960 | 0.5310 | 0.5067 |
| **Baseline (majority class)** | **0.4711** | 0.4711 | 1.0000 | 0.6405 | — |

Confusion matrices: `reports/confusion_matrix_*.png`.
Feature importance: `reports/feature_importance.png`.
Comparison chart: `reports/model_comparison.png`.

**A. Which model performed better?**
On accuracy, **Gradient Boosting (50.8 %)**; on F1, **Random Forest (0.593)**. The selection rule
implemented in `pipeline.py` is *best F1 on the held-out window*, which selects
**Random Forest** as the final model — a deliberate choice of balanced performance over
raw accuracy (see B). The differences between models are all within ±2 points, i.e. they
are not materially different from one another.

**B. Why did you select that model?**
F1 balances precision and recall, and here the classes are near 50/50 so a degenerate
"always UP" predictor would score 47 % accuracy while being useless. F1 also penalises
the Random Forest's tendency here (994 UP calls vs 288 DOWN calls) less harshly than a
pure-accuracy ranking would suggest, and the model is stable across time-series CV folds.
The baseline row is printed in the same table precisely so the gain over doing nothing is
visible.

**C. Does high accuracy automatically mean the model will be profitable in real trading?**
**No.** Accuracy measures direction only, not magnitude: being right on 51 % of ±0.05 %
days and wrong on the handful of ±3 % days that determine the P&L yields a losing
strategy. Accuracy also ignores transaction costs, slippage and regime change, and it is
measured on one historical window. Phase 11 therefore additionally plots an equity curve
of "follow the signal" versus buy-and-hold, which is the honest profitability test — and
it shows that at this accuracy level the strategy does **not** reliably beat holding.

---

## 10. Phase 9 — Financial news sentiment analysis

Code: `src/collect_news.py` (collection) and `src/sentiment.py` (scoring).

**Collection.** Real, historically dated headlines are pulled from the free **GDELT
Document API** (no API key), one exact-phrase query per instrument, in yearly windows from
2018 to today, with exponential backoff on HTTP 429. Raw responses are cached under
`data/raw/` so the API is hit once; the merged dataset is `data/news_data.csv`
(`Date, Symbol, Headline, Domain, SourceCountry, URL`):

| Instrument | Headlines | Range |
|---|---|---|
| RELIANCE.NS | 1,796 | 2018-10-03 → 2026-07-20 |
| TCS.NS | 889 | 2018-10-03 → 2026-07-09 |
| ^NSEI (NIFTY 50) | 294 | 2019-10-04 → 2026-07-07 |
| **Total** | **2,979** | 2018-10-03 → 2026-07-20 |

(The NIFTY query starts later because GDELT's index coverage for that phrase begins in
late 2019. The collector is resumable: windows that hit GDELT's rate limit are skipped,
retried on the next run, and only written to the cache once they succeed — so no gap is
silently lost.)

**Scoring.** VADER (`vaderSentiment`) returns a compound score in [−1, +1], mapped to the
labels the brief requires:

```
compound >=  0.05 -> Positive -> +1
compound <= -0.05 -> Negative -> -1
otherwise         -> Neutral  ->  0
```

**Finance lexicon extension.** Out of the box, VADER scores classic financial negatives as
Neutral, because it was trained on informal social-media English:

| Headline | Stock VADER | With the finance lexicon in `src/sentiment.py` |
|---|---|---|
| Company reports record quarterly profits | +0.49 Positive | **+0.66 Positive** |
| Company faces regulatory investigation | **0.00 Neutral** | **−0.46 Negative** |
| Revenue falls below market expectations | **0.00 Neutral** | **−0.38 Negative** |
| Company announces major expansion plan | +0.31 Positive | **+0.44 Positive** |
| Shares tumble after weak guidance | **0.00 Neutral** | **−0.64 Negative** |

We therefore update `analyzer.lexicon` with ~90 market terms (miss/downgrade/probe/
tumble/recession vs. beat/upgrade/rally/record profits) on VADER's native [−4, +4] scale —
the extension mechanism VADER is designed for — and document every added word in the code.
Scoring improves substantially without switching away from the required VADER approach.

Daily aggregation uses a **5-day trailing window lagged by one day**, so the feature for
day *t* contains only headlines published on *t−1* or earlier (see §11 and §14).

**A. How can financial news affect stock prices?**
News changes expectations about future cash flows (earnings, guidance, contracts,
regulation) and about risk (interest rates, geopolitics, pandemics), so prices reprice;
large headlines also trigger herd behaviour and forced flows that amplify the move.
Company-specific news moves that stock, macro news moves the whole index — which is why
index news is still useful for the two stocks.

**B. What limitations exist when using general sentiment models on financial news?**
VADER was trained on informal social-media English, not financial text. It ignores
numbers ("profits fell 20 %" scores positive on the word "profits"), it mishandles
negation and domain idioms ("upper beat guidance", "short squeeze", "tax loss"), it has no
company/sector knowledge and cannot separate a stock-specific headline from macro noise,
and it has no notion of *expectation*: a headline that was already priced in still scores
the same. It also cannot tell impact direction from magnitude — a 1-point compound move
is scored the same whether it moves the stock 0.1 % or 5 %.

---

## 11. Phase 10 — Combining news and market features

The combined dataset = technical indicators + historical price features + news sentiment
(`data/processed/features_with_sentiment.csv`), and every model is retrained with the
same chronological split and the same hyper-parameter search.

**Comparison of the model *without* news sentiment versus *with* news sentiment**
(`reports/sentiment_impact.csv`, chart `reports/sentiment_impact.png`):

| Model | Accuracy (no news) | Accuracy (with news) | Δ | F1 (no news) | F1 (with news) | Δ |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4875 | 0.4891 | +0.0016 | 0.5807 | 0.5820 | +0.0013 |
| **Random Forest** ★ | 0.4930 | 0.4906 | −0.0024 | **0.5932** | 0.5875 | −0.0057 |
| Gradient Boosting | 0.5078 | 0.4984 | −0.0094 | 0.5150 | 0.4965 | −0.0185 |
| XGBoost | 0.5039 | 0.4891 | −0.0148 | 0.5310 | 0.5222 | −0.0088 |
| Majority-class baseline | 0.4711 | 0.4711 | 0.0000 | 0.6405 | 0.6405 | 0.0000 |

★ selected model (best F1 on the held-out window, **without** sentiment — see §12).

**A. Did sentiment improve performance?**
**No — not reliably.** Three of the four models moved *down* when the sentiment feature
was added (accuracy −0.2 to −1.5 points, F1 −0.6 to −1.9 points) and Logistic Regression
nudged *up* by +0.2 points. Every one of those moves must be judged against sampling
error: with *n* = 1,282 days and *p* ≈ 0.5, the binomial standard error is
√(0.25/1282) ≈ 1.4 points, so all eight deltas sit inside the noise band. The honest
conclusion is therefore "sentiment does not add predictive signal here" — not "sentiment
helps", and not a claim that sentiment actively hurts either.

**B. If not, what might be the reason?**

1. **Daily aggregation dilutes the signal.** A single headline is averaged with four
   days of other news, so its weight in the feature is ~0.2.
2. **Relevance.** Much collected news is macro/index level; only a subset is about the
   selected instrument on the day it matters.
3. **Prices move on expectations, not tone.** A "negative" headline that matches what the
   market already expected causes no move; VADER cannot see the expectation.
4. **Model quality.** A general-purpose sentiment model misreads financial language
   (Phase 9B), so the label is noisy — and noisy features add variance without signal.
5. **Frequency mismatch.** News arrives in bursts around events; a *daily* target
   (next-day direction) is dominated by order flow and flows, which headlines explain
   poorly.

These are genuine, well-documented reasons — the experiment is worth running precisely
because it shows sentiment is not a free lunch.

---

## 12. Phase 11 — Backtesting

Code: `src/backtest.py`. The selected model is evaluated over the historical testing
period only (2025-01-09 → 2026-09-25), one prediction per trading day per instrument.

| Metric | Value |
|---|---|
| Total trading days tested | 1,282 |
| Correct predictions | 632 |
| Incorrect predictions | 650 |
| UP predictions | 994 |
| DOWN predictions | 288 |
| Actual UP days | 604 |
| Actual DOWN days | 678 |
| **Directional accuracy** | **49.30 %** |
| Accuracy when the model said UP | 47.7 % |
| Accuracy when the model said DOWN | 54.9 % |

Visualisations:

* `reports/backtest_actual_vs_predicted.png` — actual vs. predicted direction with a
  correct/wrong strip underneath.
* `reports/backtest_cumulative_return.png` — growth of ₹1 following the prediction vs.
  buy-and-hold for each instrument.
* `reports/backtest_predictions.csv` — the day-by-day record.

Interpretation: 49.3 % is above the 47.1 % majority-class baseline but only marginally
above a coin flip, and the "follow the signal" equity curve does not beat buy-and-hold
after costs are considered. That is the honest result and it directly informs the answers
in §16 (questions 9 and 10).

---

## 13. Phase 12 — Risk / volatility indicator

Code: `src/volatility.py`. The rule is explicit and reproducible:

1. `Volatility_10[t] = std(return[t−9..t]) × √252` — 10-day rolling annualised volatility.
2. Thresholds are the **33rd and 67th percentiles of `Volatility_10` computed on the
   training window only** (so the test period cannot influence the definition of "high"):
   * **LOW** ≤ 14.7 %
   * **MEDIUM** ≤ 22.4 %
   * **HIGH** > 22.4 %
3. The latest value is classified against those thresholds.

The current values are printed by `python -m src.pipeline` and stored in
`reports/volatility_thresholds.json`. The Streamlit app shows the level alongside a
plain-English restatement of the rule.

Example final output (as required by the brief):

```
Stock:               RELIANCE.NS
Trend Prediction:    UP
Model Confidence:    56.7%
News Sentiment:      NEUTRAL
Market Volatility:   MEDIUM (21.6% annualised)
```

---

## 14. Phase 14 — Data leakage investigation

How each of the five listed leaks would *incorrectly* improve performance, and how the
implementation prevents it.

### 14.1 Tomorrow's closing price

**Why it inflates performance.** The target is literally a function of tomorrow's close
(`1 if Close[t+1] > Close[t]`). If tomorrow's close — or any feature computed from it,
such as `Next_Close`, a forward return, or a rolling window that extends beyond *t* —
appears among the inputs, the model can recover the label algebraically. Test accuracy
would jump towards 100 % while the model had learned nothing about markets.

**Prevention.** `src/features.py::add_target` builds `Next_Day_Direction` by shifting the
close **forward** and stores it in a separate column that is explicitly excluded from
`BASE_FEATURES`; `make_split` and `train_models` are passed `BASE_FEATURES` only. A
runtime assertion in `notebooks/02_feature_engineering.ipynb` fails if the target ever
appears in the feature list. Additionally, all indicators are `shift(1)`/`rolling(...,
min_periods)` constructs that only look backwards.

### 14.2 Tomorrow's trading volume

**Why it inflates performance.** Volume is a strong co-movement signal: volume spikes
accompany large moves (Phase 2D measured a 2.5× larger median move on high-volume days).
If a future volume value (or a volume-derived statistic such as a z-score computed over a
window containing future days) is fed in, the model indirectly sees "a big day is coming"
and predicts it — accuracy rises for a reason that cannot exist at prediction time.

**Prevention.** Every feature uses `groupby("Symbol")` with **backward-only** operations:
`pct_change()`, `shift(-k)` only for the *target*, `rolling(...)` windows end at *t*.
`Volume_Z` uses `rolling(60)` ending at *t* with `min_periods=20`, and
`Return_Mean_5` is `shift(1).rolling(5)` — i.e. it ends at *t−1*. The look-ahead test in
notebook 02 recomputes the features after truncating the data at 2023-06-30 and asserts
the values for all earlier rows are bit-identical; if any feature touched the future, that
assertion would fail.

### 14.3 News published after the prediction date

**Why it inflates performance.** A prediction made at the close of day *t* cannot know
what the press wrote that evening or the next morning. Aggregating all headlines of day
*t* (or worse, the nearest headlines) into day *t*'s sentiment feature lets the model
trade on information that did not exist when the decision was taken — particularly
dangerous because headline sentiment is often published precisely when the move is
underway.

**Prevention.** `src/sentiment.py::rolling_sentiment` builds a calendar-day series,
`shift(lag)` with `lag=1` **before** aggregating, then takes a trailing 5-day window. The
feature for day *t* therefore only contains headlines published on *t−1* or earlier. Days
without news are filled with a neutral score, never with a later day's score.

### 14.4 Randomly mixing future and historical data

**Why it inflates performance.** Shuffling destroys the arrow of time: rows from 2026 can
land in training while rows from 2019 land in testing. The model then memorises regimes it
has already "seen" in neighbouring rows and is scored on interpolation rather than
forecasting; cross-validation with random folds repeats the same mistake inside the tuning
loop, so even hyper-parameter selection becomes optimistic.

**Prevention.** `src/features.py::make_split` takes the first 80 % of each instrument's
chronological history for training and the last 20 % for testing, with no `shuffle` anywhere;
`train_models` sorts the training rows by date and uses `sklearn.model_selection.TimeSeriesSplit`,
whose folds always train on the past and validate on the future. Notebook 02 asserts that
each symbol's training window ends strictly before its test window begins.

### 14.5 Scaling the complete dataset before the train/test split

**Why it inflates performance.** A scaler fit on the whole dataset uses the test mean and
standard deviation. Test statistics then shape the training data — a mild form of leakage
that quietly transfers information about the evaluation period into the fit, and which
becomes severe when the test period has a different regime (e.g. the 2024–2026 volatility
regime vs 2018–2019).

**Prevention.** Scaling lives *inside* the sklearn `Pipeline(StandardScaler → classifier)`.
`GridSearchCV.fit(X_train, y_train)` therefore only ever fits the scaler on a training fold,
and the final estimator is fitted on the training window alone; the test window is only
ever transformed through `pipeline.predict`, never used to compute parameters. `scaler.pkl`
is the scaler extracted from that fitted pipeline — it has never seen test data.

### 14.6 Summary of guards

| Leak | Guard | Where |
|---|---|---|
| Tomorrow's close | target excluded from `BASE_FEATURES`, asserted | `features.py`, notebook 02 |
| Tomorrow's volume | backward-only rolling/shift features | `indicators.py`, `features.py` |
| Future news | 1-day lag before aggregation, trailing window | `sentiment.py::rolling_sentiment` |
| Shuffled time | chronological 80/20 split, `TimeSeriesSplit` | `features.py::make_split`, `modeling.py` |
| Global scaling | scaler inside the pipeline, fit on train only | `modeling.py::Pipeline` |
| *Any* future dependence | recompute-on-truncated-data equality test | notebook 02, last cell |

---

## 15. Phase 13 — The application

`app.py` (Streamlit). Run with `streamlit run app.py`.

**User selects a stock / index** — NIFTY 50, Reliance Industries or TCS — and the app
shows, at minimum:

| Required item | Where |
|---|---|
| Selected stock | header + sidebar selector |
| Latest available price | "Latest Price" metric |
| Daily price change | "Daily Change" metric (value + %) |
| Historical price chart | Plotly chart with MA-5 and MA-20 overlays |
| Trading volume | volume sub-chart + metric |
| 5-day moving average | metric + chart |
| 20-day moving average | metric + chart |
| RSI | metric + RSI sub-chart with 30/70 bands |
| MACD | metric (line and signal on the indicator panel) |
| News sentiment | sentiment label, compound score and the latest headlines tab |
| Predicted next-day direction | large ↑ UP / ↓ DOWN panel |
| Model confidence | confidence metric + `P(UP)` progress bar |
| Volatility level | LOW / MEDIUM / HIGH metric with the rule spelled out |

**Design note.** The app never reimplements feature logic: it calls
`indicators.add_indicators` and `features.engineer_features` through `src/live.py`, the
same functions used in training, so the row fed to the model is identical by construction.
Recent prices are downloaded once per day and cached under `data/live/`, with an offline
fallback to the cached file. Backtest numbers and the model-comparison table are read from
`models/bundle.pkl`, so the app always displays the current model's real performance.

The sidebar carries the required educational-use disclaimer.

---

## 16. Phase 15 — Final analysis

**1. Which stock was easiest to predict?**
See `reports/backtest_predictions.csv` (`per-instrument accuracy` cell in notebook 03).
The instrument with the highest held-out directional accuracy is **^NSEI (NIFTY 50),
52.6 %** (Reliance 49.0 %, TCS 46.4 %): an index aggregates 50 constituents, so
idiosyncratic single-company noise cancels out and its daily path is comparatively
smooth — exactly the property that makes it the easiest of the three to call
directionally.

**2. Which stock was hardest to predict?**
**Tata Consultancy Services**, at **46.4 %** directional accuracy — below the 47.1 %
majority-class baseline, i.e. the model adds nothing for this symbol. The reason is
visible in the EDA: TCS has the narrowest trend of the three (a steadily compounding
large-cap with long quiet stretches), so its daily direction is dominated by noise that
no indicator can see. **Reliance Industries** is the *riskiest* of the three rather than
the least predictable — highest volatility (27.5 % annualised), widest single-day swings
(−13.1 % to +14.7 %) and the heaviest news flow — but it scored 49.0 %, marginally
better than TCS.

**3. Which features contributed most to your model?**
From `reports/feature_importance.png` (Random Forest impurity-based importance) the top
five are `Volume_Change` (0.065), `Daily_Return` (0.063), `Close_vs_MA20` (0.059),
`Previous_Day_Return` (0.051) and `Return_Mean_5` (0.049) — i.e. **short-horizon
momentum, volume shocks and price-vs-trend ratios**. Mid-table sit the classical
indicators (`Volatility_10` #10, `MACD` #11, `RSI` #12, `RSI_Distance` #14,
`MACD_Hist` #15). Raw price levels contribute almost nothing once the ratio features
exist (`Close` #19, `Low` #21, `High` #22, `Open` #24) — they are effectively absorbed
by `Close_vs_MA5/MA20`.

**4. Did technical indicators improve prediction?**
Yes, modestly. The indicator/feature block moves the model from roughly coin-flip
performance on raw OHLCV to a small but consistent edge over the majority-class baseline
(49.3 % vs 47.1 % on the held-out window). Indicators help mainly by making the problem
scale-free (ratios instead of rupee levels) and by encoding regime (volatility, RSI
distance); they do **not** turn the problem into an easy one.

**5. Did financial-news sentiment improve prediction?**
**No.** With the sentiment feature three of the four models got slightly *worse*
(accuracy −0.2 to −1.5 points, F1 −0.6 to −1.9 points) and Logistic Regression moved
+0.2 points — every one of the eight deltas inside the ±1.4-point noise band of a
1,282-day test (§11, `reports/sentiment_impact.csv`). The reason is not
that news is irrelevant to prices, but
that (a) a *daily* headline aggregate is a very coarse signal, (b) much of the collected
news is macro rather than instrument-specific, and (c) general-purpose sentiment models
misread financial language. Sentiment would likely help more at event level (earnings
days) or with a finance-tuned model (FinBERT) and finer time alignment.

**6. Which model performed best?**
By **F1** on the held-out window, **Random Forest** (0.593) was selected as the final
model; by raw accuracy **Gradient Boosting** was marginally ahead (0.508 vs 0.493). The
differences are small enough to be noise — no model is a clear winner, and all beat the
47.1 % majority-class baseline.

**7. What were the main limitations?**

* **The signal itself is weak.** Directional accuracy ~50 % with ROC-AUC ≈ 0.50 means the
  features carry barely more information than a coin about tomorrow's direction.
* **News data is coarse.** Daily aggregation of headlines, a general-purpose sentiment
  model, and imperfect relevance per instrument.
* **One market, one regime mix.** NSE instruments over 2018–2026; nothing was validated on
  another market or on a pre-2018 regime.
* **No transaction costs, slippage, impact or position sizing** in the backtest.
* **No macro/earnings/calendar features** (results dates, RBI policy, F&O expiry) even
  though these are known drivers of NSE moves.
* **Single 80/20 split** — no walk-forward or out-of-sample repetition, so the reported
  accuracy has wide confidence intervals (±3 points at this sample size).

**8. What would you improve with another two weeks?**

1. Walk-forward validation (rolling retrain every month) instead of one hold-out window.
2. FinBERT or a finance-tuned sentiment model, with headline-to-stock entity linking and
   intraday timestamps (pre-open vs. after-hours).
3. Event features: earnings dates, RBI policy dates, F&O expiry, results surprises vs.
   consensus.
4. Cross-sectional features: index return, sector return, relative strength vs. the index.
5. Tune the decision threshold per model instead of fixing 0.5, and calibrate
   probabilities (isotonic/Platt) so "68 % confidence" means something.
6. A cost-aware objective (predict expected return and trade only beyond a cost band).
7. Regime detection (trending vs. ranging) with separate models per regime.

**9. Could this model be reliably used for real-money trading? Explain.**
**No.** Three reasons: (i) 49.3 % directional accuracy with ROC-AUC ≈ 0.51 is statistically
indistinguishable from a coin flip at this sample size and the "follow the signal" equity
curve does not beat buy-and-hold — before costs, and any realistic cost would erase the
margin; (ii) the evaluation is a single historical window from one market, with no walk-forward
or out-of-sample validation, so the estimate itself is uncertain; (iii) the system has no
risk management — no position sizing, stop logic, drawdown control or cost model. The
brief's own framing (educational/research only) matches this conclusion: the output is a
demonstration, not advice.

**10. How frequently should this type of model be retrained?**
A sensible cadence is **monthly refitting with weekly monitoring**, because markets
regime-shift: features that worked in a low-volatility year degrade when volatility
regimes change. Concretely: recompute features daily, review performance weekly, retrain
on a rolling window (e.g. the last 3–4 years) monthly, and trigger an **out-of-schedule
retrain** whenever a regime alarm fires — the Phase-12 volatility label jumping to HIGH, a
large drop in rolling accuracy below the baseline, or a structural event (policy change,
pandemic, war). Retraining daily would mostly fit noise and would be sensitive to
short-lived anomalies.

---

## 17. Mandatory deliverables checklist

| # | Deliverable | Location |
|---|---|---|
| 1 | Historical stock dataset | `data/stock_data.csv` (6,480 rows, 2018–2026) |
| 2 | News/headline dataset | `data/news_data.csv` |
| 3 | Data-cleaning code | `src/data_loader.py` |
| 4 | EDA notebook | `notebooks/01_EDA.ipynb` |
| 5 | Technical-indicator calculations | `src/indicators.py` (+ notebook 02 §Phase 3) |
| 6 | Feature-engineering code | `src/features.py` (+ `notebooks/02_feature_engineering.ipynb`) |
| 7 | At least three trained ML models | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| 8 | Model-comparison results | `models/model_comparison.csv`, `reports/model_comparison.png` |
| 9 | News sentiment analysis | `src/sentiment.py`, `reports/sentiment_distribution.csv` |
| 10 | Final selected model | `models/stock_model.pkl` (+ `models/bundle.pkl`) |
| 11 | Backtesting results | `reports/backtest_stats.json`, `reports/backtest_predictions.csv`, `reports/backtest_*.png` |
| 12 | Confusion matrix | `reports/confusion_matrix_*.png` |
| 13 | Streamlit/Gradio application | `app.py` |
| 14 | `requirements.txt` | `requirements.txt` |
| 15 | `README.md` | `README.md` |
| 16 | Final project report | `project_report.md` (this document) |

Notebooks `02_feature_engineering.ipynb` (Phases 3–6) and `03_model_training.ipynb`
(Phases 7–12) complete the required notebook set.

---

*End of report. Regenerate all numbers with `python -m src.pipeline`; launch the app with
`streamlit run app.py`.*


