"""End-to-end test suite for the Stock Market AI project.

Run with::

    python -m pytest tests -q

The suite deliberately covers three layers:

1. **Static / structural health** - every module imports, no dead placeholders,
   notebooks are executed, requirements are installed, the deliverable audit
   passes.
2. **Methodology correctness** - datasets are clean, features contain no
   look-ahead information, the split is strictly chronological, news sentiment
   is lagged, and the docs quote the *actual* numbers produced by the code.
3. **Failure injection** - missing/corrupt news files, 0/1/5/40-row histories,
   a bundle that needs news features, and the Streamlit app under all of the
   above.  These are the paths that used to raise IndexError / KeyError /
   StreamlitAPIException.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import live  # noqa: E402
from src.features import BASE_FEATURES, TARGET  # noqa: E402

APP_PATH = ROOT / "app.py"
NEWS_PATH = ROOT / "data" / "news_data.csv"
STOCK_PATH = ROOT / "data" / "stock_data.csv"
BUNDLE_PATH = ROOT / "models" / "bundle.pkl"
LIVE_DIR = ROOT / "data" / "live"

SYMBOLS = ["^NSEI", "RELIANCE.NS", "TCS.NS"]


# =========================================================================
# 1. Static / structural health
# =========================================================================
def test_every_src_module_imports():
    import importlib

    modules = sorted(p.stem for p in (ROOT / "src").glob("*.py") if p.stem != "__init__")
    assert len(modules) >= 10, "src/ looks incomplete"
    for name in modules:
        importlib.import_module(f"src.{name}")


def test_requirements_are_installed():
    import importlib.util

    skipped = {"jupyter", "nbformat"}          # notebook-only tooling
    missing = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or "<" in line or ">" in line:
            continue
        pkg = line.split("==")[0].split(">=")[0].strip().replace("-", "_")
        if pkg.lower() in skipped:
            continue
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    assert not missing, f"requirements not importable: {missing}"


def test_no_leftover_placeholders():
    for name in ("README.md", "project_report.md"):
        text = (ROOT / name).read_text(encoding="utf-8").lower()
        for marker in ("todo", "tbd", "filled by", "xxx", "lorem ipsum"):
            assert marker not in text, f"{name} still contains '{marker}'"


def test_notebooks_are_executed_without_errors():
    import json as _json

    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = _json.loads(path.read_text(encoding="utf-8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code, f"{path.name} has no code cells"
        unexecuted = [c for c in code if c.get("execution_count") is None]
        assert not unexecuted, f"{path.name}: {len(unexecuted)} cell(s) never executed"
        errors = [o for c in code for o in c.get("outputs", [])
                  if o.get("output_type") == "error"]
        assert not errors, f"{path.name}: cell raised {errors[0].get('ename')}"


def test_deliverable_audit_passes():
    proc = subprocess.run([sys.executable, "-m", "src.audit"], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL DELIVERABLES PRESENT" in proc.stdout


def test_cli_entry_points_respond():
    for module in ("src.pipeline", "src.collect_news"):
        proc = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, f"{module} --help failed: {proc.stderr}"
        assert "usage:" in proc.stdout.lower()


# =========================================================================
# 2. Data integrity
# =========================================================================
@pytest.fixture(scope="module")
def stock_df() -> pd.DataFrame:
    assert STOCK_PATH.exists(), "data/stock_data.csv missing"
    return pd.read_csv(STOCK_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def full_dataset() -> pd.DataFrame:
    from src.features import build_dataset

    return build_dataset()


def test_stock_dataset_is_clean(stock_df):
    need = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
    assert need.issubset(stock_df.columns)
    assert not stock_df.empty and len(stock_df) > 5000
    assert set(stock_df["Symbol"]) >= set(SYMBOLS)
    assert stock_df[["Date", "Close", "Volume"]].notna().all().all()
    assert not stock_df.duplicated(subset=["Symbol", "Date"]).any()
    assert (stock_df[["Open", "High", "Low", "Close"]] > 0).all().all()
    for _, part in stock_df.groupby("Symbol"):
        assert part["Date"].is_monotonic_increasing, "rows are not sorted by date"
    assert stock_df["Date"].dt.year.min() <= 2018


def test_news_dataset_is_clean():
    assert NEWS_PATH.exists(), "data/news_data.csv missing"
    news = pd.read_csv(NEWS_PATH, parse_dates=["Date"])
    assert {"Date", "Symbol", "Headline", "URL"}.issubset(news.columns)
    assert len(news) > 1000, "expected a full headline dataset"
    assert news["Headline"].notna().all() and (news["Headline"].str.len() > 0).all()
    counts = news["Symbol"].value_counts()
    assert len(counts) == 3, f"expected 3 instruments, got {list(counts.index)}"
    assert (counts >= 100).all(), f"a symbol is under-collected: {counts.to_dict()}"
    assert news["Date"].dt.year.min() >= 2018
    assert news["Date"].max() <= pd.Timestamp.today() + pd.Timedelta(days=1)


def test_processed_feature_matrix(full_dataset):
    assert TARGET in full_dataset.columns
    assert TARGET not in BASE_FEATURES, "target leaked into the feature set"
    assert not full_dataset[list(BASE_FEATURES)].isna().any().any(), "NaN in features"
    assert set(full_dataset[TARGET].unique()) <= {0, 1}
    assert 0.35 < full_dataset[TARGET].mean() < 0.65


# =========================================================================
# 3. Methodology / leakage guards
# =========================================================================
def test_target_is_the_next_day_move(full_dataset):
    for symbol, part in full_dataset.groupby("Symbol"):
        part = part.sort_values("Date")
        future = part["Close"].shift(-1)
        known = future.notna()
        expected = (future > part["Close"]).astype(float)
        assert (part.loc[known, TARGET] == expected[known]).all(), f"{symbol}: bad target"


def test_split_is_strictly_chronological(full_dataset):
    from src.features import make_split

    train, test = make_split(full_dataset)
    assert not train.empty and not test.empty
    assert train["Date"].max() < test["Date"].min(), "train/test dates overlap"
    assert len(train) / len(full_dataset) == pytest.approx(0.8, abs=0.02)
    for symbol in SYMBOLS:
        tr = train[train["Symbol"] == symbol]
        te = test[test["Symbol"] == symbol]
        assert not tr.empty and not te.empty, f"{symbol} missing from a split"
        assert tr["Date"].max() < te["Date"].min()
    # no trading day may appear on both sides of the split
    train_keys = set(zip(train["Date"], train["Symbol"]))
    test_keys = set(zip(test["Date"], test["Symbol"]))
    assert not train_keys & test_keys, "a trading day is in both splits"


def test_features_have_no_look_ahead():
    """Truncating the future must not change a single historical feature."""
    from src.data_loader import prepare
    from src.features import build_dataset

    clean = prepare()
    part = clean[clean["Symbol"] == "RELIANCE.NS"].reset_index(drop=True)
    assert len(part) > 300
    trunc = part.iloc[: len(part) - 150].copy()

    full_f = build_dataset(part.copy())
    trunc_f = build_dataset(trunc.copy())

    shared = sorted(set(full_f["Date"]) & set(trunc_f["Date"]))
    cols = [c for c in BASE_FEATURES if c in full_f.columns]
    a = full_f.set_index("Date").loc[shared, cols].sort_index()
    b = trunc_f.set_index("Date").loc[shared, cols].sort_index()
    assert len(shared) > 200, "truncation test lost most of the sample"
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-10)


def test_news_sentiment_is_lagged():
    from src.sentiment import add_sentiment_to_features

    dates = pd.bdate_range("2024-01-01", periods=10)
    frame = pd.DataFrame({"Date": dates, "Symbol": "TEST"})
    news = pd.DataFrame({
        "Date": ["2024-01-04"],
        "Symbol": ["TEST"],
        "Headline": ["Company beats estimates, announces record profits and a rally"],
    })
    out = add_sentiment_to_features(frame, news, window_days=5, lag=1)

    same_day = out.loc[out["Date"] == "2024-01-04"].iloc[0]
    next_day = out.loc[out["Date"] == "2024-01-05"].iloc[0]
    assert same_day["Headlines"] == 0, "same-day news leaked into the feature"
    assert next_day["Headlines"] == 1, "news not visible the following day"
    assert next_day["News_Sentiment"] >= 0


def test_volatility_thresholds_are_ordered():
    path = ROOT / "reports" / "volatility_thresholds.json"
    th = json.loads(path.read_text(encoding="utf-8"))
    assert th["low_max"] < th["medium_max"] < 1.0
    assert th["low_max"] > 0.0


# =========================================================================
# 4. Trained artefacts
# =========================================================================
@pytest.fixture(scope="module")
def bundle() -> dict:
    assert BUNDLE_PATH.exists(), "models/bundle.pkl missing - run python -m src.pipeline"
    return joblib.load(BUNDLE_PATH)


def test_bundle_is_complete(bundle):
    for key in ("model", "features", "thresholds", "model_name", "backtest_stats",
                "comparison", "news_info"):
        assert key in bundle, f"bundle missing '{key}'"
    assert hasattr(bundle["model"], "predict_proba")
    assert list(bundle["features"])[: len(BASE_FEATURES)] == list(BASE_FEATURES)
    assert bundle["news_info"]["headlines"] >= 1000
    assert bundle["news_info"]["headlines"] == len(pd.read_csv(NEWS_PATH))


def test_bundle_predicts_for_every_symbol(bundle):
    for symbol in SYMBOLS:
        csv = LIVE_DIR / f"{symbol.replace('^', 'IDX_').replace('.', '_')}.csv"
        assert csv.exists(), f"missing cached history for {symbol}"
        hist = pd.read_csv(csv, parse_dates=["Date"])
        feats, last = live.latest_features(hist, features=bundle["features"],
                                           symbol=symbol)
        assert not last.empty, f"{symbol}: no usable feature row"
        missing = [f for f in bundle["features"] if f not in last.columns]
        assert not missing, f"{symbol}: missing features {missing}"
        pred = live.predict_direction(bundle["model"], bundle["features"], last)
        assert pred["direction"] in ("UP", "DOWN")
        assert 0.0 <= pred["probability_up"] <= 1.0
        assert 0.0 <= pred["confidence"] <= 100.0


def test_with_news_bundle_can_build_its_features(bundle):
    """A with-news model must be feedable from live data (latent KeyError)."""
    from src.live import NEWS_FEATURES

    hist = pd.read_csv(LIVE_DIR / "RELIANCE_NS.csv", parse_dates=["Date"])
    wanted = list(bundle["features"]) + NEWS_FEATURES
    feats, last = live.latest_features(hist, features=wanted, symbol="RELIANCE.NS")
    assert not last.empty
    for col in NEWS_FEATURES:
        assert col in last.columns, f"{col} could not be produced for a with-news model"
        assert last[col].notna().all()


def test_backtest_artifacts_are_consistent(bundle):
    stats = json.loads((ROOT / "reports" / "backtest_stats.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(ROOT / "reports" / "backtest_predictions.csv")
    assert len(preds) == stats["Total Predictions"]
    correct = int((preds["Predicted"] == preds["Actual"]).sum())
    assert correct == stats["Correct Predictions"]
    assert stats["Correct Predictions"] + stats["Incorrect Predictions"] == len(preds)
    assert 45.0 <= stats["Directional Accuracy %"] <= 60.0
    assert stats == bundle["backtest_stats"]


def test_model_comparison_covers_both_experiments():
    df = pd.read_csv(ROOT / "models" / "model_comparison.csv")
    assert {"Experiment", "Model", "Accuracy", "F1"}.issubset(df.columns)
    assert set(df["Experiment"]) == {"without news", "with news"}
    assert df.groupby("Experiment")["Model"].count().min() >= 4   # 4 models + baseline


# =========================================================================
# 5. Failure injection - helpers
# =========================================================================
def test_price_snapshot_survives_tiny_histories():
    one = pd.DataFrame({"Date": pd.to_datetime(["2026-01-05"]),
                        "Close": [100.0], "Open": [99.0], "High": [101.0],
                        "Low": [98.0], "Volume": [10]})
    snap = live.price_snapshot(one)
    assert snap["close"] == 100.0 and snap["change"] == 0.0
    empty = live.price_snapshot(pd.DataFrame(columns=["Date", "Close"]))
    assert np.isnan(empty["close"])


def test_volatility_needs_enough_sessions():
    idx = pd.bdate_range("2026-01-01", periods=5)
    short = pd.DataFrame({"Date": idx, "Close": np.linspace(100, 104, 5)})
    out = live.volatility_from_history(short, {"low_max": 0.1, "medium_max": 0.2})
    assert out["level"] == "N/A" and np.isnan(out["value"])


def test_missing_news_file_is_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(live, "NEWS_CSV", tmp_path / "nope.csv")
    frames, summary = live.latest_news("RELIANCE.NS")
    assert frames.empty and summary["label"] == "N/A"


def test_corrupt_news_file_is_not_fatal(monkeypatch, tmp_path):
    bad = tmp_path / "news_data.csv"
    bad.write_text("this,is,not,a,news,file\n1,2,3,4,5\n", encoding="utf-8")
    monkeypatch.setattr(live, "NEWS_CSV", bad)
    frames, summary = live.latest_news("RELIANCE.NS")
    assert frames.empty and summary["label"] == "N/A"


def test_newsless_history_still_builds_features():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2026-01-01", periods=40)
    close = np.linspace(100, 110, 40) + rng.normal(0, 0.4, 40)
    hist = pd.DataFrame({
        "Date": idx, "Symbol": "RELIANCE.NS",
        "Open": close - 0.3, "High": close + 1.0, "Low": close - 1.0,
        "Close": close,
        "Volume": rng.integers(500_000, 2_000_000, 40),
    })
    feats, last = live.latest_features(hist, symbol="RELIANCE.NS")
    assert not last.empty
    assert list(last["Date"])[-1] == idx[-1]


def test_flat_volume_window_does_not_produce_infinities():
    """Zero-variance rolling windows yield NaN/inf, never a sklearn blow-up."""
    idx = pd.bdate_range("2026-01-01", periods=40)
    hist = pd.DataFrame({
        "Date": idx, "Symbol": "RELIANCE.NS",
        "Open": np.full(40, 100.0), "High": np.full(40, 100.0),
        "Low": np.full(40, 100.0), "Close": np.full(40, 100.0),
        "Volume": np.full(40, 1_000),
    })
    _, last = live.latest_features(hist, symbol="RELIANCE.NS")
    if not last.empty:
        assert np.isfinite(last.to_numpy(dtype=float)).all()


def test_pipeline_degrades_when_news_is_unavailable(tmp_path):
    from src.pipeline import _load_news_or_none

    assert _load_news_or_none(lambda: (_ for _ in ()).throw(FileNotFoundError("x"))) is None
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    from src.sentiment import load_news

    assert _load_news_or_none(lambda: load_news(empty)) is None
    assert _load_news_or_none() is not None          # the real dataset works


# =========================================================================
# 6. Failure injection - the Streamlit app
# =========================================================================
def _run_app(at):
    at.run()
    return [str(e.value) for e in at.exception]


def _app_text(at) -> str:
    blobs = []
    for name in ("title", "header", "subheader", "markdown", "caption", "write",
                 "warning", "error", "info", "success", "metric", "code"):
        try:
            elements = at.get(name)
        except Exception:
            continue
        for el in elements:
            value = getattr(el, "value", None)
            if isinstance(value, str):
                blobs.append(value)
    return " ".join(blobs)


@pytest.fixture
def AppTest():
    from streamlit.testing.v1 import AppTest

    def factory():
        return AppTest.from_file(str(APP_PATH), default_timeout=180)

    return factory


def test_app_renders_without_exceptions(AppTest):
    at = AppTest()
    errors = _run_app(at)
    assert not errors, errors
    assert at.title and "STOCK MARKET AI" in at.title[0].value.upper()
    assert len(at.metric) >= 8, "expected price / change / volatility / prediction metrics"
    assert len(at.tabs) == 3
    assert "not investment advice" in _app_text(at).lower()


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_app_works_for_every_instrument(AppTest, symbol):
    at = AppTest()
    at.run()                                # first render fills the element tree
    at.sidebar.selectbox[0].select(symbol)
    errors = _run_app(at)
    assert not errors, errors
    assert len(at.metric) >= 8


def test_app_survives_a_short_history(AppTest, monkeypatch):
    """40 rows: the chart slider's 60-day floor used to raise an API exception."""
    idx = pd.bdate_range("2026-06-01", periods=40)
    base = np.linspace(100, 118, 40)
    short = pd.DataFrame({
        "Date": idx, "Symbol": "RELIANCE.NS", "Open": base - 0.5,
        "High": base + 1.0, "Low": base - 1.0, "Close": base,
        "Volume": np.full(40, 2_000_000),
    })
    monkeypatch.setattr(live, "get_history",
                        lambda *a, **k: short.copy(), raising=True)
    at = AppTest()
    errors = _run_app(at)
    assert not errors, errors


def test_app_survives_a_tiny_history(AppTest, monkeypatch):
    """5 rows: price snapshot / indicator row used to raise IndexError."""
    idx = pd.bdate_range("2026-09-01", periods=5)
    tiny = pd.DataFrame({
        "Date": idx, "Symbol": "TCS.NS", "Open": [1.0, 2.0, 3.0, 4.0, 5.0],
        "High": [2.0, 3.0, 4.0, 5.0, 6.0], "Low": [1.0, 1.0, 2.0, 3.0, 4.0],
        "Close": [1.5, 2.5, 3.5, 4.5, 5.5], "Volume": [10, 20, 30, 40, 50],
    })
    monkeypatch.setattr(live, "get_history",
                        lambda *a, **k: tiny.copy(), raising=True)
    at = AppTest()
    errors = _run_app(at)
    assert not errors, errors


def test_app_handles_a_model_that_needs_news(AppTest, monkeypatch, bundle):
    """Regression: a with-news bundle crashed with KeyError on live rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from src.live import NEWS_FEATURES

    features = list(bundle["features"]) + NEWS_FEATURES
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(150, len(features))), columns=features)
    y = pd.Series(rng.integers(0, 2, 150))
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=400))]).fit(X, y)

    fake = dict(bundle)
    fake.update(model=model, features=features, with_news=True)

    real_load = joblib.load

    def loader(path, *args, **kwargs):
        return fake if str(path).endswith("bundle.pkl") else real_load(path, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", loader)
    at = AppTest()
    errors = _run_app(at)
    assert not errors, errors


def test_app_survives_a_corrupt_bundle(AppTest, monkeypatch):
    """Truncated/incompatible bundle.pkl -> friendly error, not a traceback."""
    import streamlit as st

    def explode(*args, **kwargs):
        raise ValueError("corrupt pickle")

    st.cache_resource.clear()               # a previous test may have cached it
    monkeypatch.setattr(joblib, "load", explode)
    at = AppTest()
    errors = _run_app(at)
    assert not errors, errors
    assert "no trained model" in _app_text(at).lower()


# =========================================================================
# 7. Docs must quote the numbers the code actually produces
# =========================================================================
def test_report_quotes_the_real_split(full_dataset):
    from src.features import make_split

    train, test = make_split(full_dataset)
    report = (ROOT / "project_report.md").read_text(encoding="utf-8")
    for stamp in (str(train["Date"].min().date()), str(train["Date"].max().date()),
                  str(test["Date"].min().date()), str(test["Date"].max().date())):
        assert stamp in report, f"report does not mention the real date {stamp}"


def test_readme_quotes_the_real_backtest(bundle):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stats = bundle["backtest_stats"]
    accuracy = f"{stats['Directional Accuracy %']:.1f}"
    assert accuracy in readme, f"README missing backtest accuracy {accuracy}%"
    assert str(stats["Total Predictions"]).replace(",", "") in readme.replace(",", "")


def test_report_has_no_stale_headline_counts():
    news = pd.read_csv(NEWS_PATH)
    report = (ROOT / "project_report.md").read_text(encoding="utf-8")
    assert f"{len(news):,}" in report, "report does not quote the real headline count"
