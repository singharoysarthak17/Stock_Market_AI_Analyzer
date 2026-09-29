"""Phase 13 - Stock Market AI Analyzer (Streamlit app).

Run with:

    streamlit run app.py

The app reuses the *same* ``src`` functions that produced the training data, so
the features it feeds to the model are identical to the ones the model learned.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.live import (  # noqa: E402
    DISPLAY,
    get_history,
    latest_features,
    latest_news,
    predict_direction,
    price_snapshot,
    volatility_from_history,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
BUNDLE_PATH = MODELS_DIR / "bundle.pkl"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

st.set_page_config(page_title="Stock Market AI Analyzer", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model ...")
def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        return {}
    try:
        return joblib.load(BUNDLE_PATH)
    except Exception:                # truncated / incompatible artefact
        return {}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def history(symbol: str, refresh: bool = False):
    return get_history(symbol, force_refresh=refresh)


# --------------------------------------------------------------------------
def main() -> None:
    bundle = load_bundle()
    if not bundle:
        st.error(
            "No trained model found. Run `python -m src.pipeline` first "
            "(creates models/bundle.pkl)."
        )
        st.stop()

    model = bundle["model"]
    features = bundle["features"]
    thresholds = bundle["thresholds"]
    model_name = bundle.get("model_name", "model")
    with_news = bundle.get("with_news", False)

    # ---------------- sidebar ---------------------------------------------
    st.sidebar.title("📈 Stock Market AI")
    st.sidebar.caption("AI-Powered Stock Market Trend & News Sentiment Analyzer")
    symbol = st.sidebar.selectbox(
        "Select a stock / index",
        options=list(DISPLAY.keys()),
        format_func=lambda s: f"{DISPLAY[s]}  ({s})",
    )
    refresh = st.sidebar.button("↻ Refresh data")
    st.sidebar.divider()
    st.sidebar.subheader("Model")
    st.sidebar.write(f"**{model_name}**")
    st.sidebar.write(f"News sentiment features: **{'yes' if with_news else 'no'}**")
    stats = bundle.get("backtest_stats", {})
    if stats:
        st.sidebar.write(f"Held-out accuracy: **{stats.get('Directional Accuracy %', 0):.1f}%**")
    st.sidebar.caption(
        f"Train window: up to {bundle.get('train_end', 'n/a')}  |  "
        f"Test window: from {bundle.get('test_start', 'n/a')}"
    )
    st.sidebar.divider()
    st.sidebar.warning(
        "Educational / research use only. This is not investment advice and "
        "predictions are not reliable enough for real-money trading."
    )

    # ---------------- data -------------------------------------------------
    try:
        hist = history(symbol, refresh=refresh)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if hist is None or len(hist) < 2:
        st.error("Not enough price history for this instrument yet - press "
                 "'↻ Refresh data' or try another symbol.")
        st.stop()

    feats, last_row = latest_features(hist, features=features, symbol=symbol)
    if last_row.empty:
        st.error("History is too short to build the indicator window "
                 "(~60 trading days needed).")
        st.stop()
    missing = [f for f in features if f not in last_row.columns]
    if missing:
        st.error("Model expects features that could not be built: "
                 f"{', '.join(missing)}. Re-run `python -m src.pipeline`.")
        st.stop()

    snap = price_snapshot(hist)
    pred = predict_direction(model, features, last_row)
    vol = volatility_from_history(hist, thresholds)
    headlines, senti = latest_news(symbol)

    # ---------------- header ----------------------------------------------
    st.title("STOCK MARKET AI ANALYZER")
    st.caption(f"{DISPLAY[symbol]} ({symbol})  •  data as of {snap['date']}")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Market snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest Price", f"₹{snap['close']:,.2f}" if symbol != "^NSEI"
                  else f"{snap['close']:,.2f}")
        c2.metric("Daily Change", f"{snap['change']:+,.2f}", f"{snap['change_pct']:+.2f}%")
        c3.metric("Volume", f"{snap['volume']:,.0f}" if snap["volume"] > 0 else "n/a")
        c4.metric("Volatility Level", vol["level"],
                  f"{vol['value']:.1f}% ann." if math.isfinite(vol["value"])
                  else "not enough sessions yet")

        st.subheader("Next-day ML prediction")
        pc1, pc2, pc3 = st.columns([2, 2, 3])
        icon = "↑" if pred["direction"] == "UP" else "↓"
        color = "green" if pred["direction"] == "UP" else "red"
        pc1.markdown(
            f"<h1 style='color:{color};'>{icon} {pred['direction']}</h1>",
            unsafe_allow_html=True,
        )
        pc2.metric("Model Confidence", f"{pred['confidence']:.1f}%")
        pc3.caption(
            f"Probability of UP = {100 * pred['probability_up']:.1f}%  •  "
            f"last row fed to the model: {snap['date']}"
        )
        st.progress(pred["probability_up"], text=f"P(UP) = {100 * pred['probability_up']:.1f}%")

    with right:
        st.subheader("Indicators")
        r = last_row.iloc[0]
        i1, i2 = st.columns(2)
        i1.metric("RSI (14)", f"{r['RSI']:.1f}")
        i2.metric("MACD", f"{r['MACD']:,.2f}")
        i3, i4 = st.columns(2)
        i3.metric("5-Day MA", f"{r['MA_5']:,.2f}")
        i4.metric("20-Day MA", f"{r['MA_20']:,.2f}")
        st.caption(
            f"Close vs MA-5: {100 * (r['Close_vs_MA5']):+.2f}%  |  "
            f"Close vs MA-20: {100 * (r['Close_vs_MA20']):+.2f}%"
        )

        st.subheader("News sentiment")
        s_label = senti["label"]
        emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}.get(s_label, "⚪")
        s1, s2 = st.columns(2)
        s1.metric("Sentiment", f"{emoji} {s_label.upper()}")
        s2.metric("Compound score", f"{senti['compound']:+.3f}")
        st.caption(
            f"based on {senti['headlines']} recent headlines"
            + (f" (latest {senti['last_date']})" if senti.get("last_date") else "")
        )

    # ---------------- chart ------------------------------------------------
    st.subheader("Historical price with moving averages")
    # bounds are clamped: a freshly downloaded / short history must not make
    # Streamlit raise "value is below the minimum bound"
    span = max(len(hist), 2)
    lo, hi = (60, span) if span > 60 else (max(1, span // 2), span)
    if hi <= lo:
        lo, hi = 1, max(span, 2)
    step = 30 if hi - lo > 60 else 1
    default = max(lo, min(365, hi))
    window = st.slider("Chart window (days)", lo, hi, default, step=step)
    plot = feats.tail(window)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03, subplot_titles=("Price & moving averages", "Volume", "RSI"),
    )
    fig.add_trace(go.Scatter(x=plot["Date"], y=plot["Close"], name="Close",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot["Date"], y=plot["MA_5"], name="MA 5",
                             line=dict(color="#ff7f0e", width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot["Date"], y=plot["MA_20"], name="MA 20",
                             line=dict(color="#2ca02c", width=1.3)), row=1, col=1)
    fig.add_trace(go.Bar(x=plot["Date"], y=plot["Volume"], name="Volume",
                         marker_color="#9ecae1"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot["Date"], y=plot["RSI"], name="RSI",
                             line=dict(color="#9467bd", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.update_layout(height=760, legend=dict(orientation="h", y=1.04),
                      margin=dict(t=40, b=20))
    st.plotly_chart(fig, width="stretch")

    # ---------------- tabs -------------------------------------------------
    tab_news, tab_model, tab_about = st.tabs(
        ["📰 News headlines", "📊 Model & backtest", "ℹ️ About / methodology"]
    )

    with tab_news:
        if headlines.empty:
            st.info("No headlines available for this symbol yet.")
        else:
            for _, row in headlines.iterrows():
                emoji = {"Positive": "🟢", "Negative": "🔴"}.get(row["label"], "⚪")
                st.write(
                    f"{emoji} **{pd.to_datetime(row['Date']).date()}** — {row['Headline']}  "
                    f"`{row['compound']:+.3f}`"
                )

    with tab_model:
        mc1, mc2 = st.columns([3, 2])
        comp = bundle.get("comparison")
        if isinstance(comp, pd.DataFrame) and not comp.empty:
            with mc1:
                st.subheader("Model comparison (held-out test window)")
                show = comp.drop(columns=[c for c in ["BestParams", "Experiment"] if c in comp.columns])
                st.dataframe(show, width="stretch", hide_index=True)
        with mc2:
            st.subheader("Backtest")
            for k in ["Total Predictions", "Correct Predictions", "Incorrect Predictions",
                      "UP Predictions", "DOWN Predictions"]:
                if k in stats:
                    st.write(f"**{k}:** {stats[k]}")
            st.write(f"**Directional accuracy:** {stats.get('Directional Accuracy %', 0):.1f}%")
            cm_img = sorted(REPORTS_DIR.glob("confusion_matrix_*.png"))
            if cm_img:
                st.image(cm_img[-1], caption="Confusion matrix (test window)")
        st.caption(
            "Volatility rule: 10-day rolling annualised volatility of returns, "
            f"thresholds fixed on the training window - LOW <= {100*thresholds['low_max']:.1f}%, "
            f"MEDIUM <= {100*thresholds['medium_max']:.1f}%, HIGH above that."
        )

    with tab_about:
        st.markdown(
            """
**What this app does**

1. Downloads recent OHLCV data for the selected NSE instrument (Yahoo Finance, cached daily).
2. Recomputes the *training* indicators (MA-5, MA-20, RSI-14, MACD, 10-day volatility,
   volume change) plus the engineered features with the same `src` code used to train.
3. Scores the latest financial headlines with VADER.
4. Asks the trained classifier whether tomorrow's close will be higher than today's.

**Honest limitations**

* Next-day direction is close to a coin flip: the held-out accuracy of these models is
  only marginally above the naive baseline, and financial markets are weakly predictable.
* Sentiment comes from a curated headline dataset; general-purpose sentiment models are
  not trained on financial language.
* Nothing here is investment advice.
"""
        )


if __name__ == "__main__":
    main()
