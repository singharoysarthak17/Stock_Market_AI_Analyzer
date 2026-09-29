"""End-to-end pipeline: runs Phases 1-12 and writes every artifact.

    python -m src.pipeline            # full run
    python -m src.pipeline --no-news  # skip Phase 9/10 (no news file)

Artifacts produced
------------------
    data/processed/features.csv        feature matrix (clean + indicators + target)
    models/stock_model.pkl             final selected model (whole pipeline)
    models/scaler.pkl                  the fitted StandardScaler (deliverable #15)
    models/bundle.pkl                  everything the Streamlit app needs
    models/model_comparison.csv        Phase-8 comparison table (with/without news)
    reports/confusion_matrix_*.png     Phase-8 confusion matrices
    reports/model_comparison.png       Phase-8 bar chart
    reports/feature_importance.png     Phase-15 Q3 evidence
    reports/sentiment_impact.csv       Phase-10 with/without news table
    reports/backtest_predictions.csv   Phase-11 day-by-day results
    reports/backtest_stats.json        Phase-11 accuracy summary
    reports/backtest_*.png             Phase-11 charts
    reports/volatility_thresholds.json Phase-12 rule
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import BASE_FEATURES, build_dataset, make_split
from .modeling import REPORTS_DIR, MODELS_DIR, save_model, train_models
from .volatility import fit_thresholds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------------------------------
def build_frames():
    full = build_dataset()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    full.to_csv(PROCESSED / "features.csv", index=False)
    train, test = make_split(full)
    return full, train, test


# --------------------------------------------------------------------------
def plot_comparison(comparison: pd.DataFrame, save_as: str = "model_comparison.png") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = comparison[~comparison["Model"].str.startswith("Baseline")].copy()
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    ax = df.set_index("Model")[metrics].plot(kind="bar", figsize=(12, 5), rot=15)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison - Phase 8")
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
    fig = ax.get_figure()
    fig.tight_layout()
    path = REPORTS_DIR / save_as
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_confusion(tm, save_as: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = tm.metrics["ConfusionMatrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred DOWN", "Pred UP"])
    ax.set_yticks([0, 1], ["Actual DOWN", "Actual UP"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax.set_title(f"{tm.name}\naccuracy={tm.metrics['Accuracy']:.3f}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    path = REPORTS_DIR / save_as
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_importance(tm, save_as: str = "feature_importance.png") -> Path | None:
    imp = tm.importance
    if imp is None:
        return None
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top = imp.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    top.iloc[::-1].plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_title(f"Top 15 features - {tm.name} (|coefficient| / importance)")
    ax.set_xlabel("importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = REPORTS_DIR / save_as
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_sentiment_impact(comparison_no: pd.DataFrame, comparison_yes: pd.DataFrame) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    a = comparison_no[~comparison_no.Model.str.startswith("Baseline")].set_index("Model")
    b = comparison_yes[~comparison_yes.Model.str.startswith("Baseline")].set_index("Model")
    diff = (b[["Accuracy", "F1", "ROC-AUC"]] - a[["Accuracy", "F1", "ROC-AUC"]]) * 100
    ax = diff.plot(kind="bar", figsize=(11, 5), rot=15)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("change (percentage points)")
    ax.set_title("Phase 10 - effect of adding news sentiment")
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%+.2f", fontsize=8, padding=2)
    fig = ax.get_figure()
    fig.tight_layout()
    path = REPORTS_DIR / "sentiment_impact.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def _load_news_or_none(loader=None) -> pd.DataFrame | None:
    """News dataset -> DataFrame, or ``None`` when it is unavailable/unreadable.

    A missing or empty ``data/news_data.csv`` must not kill the whole run: the
    without-news experiment (Phases 1-8 and 11-12) stays fully reproducible, so
    the pipeline degrades to its ``--no-news`` behaviour instead of crashing.
    """
    from .sentiment import load_news

    fn = loader or load_news
    try:
        return fn()
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        print(f"!! News dataset unavailable ({exc})\n"
              f"!! Continuing WITHOUT news sentiment (--no-news behaviour).")
        return None


# --------------------------------------------------------------------------
def run(use_news: bool = True, verbose: bool = True) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASES 1-6 : data -> indicators -> features -> target -> split")
    print("=" * 72)
    full, train, test = build_frames()
    print(f"train {len(train)} rows  ({train.Date.min().date()} -> {train.Date.max().date()})")
    print(f"test  {len(test)} rows  ({test.Date.min().date()} -> {test.Date.max().date()})")

    results: dict = {"train_rows": len(train), "test_rows": len(test)}

    # ---------------- Phase 7/8 : without news -----------------------------
    print("\n" + "=" * 72)
    print("PHASES 7-8 : models WITHOUT news sentiment")
    print("=" * 72)
    comp_no, models_no = train_models(train, test, list(BASE_FEATURES), verbose=verbose)

    # ---------------- Phase 9/10 : with news -------------------------------
    comp_yes, models_yes, train_n, test_n, news_info = None, {}, train, test, {}
    if use_news:
        from .sentiment import add_sentiment_to_features, load_news, sentiment_report

        news = _load_news_or_none(load_news)
        if news is None:
            use_news = False

    if use_news:
        print("\n" + "=" * 72)
        print("PHASE 9 : news sentiment")
        print("=" * 72)
        s_report = sentiment_report(news)
        print(s_report.to_string())
        s_report.to_csv(REPORTS_DIR / "sentiment_distribution.csv")

        feat_news = add_sentiment_to_features(full, news, window_days=5, lag=1)
        feat_news.to_csv(PROCESSED / "features_with_sentiment.csv", index=False)
        train_n, test_n = make_split(feat_news)

        print("\n" + "=" * 72)
        print("PHASE 10 : models WITH news sentiment")
        print("=" * 72)
        feats = list(BASE_FEATURES) + [
            "Headlines", "News_Sentiment_Compound", "News_Sentiment_Mean_Value", "News_Sentiment"
        ]
        comp_yes, models_yes = train_models(train_n, test_n, feats, verbose=verbose)
        comp_yes.to_csv(REPORTS_DIR / "model_comparison_with_news.csv", index=False)

        impact = comp_no[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].merge(
            comp_yes[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]],
            on="Model", suffixes=("_no_news", "_with_news"),
        )
        for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            impact[f"delta_{m}"] = (impact[f"{m}_with_news"] - impact[f"{m}_no_news"]).round(4)
        impact.to_csv(REPORTS_DIR / "sentiment_impact.csv", index=False)
        plot_sentiment_impact(comp_no, comp_yes)
        print("\nPhase 10 - with vs without sentiment (delta):")
        print(impact[["Model", "Accuracy_no_news", "Accuracy_with_news", "delta_Accuracy",
                      "F1_no_news", "F1_with_news", "delta_F1"]].to_string(index=False))
        news_info = {"headlines": int(len(news)),
                     "date_min": str(pd.to_datetime(news["Date"]).min().date()),
                     "date_max": str(pd.to_datetime(news["Date"]).max().date())}

    # ---------------- pick the final model ---------------------------------
    comp_all = comp_no if comp_yes is None else pd.concat(
        [comp_no.assign(Experiment="without news"),
         comp_yes.assign(Experiment="with news")], ignore_index=True)
    comp_all.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    comp_all.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    if comp_yes is None:
        pool_no = comp_no[~comp_no.Model.str.startswith("Baseline")]
        best_name = str(pool_no.loc[pool_no["F1"].idxmax(), "Model"])
        with_news = False
    else:
        # reset the index: both tables start at 0, duplicated labels would make
        # .loc[idxmax()] return two rows instead of one
        c1 = comp_no[~comp_no.Model.str.startswith("Baseline")].assign(s="no").reset_index(drop=True)
        c2 = comp_yes[~comp_yes.Model.str.startswith("Baseline")].assign(s="yes").reset_index(drop=True)
        pool = pd.concat([c1, c2], ignore_index=True)
        best_row = pool.loc[pool["F1"].idxmax()]
        best_name, with_news = str(best_row["Model"]), bool(best_row["s"] == "yes")

    final_models = models_yes if with_news else models_no
    final_model = final_models[best_name]
    features = list(BASE_FEATURES) + ([] if not with_news else [
        "Headlines", "News_Sentiment_Compound", "News_Sentiment_Mean_Value", "News_Sentiment"
    ])
    eval_df = test_n if with_news else test

    print("\n" + "=" * 72)
    print(f"SELECTED MODEL: {best_name}  (with news: {with_news})")
    print("=" * 72)

    # ---------------- Phase 11 : backtest ----------------------------------
    from .backtest import run_backtest

    stats, pred_df = run_backtest(eval_df, final_model, features, save_plots=True)
    print("\nPhase 11 backtest:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    (REPORTS_DIR / "backtest_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    pred_df.to_csv(REPORTS_DIR / "backtest_predictions.csv", index=False)

    # Confusion matrix for *every* model (Phase-8 deliverable), written here so
    # reports/ can never hold figures left over from an older run.  Notebook 03
    # writes the same four files with the same names, so both entry points agree.
    for name, tm in models_no.items():
        slug = name.replace(" ", "_").lower()
        print(f"Saved {plot_confusion(tm, f'confusion_matrix_{slug}.png')}")
    if with_news:
        slug = best_name.replace(" ", "_").lower()
        print(f"Saved {plot_confusion(final_model, f'confusion_matrix_{slug}_with_news.png')}")
    plot_comparison(comp_all)
    imp_path = plot_importance(final_model)
    if imp_path:
        print(f"Saved {imp_path}")

    # ---------------- Phase 12 : volatility --------------------------------
    thresholds = fit_thresholds(train)
    (REPORTS_DIR / "volatility_thresholds.json").write_text(
        json.dumps({k: round(v, 6) for k, v in thresholds.items()}, indent=2), encoding="utf-8")
    print(f"\nPhase 12 thresholds (train percentiles): {thresholds}")

    # ---------------- save deliverables ------------------------------------
    save_model(final_model)
    scaler = final_model.pipeline.named_steps["scaler"]
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"Saved {MODELS_DIR / 'scaler.pkl'}")

    bundle = {
        "model": final_model.pipeline,
        "features": features,
        "base_features": list(BASE_FEATURES),
        "model_name": best_name,
        "with_news": with_news,
        "thresholds": thresholds,
        "comparison": comp_all,
        "backtest_stats": stats,
        "train_end": str(train["Date"].max().date()),
        "test_start": str(test["Date"].min().date()),
        "news_info": news_info,
        "metrics": {k: v for k, v in final_model.metrics.items() if k != "ConfusionMatrix"},
        "best_params": final_model.best_params,
    }
    joblib.dump(bundle, MODELS_DIR / "bundle.pkl")
    print(f"Saved {MODELS_DIR / 'bundle.pkl'}")

    results.update({
        "model": best_name, "with_news": with_news, "backtest": stats,
        "thresholds": thresholds, "news": news_info,
    })
    print("\nDone. Artifacts in models/ and reports/.")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-news", action="store_true", help="skip Phases 9/10")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(use_news=not args.no_news, verbose=not args.quiet)
