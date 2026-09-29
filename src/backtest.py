"""Phase 11 - Backtesting the final model over the historical test period.

For every trading day in the held-out window we compare the predicted direction
with the actual direction and report:

    total / correct / incorrect predictions
    UP predictions / DOWN predictions
    directional accuracy

plus a chart of actual vs. predicted direction (and the resulting cumulative
"strategy" return of always following the prediction).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def backtest(test_df: pd.DataFrame, pred_col: str = "Predicted", actual_col: str = "Actual") -> dict:
    """Compute the Phase-11 backtest statistics."""
    y_pred = test_df[pred_col].astype(int)
    y_true = test_df[actual_col].astype(int)

    correct = int((y_pred == y_true).sum())
    total = int(len(y_pred))
    stats = {
        "Total Predictions": total,
        "Correct Predictions": correct,
        "Incorrect Predictions": total - correct,
        "UP Predictions": int((y_pred == 1).sum()),
        "DOWN Predictions": int((y_pred == 0).sum()),
        "Actual UP Days": int((y_true == 1).sum()),
        "Actual DOWN Days": int((y_true == 0).sum()),
        "Directional Accuracy": round(correct / total, 4) if total else float("nan"),
        "Directional Accuracy %": round(100.0 * correct / total, 2) if total else float("nan"),
    }
    # accuracy when the model said UP vs. when it said DOWN
    for label, mask in (("UP", y_pred == 1), ("DOWN", y_pred == 0)):
        if mask.sum():
            stats[f"Accuracy on {label} calls"] = round(
                float((y_pred[mask] == y_true[mask]).mean()), 4
            )
    return stats


def actual_vs_predicted_df(test_df: pd.DataFrame, pred_col: str = "Predicted") -> pd.DataFrame:
    return test_df[["Date", "Symbol", "Close", "Actual", pred_col]].rename(
        columns={pred_col: "Predicted"}
    )


def plot_actual_vs_predicted(
    df: pd.DataFrame,
    pred_col: str = "Predicted",
    title: str = "Actual vs. predicted next-day direction",
    save_as: str = "backtest_actual_vs_predicted.png",
) -> Path:
    """Line/scatter comparison of the true and predicted direction."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    dates = pd.to_datetime(df["Date"])
    axes[0].plot(dates, df["Actual"], lw=1.0, color="#1f77b4", label="Actual direction")
    axes[0].plot(dates, df[pred_col], lw=1.0, color="#ff7f0e", alpha=0.75, label="Predicted direction")
    axes[0].set_ylabel("Direction (1 = UP, 0 = DOWN)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # daily correctness strip
    correct = (df["Actual"].values == df[pred_col].values).astype(int)
    axes[1].scatter(dates, correct, c=np.where(correct == 1, "#2ca02c", "#d62728"),
                    s=12, alpha=0.7)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Wrong", "Correct"])
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    path = REPORTS_DIR / save_as
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_cumulative_returns(
    df: pd.DataFrame,
    pred_col: str = "Predicted",
    save_as: str = "backtest_cumulative_return.png",
) -> Path:
    """Cumulative return of a naive 'trade the signal' rule vs. buy & hold."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for _, part in df.groupby("Symbol"):
        part = part.sort_values("Date").copy()
        part["ret"] = part["Close"].pct_change(fill_method=None).fillna(0.0)
        # a signal given at the close of day t earns the return of day t+1
        part["strat_ret"] = part[pred_col].shift(-1) * part["ret"]
        part["strategy"] = (1 + part["strat_ret"]).cumprod()
        part["hold"] = (1 + part["ret"]).cumprod()
        out.append(part)

    data = pd.concat(out)
    fig, ax = plt.subplots(figsize=(14, 5))
    for symbol, part in data.groupby("Symbol"):
        ax.plot(pd.to_datetime(part["Date"]), part["strategy"], lw=1.2,
                label=f"{symbol} - follow prediction")
        ax.plot(pd.to_datetime(part["Date"]), part["hold"], lw=1.0, ls="--", alpha=0.7,
                label=f"{symbol} - buy & hold")
    ax.set_title("Cumulative growth: strategy following predictions vs. buy & hold")
    ax.set_ylabel("Growth of 1.0")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = REPORTS_DIR / save_as
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_backtest(test_df, model, features, save_plots: bool = True):
    """End-to-end: predict the test window, score it, save the visuals."""
    df = test_df.copy()
    proba = model.pipeline.predict_proba(df[features])[:, 1]
    df["Predicted"] = (proba >= 0.5).astype(int)
    df["Confidence"] = np.where(df["Predicted"] == 1, proba, 1 - proba)
    df["Actual"] = df["Next_Day_Direction"].astype(int)

    stats = backtest(df)
    if save_plots:
        p1 = plot_actual_vs_predicted(df)
        p2 = plot_cumulative_returns(df)
        print(f"Saved {p1}\nSaved {p2}")
    return stats, df


if __name__ == "__main__":
    print(__doc__)
