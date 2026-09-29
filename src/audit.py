"""Deliverable audit - verifies the 16 mandatory submission items exist.

    python -m src.audit
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DELIVERABLES = [
    ("1. Historical stock dataset", ["data/stock_data.csv"]),
    ("2. News/headline dataset", ["data/news_data.csv"]),
    ("3. Data-cleaning code", ["src/data_loader.py"]),
    ("4. EDA notebook", ["notebooks/01_EDA.ipynb"]),
    ("5. Technical-indicator calculations", ["src/indicators.py"]),
    ("6. Feature-engineering code", ["src/features.py"]),
    ("7. At least three trained ML models",
     ["models/stock_model.pkl", "models/scaler.pkl", "models/bundle.pkl"]),
    ("8. Model-comparison results",
     ["models/model_comparison.csv", "reports/model_comparison.csv",
      "reports/model_comparison.png"]),
    ("9. News sentiment analysis",
     ["src/sentiment.py", "reports/sentiment_distribution.csv"]),
    ("10. Final selected model", ["models/stock_model.pkl"]),
    ("11. Backtesting results",
     ["reports/backtest_stats.json", "reports/backtest_predictions.csv",
      "reports/backtest_actual_vs_predicted.png"]),
    ("12. Confusion matrix", None),          # matched by glob below
    ("13. Streamlit/Gradio application", ["app.py"]),
    ("14. requirements.txt", ["requirements.txt"]),
    ("15. README.md", ["README.md"]),
    ("16. Final project report", ["project_report.md"]),
]

EXTRA = [
    ("Supporting notebooks", ["notebooks/02_feature_engineering.ipynb",
                              "notebooks/03_model_training.ipynb"]),
    ("Volatility rule (Phase 12)", ["reports/volatility_thresholds.json"]),
    ("Sentiment with/without table (Phase 10)", ["reports/sentiment_impact.csv"]),
    ("Feature importance chart", ["reports/feature_importance.png"]),
    ("Price dataset processed", ["data/processed/features.csv"]),
]


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def main() -> int:
    ok_all = True
    print("=" * 74)
    print("MANDATORY DELIVERABLES")
    print("=" * 74)
    for name, files in DELIVERABLES:
        if files is None:  # glob pattern
            found = sorted((ROOT / "reports").glob("confusion_matrix_*.png"))
            good = bool(found)
            detail = ", ".join(p.name for p in found) if found else "MISSING"
        else:
            good = all(_size(ROOT / f) > 0 for f in files)
            detail = ", ".join(files)
        ok_all &= good
        print(f"[{'OK ' if good else 'MISS'}] {name:<42} {detail}")

    print("-" * 74)
    print("SUPPORTING ARTIFACTS")
    print("-" * 74)
    for name, files in EXTRA:
        good = all(_size(ROOT / f) > 0 for f in files)
        ok_all &= good
        print(f"[{'OK ' if good else 'MISS'}] {name:<42} {', '.join(files)}")

    stats_path = ROOT / "reports" / "backtest_stats.json"
    if stats_path.exists():
        print("-" * 74)
        print("BACKTEST SUMMARY")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for k, v in stats.items():
            print(f"  {k}: {v}")

    cmp_path = ROOT / "models" / "model_comparison.csv"
    if cmp_path.exists():
        import pandas as pd

        df = pd.read_csv(cmp_path)
        print("-" * 74)
        cols = [c for c in ["Experiment", "Model", "Accuracy", "Precision", "Recall", "F1"]
                if c in df.columns]
        print(df[cols].to_string(index=False))

    print("=" * 74)
    print("ALL DELIVERABLES PRESENT" if ok_all else "SOME DELIVERABLES MISSING")
    print("=" * 74)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
