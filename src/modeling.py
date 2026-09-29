"""Phases 7 & 8 - Model development and evaluation.

Every model is wrapped in a ``Pipeline(StandardScaler -> classifier)`` so the
scaler is **fit on the training window only** (Phase-14 leakage guard #5), and
hyper-parameters are chosen with ``TimeSeriesSplit`` cross-validation, which
never lets the future inform the past (guard for Phase 6/14).

Models required by the brief:
    1. Logistic Regression
    2. Random Forest Classifier
    3. Gradient Boosting Classifier
    4. XGBoost (optional - skipped automatically if not installed)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import TARGET, make_split

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Model definitions (Phase 7)
# --------------------------------------------------------------------------
def _candidate_models() -> dict[str, tuple[Pipeline, dict]]:
    candidates: dict[str, tuple[Pipeline, dict]] = {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))]),
            {"clf__max_depth": [None, 6, 12], "clf__min_samples_leaf": [1, 5]},
        ),
        "Gradient Boosting": (
            Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(
                n_estimators=300, random_state=RANDOM_STATE))]),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5]},
        ),
    }
    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            {"clf__max_depth": [3, 5]},
        )
    except Exception:
        pass
    return candidates


# --------------------------------------------------------------------------
# Metrics (Phase 8)
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy / Precision / Recall / F1 / ROC-AUC / confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ConfusionMatrix": cm,
    }
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["ROC-AUC"] = np.nan
    return metrics


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    metrics: dict
    best_params: dict
    oof_feature_importance: pd.Series | None = None
    predictions: pd.DataFrame | None = None

    @property
    def importance(self) -> pd.Series | None:
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            return pd.Series(clf.feature_importances_, index=self._feature_names)
        if hasattr(clf, "coef_"):
            return pd.Series(np.abs(clf.coef_).ravel(), index=self._feature_names)
        return None

    _feature_names: list = field(default_factory=list, repr=False)


# --------------------------------------------------------------------------
# Training (Phase 7)
# --------------------------------------------------------------------------
def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    cv_splits: int = 4,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, TrainedModel]]:
    """Train every candidate, tune with time-series CV, score on the held-out window."""
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # chronological CV - rows must be time ordered for the folds to be honest
    order = np.argsort(train_df["Date"].values, kind="stable")
    cv = TimeSeriesSplit(n_splits=cv_splits)

    rows, models = [], {}
    for name, (pipe, grid) in _candidate_models().items():
        if verbose:
            print(f"\n=== {name} ===")
        gs = GridSearchCV(
            pipe, grid, cv=cv, scoring="f1", n_jobs=-1, refit=True, error_score=0.0
        )
        gs.fit(X_train.iloc[order], y_train.iloc[order])

        best = gs.best_estimator_
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
        metrics = evaluate(y_test, y_pred, y_prob)

        tm = TrainedModel(
            name=name,
            pipeline=best,
            metrics=metrics,
            best_params={k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            _feature_names=features,
        )
        models[name] = tm

        rows.append({
            "Model": name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1": round(metrics["F1"], 4),
            "ROC-AUC": round(metrics.get("ROC-AUC", np.nan), 4),
            "BestParams": tm.best_params,
        })
        if verbose:
            print(f"  params={tm.best_params}")
            print(f"  accuracy={metrics['Accuracy']:.4f}  f1={metrics['F1']:.4f}  "
                  f"auc={metrics.get('ROC-AUC', float('nan')):.4f}")
            print(f"  confusion matrix:\n{metrics['ConfusionMatrix']}")

    comparison = pd.DataFrame(rows)

    # no-skill baseline for honest context
    majority = int(np.round(y_train.mean() >= 0.5))
    base_pred = np.full(len(y_test), majority)
    comparison.loc[len(comparison)] = {
        "Model": "Baseline (majority class)",
        "Accuracy": round(accuracy_score(y_test, base_pred), 4),
        "Precision": round(precision_score(y_test, base_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, base_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, base_pred, zero_division=0), 4),
        "ROC-AUC": np.nan,
        "BestParams": {},
    }
    return comparison, models


def select_best(comparison: pd.DataFrame, metric: str = "F1") -> str:
    """Best *trained* model by ``metric`` (baseline row excluded).

    The index is reset first: after concatenating two comparison tables the row
    labels collide, and ``.loc[idxmax()]`` would otherwise return *two* rows
    (which then produced a bogus model name like "1    Random Forest\\n1 ...").
    """
    cand = comparison[~comparison["Model"].str.startswith("Baseline")].copy()
    cand = cand.reset_index(drop=True)
    return str(cand.loc[cand[metric].idxmax(), "Model"])


def save_model(model: TrainedModel, filename: str = "stock_model.pkl") -> Path:
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model.pipeline, path)
    print(f"Saved model -> {path}")
    return path


def load_model(filename: str = "stock_model.pkl"):
    import joblib

    return joblib.load(MODELS_DIR / filename)


if __name__ == "__main__":
    from .features import BASE_FEATURES, build_dataset

    full = build_dataset()
    train, test = make_split(full)
    comparison, models = train_models(train, test, list(BASE_FEATURES))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n", comparison.to_string(index=False))
