"""
model.py
--------
Train, evaluate and use machine-learning models to predict ATP match outcomes.

Three classifiers are compared:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost (main model)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )


def build_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "XGBoost",
) -> dict:
    """
    Train the chosen model and return a results dict with metrics and the
    fitted estimator.
    """
    model_map = {
        "LogisticRegression": build_logistic_regression,
        "RandomForest": build_random_forest,
        "XGBoost": build_xgboost,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(model_map)}")

    clf = model_map[model_name]()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    results = {
        "model_name": model_name,
        "model": clf,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return results


def cross_validate_all(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation for all three models and return a
    summary DataFrame.
    """
    models = {
        "LogisticRegression": build_logistic_regression(),
        "RandomForest": build_random_forest(),
        "XGBoost": build_xgboost(),
    }

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []

    for name, clf in models.items():
        scores = cross_val_score(clf, X, y, cv=cv_splitter, scoring="accuracy", n_jobs=-1)
        auc_scores = cross_val_score(clf, X, y, cv=cv_splitter, scoring="roc_auc", n_jobs=-1)
        rows.append({
            "Model": name,
            "CV Accuracy (mean)": round(scores.mean(), 4),
            "CV Accuracy (std)": round(scores.std(), 4),
            "CV ROC-AUC (mean)": round(auc_scores.mean(), 4),
            "CV ROC-AUC (std)": round(auc_scores.std(), 4),
        })

    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_names: list) -> pd.Series:
    """
    Extract feature importances from a tree-based model or a Pipeline
    containing one.
    """
    clf = model
    # Unwrap Pipeline if necessary
    if hasattr(clf, "named_steps"):
        clf = clf.named_steps.get("clf", clf)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        raise ValueError("Model does not expose feature_importances_ or coef_.")

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Single-match prediction helper
# ---------------------------------------------------------------------------

def predict_match(
    model,
    p1_rank: int,
    p2_rank: int,
    p1_win_rate: float,
    p2_win_rate: float,
    surface: str = "Hard",
    best_of: int = 3,
    p1_avg_rank: float = None,
    p2_avg_rank: float = None,
    age_diff: float = 0.0,
    h2h_win_rate: float = 0.5,
) -> dict:
    """
    Predict the outcome of a single match.

    Returns a dict with keys 'p1_win_probability' and 'predicted_winner'.
    """
    p1_avg_rank = p1_avg_rank if p1_avg_rank is not None else p1_rank
    p2_avg_rank = p2_avg_rank if p2_avg_rank is not None else p2_rank

    features = pd.DataFrame([{
        "rank_diff": p2_rank - p1_rank,
        "p1_rank": p1_rank,
        "p2_rank": p2_rank,
        "p1_win_rate": p1_win_rate,
        "p2_win_rate": p2_win_rate,
        "win_rate_diff": p1_win_rate - p2_win_rate,
        "p1_avg_rank": p1_avg_rank,
        "p2_avg_rank": p2_avg_rank,
        "avg_rank_diff": p2_avg_rank - p1_avg_rank,
        "age_diff": age_diff,
        "h2h_win_rate": h2h_win_rate,
        "surface_clay": int(surface == "Clay"),
        "surface_grass": int(surface == "Grass"),
        "surface_hard": int(surface == "Hard"),
        "best_of_5": int(best_of == 5),
    }])

    prob = model.predict_proba(features)[0][1]
    return {
        "p1_win_probability": round(prob, 4),
        "p2_win_probability": round(1 - prob, 4),
        "predicted_winner": "Player 1" if prob >= 0.5 else "Player 2",
    }
