"""
visualizations.py
-----------------
Plotting utilities for ATP Tennis match prediction analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE = {
    "Hard": "#4C72B0",
    "Clay": "#C44E52",
    "Grass": "#55A868",
    "default": "#8172B2",
}


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_surface_distribution(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """Bar chart of match counts by surface."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    counts = df["surface"].value_counts()
    colors = [PALETTE.get(s, PALETTE["default"]) for s in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Number of Matches by Surface", fontsize=13)
    ax.set_xlabel("Surface")
    ax.set_ylabel("Match Count")
    ax.tick_params(axis="x", rotation=0)
    return ax


def plot_rank_vs_win_rate(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """Scatter plot: current rank vs rolling win rate (winner side)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    col = next((c for c in df.columns if c.startswith("winner_win_rate_")), None)
    if col is None:
        ax.text(0.5, 0.5, "Run compute_player_stats() first", ha="center", va="center")
        return ax

    sample = df.sample(min(500, len(df)), random_state=0)
    ax.scatter(sample["winner_rank"], sample[col], alpha=0.4, s=15, color=PALETTE["Hard"])
    ax.set_title("Current Rank vs Recent Win Rate (Winners)", fontsize=13)
    ax.set_xlabel("ATP Ranking")
    ax.set_ylabel("Win Rate (last N matches)")
    return ax


def plot_age_distribution(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """Histogram of winner and loser ages."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.hist(df["winner_age"].dropna(), bins=25, alpha=0.6, color=PALETTE["Hard"], label="Winner")
    ax.hist(df["loser_age"].dropna(), bins=25, alpha=0.6, color=PALETTE["Clay"], label="Loser")
    ax.set_title("Age Distribution: Winners vs Losers", fontsize=13)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.legend()
    return ax


def plot_win_rate_by_surface(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """Box plots of winner win rates split by surface."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    col = next((c for c in df.columns if c.startswith("winner_win_rate_")), None)
    if col is None:
        ax.text(0.5, 0.5, "Run compute_player_stats() first", ha="center", va="center")
        return ax

    surfaces = sorted(df["surface"].dropna().unique())
    data_by_surface = [df.loc[df["surface"] == s, col].dropna().values for s in surfaces]
    colors = [PALETTE.get(s, PALETTE["default"]) for s in surfaces]

    bp = ax.boxplot(data_by_surface, patch_artist=True, labels=surfaces)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Winner Win Rate by Surface", fontsize=13)
    ax.set_xlabel("Surface")
    ax.set_ylabel("Win Rate")
    return ax


def plot_top_players(df: pd.DataFrame, top_n: int = 15, ax: plt.Axes = None) -> plt.Axes:
    """Horizontal bar chart of the most frequent match winners."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    top = df["winner_name"].value_counts().head(top_n)
    top[::-1].plot(kind="barh", ax=ax, color=PALETTE["Hard"], edgecolor="white")
    ax.set_title(f"Top {top_n} Players by Wins", fontsize=13)
    ax.set_xlabel("Number of Wins")
    return ax


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    ax: plt.Axes = None,
    title: str = "Confusion Matrix",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["P2 Wins", "P1 Wins"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13)
    return ax


def plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    ax: plt.Axes = None,
    label: str = "Model",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = np.trapz(tpr, fpr)
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc:.3f})", color=PALETTE["Hard"])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend()
    return ax


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 15,
    ax: plt.Axes = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    top = importances.head(top_n)[::-1]
    top.plot(kind="barh", ax=ax, color=PALETTE["default"], edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13)
    ax.set_xlabel("Importance Score")
    return ax


def plot_model_comparison(cv_results: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """Grouped bar chart comparing CV accuracy and ROC-AUC across models."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(cv_results))
    width = 0.35

    ax.bar(x - width / 2, cv_results["CV Accuracy (mean)"], width, label="Accuracy", color=PALETTE["Hard"])
    ax.bar(x + width / 2, cv_results["CV ROC-AUC (mean)"], width, label="ROC-AUC", color=PALETTE["Clay"])
    ax.set_xticks(x)
    ax.set_xticklabels(cv_results.index, rotation=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (5-Fold CV)", fontsize=13)
    ax.legend()
    return ax
