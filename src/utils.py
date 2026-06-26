"""Shared configuration constants, logging setup, and plot helpers."""

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import joblib

# ============================================================
# Project paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

RAW_CSV = os.path.join(DATA_RAW_DIR, "WA_Fn-UseC_-Telco-Customer-Churn (1).csv")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_pipeline.pkl")

# ============================================================
# Business constants
# ============================================================
RETENTION_COST = 500        # ₹ per customer
ANNUAL_REVENUE = 6000       # ₹ per customer
SAVE_RATE = 0.6             # probability a retention offer succeeds
DEFAULT_THRESHOLD = 0.15    # classification threshold for production


# ============================================================
# Logging
# ============================================================
def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Configure project-wide logging to console and optionally a file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        handlers=handlers,
    )


# ============================================================
# Model loading helper
# ============================================================
def load_production_model(path: Optional[str] = None):
    """Load the production model pipeline with clear error messaging."""
    path = path or MODEL_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            "Train the model first via the modeling notebook."
        )
    return joblib.load(path)


# ============================================================
# Plot helpers
# ============================================================
def plot_feature_importance(
    series,
    top_n: int = 15,
    title: str = "Top Feature Importances",
    figsize: tuple = (10, 6),
) -> None:
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=figsize)
    series.nlargest(top_n).sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_profit_curve(
    profit_df,
    optimal_threshold: Optional[float] = None,
    figsize: tuple = (10, 5),
) -> None:
    """Line chart of net profit vs. threshold, with optional optimum marker."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(profit_df["threshold"], profit_df["net_profit"],
            linewidth=2, color="#2ca02c")
    ax.fill_between(profit_df["threshold"], profit_df["net_profit"],
                    alpha=0.15, color="#2ca02c")

    if optimal_threshold is not None:
        row = profit_df.loc[
            (profit_df["threshold"] - optimal_threshold).abs().idxmin()
        ]
        ax.axvline(optimal_threshold, linestyle="--", color="red", alpha=0.7)
        ax.scatter(optimal_threshold, row["net_profit"],
                   color="red", s=100, zorder=5,
                   label=f"Optimal: {optimal_threshold:.2f}")
        ax.legend()

    ax.set_title("Net Profit vs Classification Threshold", fontsize=14, fontweight="bold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net Profit (₹)")
    plt.tight_layout()
    plt.show()