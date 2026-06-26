"""Data loading, cleaning, and persistence for the Telco Churn dataset."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Loading data from %s", path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco Churn data.

    * Converts ``TotalCharges`` to numeric, filling blanks with the column median.
    * Maps the ``Churn`` column from Yes/No strings to 1/0 integers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe straight from ``load_data``.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the dataframe.
    """
    df = df.copy()

    # --- TotalCharges: coerce blanks to NaN, then fill with median ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_total = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)
    logger.info("TotalCharges: filled %d NaNs with median %.2f",
                df["TotalCharges"].isna().sum(), median_total)

    # --- Target encoding ---
    df["Churn"] = df["Churn"].str.strip().map({"Yes": 1, "No": 0})

    return df


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``customerID`` column if present."""
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped customerID column.")
    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV."""
    df.to_csv(path, index=False)
    logger.info("Saved processed data to %s (%d rows)", path, len(df))