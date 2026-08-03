"""Database helper module to log model predictions for auditing, health metrics, and drift analysis."""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd

DB_DIR = Path(__file__).resolve().parent.parent / "logs"
DB_PATH = DB_DIR / "predictions.db"

def init_db():
    """Initialize the database and create the predictions table if it does not exist."""
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            monthly_charges REAL,
            total_charges REAL,
            tenure INTEGER,
            contract TEXT,
            risk_probability REAL,
            risk_level TEXT,
            expected_profit REAL,
            clv REAL,
            action_quadrant TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(
    monthly_charges: float,
    total_charges: float,
    tenure: int,
    contract: str,
    risk_probability: float,
    risk_level: str,
    expected_profit: float,
    clv: float,
    action_quadrant: str
):
    """Log a single prediction request and output metrics to the SQLite audit database."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, monthly_charges, total_charges, tenure, contract,
            risk_probability, risk_level, expected_profit, clv, action_quadrant
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, monthly_charges, total_charges, tenure, contract,
        risk_probability, risk_level, expected_profit, clv, action_quadrant
    ))
    conn.commit()
    conn.close()

def get_recent_predictions(limit: int = 100) -> pd.DataFrame:
    """Retrieve the most recent logged predictions from the database as a pandas DataFrame."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,)
        )
        return df
    finally:
        conn.close()

def get_all_predictions_count() -> int:
    """Get the total count of logged predictions in the database."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return count
