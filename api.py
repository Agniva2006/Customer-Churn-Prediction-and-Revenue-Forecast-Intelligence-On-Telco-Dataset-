"""FastAPI service for real-time and batch churn prediction.

Endpoints
---------
* ``POST /predict``        — single customer prediction
* ``POST /predict_batch``  — CSV file batch prediction
* ``GET  /health``         — liveness / readiness probe
* ``GET  /model-info``     — model metadata
"""

import io
import logging
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==============================
# App & CORS
# ==============================
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predict customer churn probability and get profit-optimised retention recommendations.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Logging
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "predictions.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================
# Model loading (graceful)
# ==============================
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except FileNotFoundError:
    model = None
    logger.error("Model file not found at %s — prediction endpoints will return 503.", MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    model = None
    logger.error("Failed to load model: %s", exc)


def _require_model():
    """Guard: raise 503 if the model is unavailable."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs for details.",
        )


# ==============================
# Business constants
# ==============================
THRESHOLD = 0.15
RETENTION_COST = 500
ANNUAL_REVENUE = 6000
SAVE_RATE = 0.6


# ==============================
# Input / output schemas
# ==============================
class CustomerData(BaseModel):
    """Schema for a single customer prediction request."""

    tenure: int = Field(..., ge=0, le=72, description="Months as a customer")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")
    PhoneService: str = Field(..., description="Yes / No")
    OnlineSecurity: str = Field(..., description="Yes / No / No internet service")
    OnlineBackup: str = Field(..., description="Yes / No / No internet service")
    DeviceProtection: str = Field(..., description="Yes / No / No internet service")
    TechSupport: str = Field(..., description="Yes / No / No internet service")
    StreamingTV: str = Field(..., description="Yes / No / No internet service")
    StreamingMovies: str = Field(..., description="Yes / No / No internet service")
    Contract: str = Field(..., description="Month-to-month / One year / Two year")
    InternetService: str = Field(..., description="DSL / Fiber optic / No")
    PaymentMethod: str = Field(..., description="Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)")

    model_config = {"json_schema_extra": {"examples": [{
        "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 846.0,
        "PhoneService": "Yes", "OnlineSecurity": "No",
        "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
    }]}}


class PredictionResult(BaseModel):
    churn_probability: float
    risk_level: str
    decision: str
    threshold: float
    expected_profit: float


# ==============================
# Helpers
# ==============================
def _risk_level(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


def _decision_and_profit(prob: float):
    if prob >= THRESHOLD:
        decision = "retain"
        expected_profit = (SAVE_RATE * ANNUAL_REVENUE) - RETENTION_COST
    else:
        decision = "no_action"
        expected_profit = 0.0
    return decision, expected_profit


# ==============================
# Endpoints
# ==============================
@app.get("/health")
def health():
    """Liveness / readiness probe."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info")
def model_info():
    """Return metadata about the loaded model."""
    _require_model()
    info = {
        "model_type": type(model).__name__,
        "model_path": MODEL_PATH,
        "threshold": THRESHOLD,
        "business_params": {
            "retention_cost": RETENTION_COST,
            "annual_revenue": ANNUAL_REVENUE,
            "save_rate": SAVE_RATE,
        },
    }
    # Attempt to pull feature names from the model/pipeline
    if hasattr(model, "feature_names_in_"):
        info["features"] = list(model.feature_names_in_)
    return info


@app.post("/predict", response_model=PredictionResult)
def predict(data: CustomerData):
    """Predict churn probability for a single customer."""
    _require_model()

    df = pd.DataFrame([data.model_dump()])
    prob = float(model.predict_proba(df)[:, 1][0])
    decision, expected_profit = _decision_and_profit(prob)
    risk = _risk_level(prob)

    logger.info(
        "SINGLE | tenure=%d monthly=%.1f contract=%s | prob=%.4f decision=%s",
        data.tenure, data.MonthlyCharges, data.Contract, prob, decision,
    )

    return PredictionResult(
        churn_probability=round(prob, 4),
        risk_level=risk,
        decision=decision,
        threshold=THRESHOLD,
        expected_profit=round(expected_profit, 2),
    )


@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    """Batch-predict churn probabilities from an uploaded CSV."""
    _require_model()

    contents = file.file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    probs = model.predict_proba(df)[:, 1]
    results = []

    for prob in probs:
        prob = float(prob)
        decision, expected_profit = _decision_and_profit(prob)
        results.append({
            "churn_probability": round(prob, 4),
            "risk_level": _risk_level(prob),
            "decision": decision,
            "expected_profit": round(expected_profit, 2),
        })

    logger.info("BATCH | rows=%d", len(df))

    return {
        "total_records": len(results),
        "summary": {
            "avg_churn_probability": round(float(probs.mean()), 4),
            "retain_count": sum(1 for r in results if r["decision"] == "retain"),
            "no_action_count": sum(1 for r in results if r["decision"] == "no_action"),
        },
        "predictions": results,
    }