"""Enterprise FastAPI service for real-time churn prediction, SHAP explainability, and drift monitoring.

Endpoints
---------
* ``POST /predict``        — single customer prediction with SHAP drivers & CLV action quadrant
* ``POST /explain``        — detailed feature attribution drivers
* ``POST /predict_batch``  — CSV file batch prediction with aggregate metrics
* ``POST /monitor/drift``  — real-time data drift evaluation (PSI & KS statistics)
* ``GET  /health``         — liveness / readiness probe
* ``GET  /model-info``     — model metadata & pipeline topology
"""

import io
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.profit_simulation import compute_individualized_profit
from src.explainability import get_top_churn_drivers
from src.drift import evaluate_batch_drift
from src.database import log_prediction, get_recent_predictions

# ==============================
# App & CORS Configuration
# ==============================
app = FastAPI(
    title="Telecom Churn & Revenue Intelligence API",
    description="Enterprise predictive intelligence system delivering churn probabilities, SHAP explainability, CLV retention actions, and MLOps drift monitoring.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Logging Configuration
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
# Model Loading (Graceful)
# ==============================
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

model = None
baseline_df = None

def load_pipeline_model():
    global model, baseline_df
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Production pipeline model loaded successfully from %s", MODEL_PATH)
    except Exception as exc:
        model = None
        logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)

    if os.path.exists(RAW_DATA_PATH):
        try:
            baseline_df = pd.read_csv(RAW_DATA_PATH)
        except Exception:
            baseline_df = None

load_pipeline_model()


def _require_model():
    """Guard: raise HTTP 503 if model is unavailable."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Production model pipeline is not loaded. Train the pipeline first by running 'python train.py'.",
        )


# ==============================
# Business Constants
# ==============================
DEFAULT_THRESHOLD = 0.15
RETENTION_COST = 500.0
SAVE_RATE = 0.6


# ==============================
# Input & Output Schemas
# ==============================
class CustomerData(BaseModel):
    """Schema for a single customer prediction request."""

    tenure: int = Field(..., ge=0, le=72, description="Months as a customer")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount (₹)")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date (₹)")
    PhoneService: str = Field("Yes", description="Yes / No")
    MultipleLines: str = Field("No", description="Yes / No / No phone service")
    InternetService: str = Field("Fiber optic", description="DSL / Fiber optic / No")
    OnlineSecurity: str = Field("No", description="Yes / No / No internet service")
    OnlineBackup: str = Field("No", description="Yes / No / No internet service")
    DeviceProtection: str = Field("No", description="Yes / No / No internet service")
    TechSupport: str = Field("No", description="Yes / No / No internet service")
    StreamingTV: str = Field("No", description="Yes / No / No internet service")
    StreamingMovies: str = Field("No", description="Yes / No / No internet service")
    Contract: str = Field("Month-to-month", description="Month-to-month / One year / Two year")
    PaperlessBilling: str = Field("Yes", description="Yes / No")
    PaymentMethod: str = Field("Electronic check", description="Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)")
    gender: str = Field("Female", description="Male / Female")
    SeniorCitizen: int = Field(0, ge=0, le=1, description="0 or 1")
    Partner: str = Field("No", description="Yes / No")
    Dependents: str = Field("No", description="Yes / No")

    model_config = {"json_schema_extra": {"examples": [{
        "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 846.0,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "gender": "Female",
        "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    }]}}


class ChurnDriver(BaseModel):
    feature: str
    impact: str
    shap_value: Optional[float] = None


class PredictionResult(BaseModel):
    churn_probability: float
    risk_level: str
    decision: str
    threshold: float
    clv: float
    expected_profit: float
    action_quadrant: str
    priority: str
    top_churn_drivers: List[ChurnDriver]


# ==============================
# Helper Functions
# ==============================
def _risk_level(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


# ==============================
# Service Endpoints
# ==============================
@app.get("/health")
def health():
    """Liveness probe returning server and model status."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.0.0",
    }


@app.get("/model-info")
def model_info():
    """Return metadata about the loaded production pipeline."""
    _require_model()
    return {
        "model_type": type(model).__name__,
        "pipeline_steps": [step[0] for step in model.steps] if hasattr(model, "steps") else [],
        "model_path": MODEL_PATH,
        "default_threshold": DEFAULT_THRESHOLD,
        "business_assumptions": {
            "retention_cost": RETENTION_COST,
            "save_rate": SAVE_RATE,
        },
    }


@app.post("/predict", response_model=PredictionResult)
def predict(data: CustomerData, threshold: float = Query(DEFAULT_THRESHOLD, ge=0.01, le=0.99)):
    """Predict churn probability, compute SHAP feature attributions, and assign CLV retention action."""
    _require_model()

    input_df = pd.DataFrame([data.model_dump()])

    try:
        prob = float(model.predict_proba(input_df)[:, 1][0])
    except Exception as exc:
        logger.error("Inference execution failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}") from exc

    # SHAP explanations
    raw_drivers = get_top_churn_drivers(model, input_df, top_n=3)
    drivers = [ChurnDriver(**d) for d in raw_drivers]

    # Individualized CLV profit & action matrix
    profit_info = compute_individualized_profit(
        monthly_charge=data.MonthlyCharges,
        prob=prob,
        threshold=threshold,
        retention_cost=RETENTION_COST,
        save_rate=SAVE_RATE,
    )

    try:
        log_prediction(
            monthly_charges=float(data.MonthlyCharges),
            total_charges=float(data.TotalCharges),
            tenure=int(data.tenure),
            contract=str(data.Contract),
            risk_probability=round(prob, 4),
            risk_level=_risk_level(prob),
            expected_profit=float(profit_info["expected_profit"]),
            clv=float(profit_info["clv"]),
            action_quadrant=str(profit_info["action_quadrant"])
        )
    except Exception as db_err:
        logger.error("Failed to log prediction to SQLite: %s", db_err)

    logger.info(
        "PREDICT | tenure=%d monthly=%.1f contract=%s | prob=%.4f decision=%s action='%s'",
        data.tenure, data.MonthlyCharges, data.Contract, prob, profit_info["decision"], profit_info["action_quadrant"]
    )

    return PredictionResult(
        churn_probability=round(prob, 4),
        risk_level=_risk_level(prob),
        decision=profit_info["decision"],
        threshold=threshold,
        clv=profit_info["clv"],
        expected_profit=profit_info["expected_profit"],
        action_quadrant=profit_info["action_quadrant"],
        priority=profit_info["priority"],
        top_churn_drivers=drivers,
    )


@app.post("/explain")
def explain(data: CustomerData):
    """Return comprehensive feature attributions for a single customer."""
    _require_model()
    input_df = pd.DataFrame([data.model_dump()])
    drivers = get_top_churn_drivers(model, input_df, top_n=5)
    return {"customer": data.model_dump(), "top_churn_drivers": drivers}


@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...), threshold: float = Query(DEFAULT_THRESHOLD)):
    """Batch-predict churn probabilities from an uploaded CSV file."""
    _require_model()

    contents = file.file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {exc}") from exc

    try:
        probs = model.predict_proba(df)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Batch model prediction failed: {exc}") from exc

    results = []
    total_expected_profit = 0.0

    for idx, prob in enumerate(probs):
        prob = float(prob)
        monthly_val = float(df.iloc[idx]["MonthlyCharges"]) if "MonthlyCharges" in df.columns else 70.0
        total_charges_val = float(df.iloc[idx]["TotalCharges"]) if "TotalCharges" in df.columns else 0.0
        tenure_val = int(df.iloc[idx]["tenure"]) if "tenure" in df.columns else 0
        contract_val = str(df.iloc[idx]["Contract"]) if "Contract" in df.columns else "Month-to-month"

        profit_info = compute_individualized_profit(monthly_val, prob, threshold=threshold)
        total_expected_profit += profit_info["expected_profit"]

        # Log prediction to database
        try:
            log_prediction(
                monthly_charges=monthly_val,
                total_charges=total_charges_val,
                tenure=tenure_val,
                contract=contract_val,
                risk_probability=round(prob, 4),
                risk_level=_risk_level(prob),
                expected_profit=float(profit_info["expected_profit"]),
                clv=float(profit_info["clv"]),
                action_quadrant=str(profit_info["action_quadrant"])
            )
        except Exception as db_err:
            logger.error("Failed to log batch prediction row %d: %s", idx, db_err)

        results.append({
            "record_index": idx,
            "churn_probability": round(prob, 4),
            "risk_level": _risk_level(prob),
            "decision": profit_info["decision"],
            "clv": profit_info["clv"],
            "expected_profit": profit_info["expected_profit"],
            "action_quadrant": profit_info["action_quadrant"],
        })

    logger.info("BATCH_PREDICT | rows=%d total_profit=%.2f", len(df), total_expected_profit)

    return {
        "total_records": len(results),
        "summary": {
            "mean_churn_probability": round(float(probs.mean()), 4),
            "retain_count": sum(1 for r in results if r["decision"] == "retain"),
            "no_action_count": sum(1 for r in results if r["decision"] == "no_action"),
            "total_expected_net_profit": round(total_expected_profit, 2),
        },
        "predictions": results,
    }


@app.post("/monitor/drift")
def monitor_drift(file: UploadFile = File(...)):
    """Evaluate real-time data drift (PSI and KS statistics) against baseline dataset."""
    if baseline_df is None:
        raise HTTPException(status_code=503, detail="Baseline reference dataset unavailable for drift comparison.")

    contents = file.file.read()
    try:
        current_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc

    drift_report = evaluate_batch_drift(baseline_df, current_df)
    return drift_report


@app.get("/monitor/recent")
def monitor_recent(limit: int = Query(100, ge=1, le=1000)):
    """Retrieve logged prediction audit history from the SQLite database."""
    try:
        df = get_recent_predictions(limit)
        # Convert DataFrame to a list of dicts
        records = df.to_dict(orient="records")
        return {"count": len(records), "records": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to query audit database: {exc}")