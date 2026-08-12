"""Enterprise FastAPI service for real-time churn prediction, SHAP explainability, and drift monitoring.

Endpoints
---------
* ``POST /predict``            — single customer prediction with SHAP drivers & CLV action quadrant
* ``POST /explain``            — detailed feature attribution drivers
* ``POST /predict_batch``      — CSV file batch prediction with aggregate metrics
* ``POST /monitor/drift``      — real-time data drift evaluation (PSI & KS statistics)
* ``GET  /health``             — liveness / readiness probe
* ``GET  /model-info``         — model metadata & pipeline topology
* ``POST /forecast/revenue``   — ARIMA revenue forecast with confidence intervals
* ``GET  /forecast/monte-carlo`` — Monte Carlo revenue risk simulation
"""

import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.profit_simulation import compute_individualized_profit, monte_carlo_revenue
from src.explainability import get_top_churn_drivers
from src.drift import evaluate_batch_drift
from src.database import log_prediction, get_recent_predictions
from src.forecasting import create_monthly_revenue, arima_forecast
from src.data_processing import clean_data
from src.auth import (
    register_user, login_user, get_user_by_token,
    update_profile, update_settings, logout_user
)

# ==============================
# App & CORS Configuration
# ==============================
app = FastAPI(
    title="Telecom Churn & Revenue Intelligence API",
    description="Enterprise predictive intelligence system delivering churn probabilities, SHAP explainability, CLV retention actions, revenue forecasting, and MLOps drift monitoring.",
    version="3.1.0",
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
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

model = None
baseline_df = None
model_metadata = None

def load_pipeline_model():
    global model, baseline_df, model_metadata
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Production pipeline model loaded successfully from %s", MODEL_PATH)
    except Exception as exc:
        model = None
        logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)

    if os.path.exists(RAW_DATA_PATH):
        try:
            raw_df = pd.read_csv(RAW_DATA_PATH)
            baseline_df = clean_data(raw_df)
        except Exception as e:
            logger.error("Failed to load or clean baseline data: %s", e)
            baseline_df = None

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
            logger.info("Model metadata loaded from %s", METADATA_PATH)
        except Exception as exc:
            model_metadata = None
            logger.warning("Failed to load model metadata: %s", exc)

load_pipeline_model()


def _require_model():
    """Guard: raise HTTP 503 if model is unavailable."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Production model pipeline is not loaded. Train the pipeline first by running 'python train.py'.",
        )


def _require_baseline():
    """Guard: raise HTTP 503 if baseline dataset is unavailable."""
    if baseline_df is None:
        raise HTTPException(
            status_code=503,
            detail="Baseline reference dataset unavailable. Ensure data/raw/ contains the Telco CSV.",
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
        "baseline_loaded": baseline_df is not None,
        "metadata_loaded": model_metadata is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.1.0",
    }


@app.get("/model-info")
def model_info():
    """Return metadata about the loaded production pipeline, including training metrics."""
    _require_model()

    info = {
        "model_type": type(model).__name__,
        "pipeline_steps": [step[0] for step in model.steps] if hasattr(model, "steps") else [],
        "model_path": MODEL_PATH,
        "default_threshold": DEFAULT_THRESHOLD,
        "business_assumptions": {
            "retention_cost": RETENTION_COST,
            "save_rate": SAVE_RATE,
        },
    }

    # Merge training metadata if available
    if model_metadata:
        info["model_version"] = model_metadata.get("model_version", "unknown")
        info["trained_at"] = model_metadata.get("trained_at", "unknown")
        info["pipeline_type"] = model_metadata.get("pipeline_type", "unknown")
        info["base_estimators"] = model_metadata.get("base_estimators", [])
        info["meta_learner"] = model_metadata.get("meta_learner", "unknown")
        info["dataset"] = model_metadata.get("dataset", {})
        info["performance"] = model_metadata.get("performance", {})

    return info


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


# ==============================
# Revenue Forecasting Endpoints
# ==============================
@app.post("/forecast/revenue")
def forecast_revenue(
    steps: int = Query(6, ge=1, le=24, description="Number of future periods to forecast"),
    order_p: int = Query(1, ge=0, le=5, description="ARIMA autoregressive order (p)"),
    order_d: int = Query(1, ge=0, le=2, description="ARIMA differencing order (d)"),
    order_q: int = Query(1, ge=0, le=5, description="ARIMA moving average order (q)"),
):
    """Generate ARIMA revenue forecast with confidence intervals from the baseline dataset."""
    _require_baseline()

    try:
        monthly_revenue = create_monthly_revenue(baseline_df)
        revenue_series = monthly_revenue["Revenue"]

        if len(revenue_series) < 4:
            raise HTTPException(
                status_code=422,
                detail="Insufficient monthly revenue data points for ARIMA modeling (minimum 4 required)."
            )

        predicted, ci = arima_forecast(
            revenue_series,
            steps=steps,
            order=(order_p, order_d, order_q),
        )

        forecast_data = []
        for i in range(steps):
            forecast_data.append({
                "period": int(len(revenue_series) + i),
                "predicted_revenue": round(float(predicted.iloc[i]), 2),
                "lower_bound": round(float(ci.iloc[i]["lower"]), 2),
                "upper_bound": round(float(ci.iloc[i]["upper"]), 2),
            })

        historical = []
        for _, row in monthly_revenue.iterrows():
            historical.append({
                "month": int(row["Month"]),
                "active_customers": int(row["Active_Customers"]),
                "churned": int(row["Churned"]),
                "avg_monthly_charges": round(float(row["Avg_Monthly_Charges"]), 2),
                "revenue": round(float(row["Revenue"]), 2),
            })

        logger.info("FORECAST_REVENUE | steps=%d order=(%d,%d,%d)", steps, order_p, order_d, order_q)

        return {
            "historical_monthly_revenue": historical,
            "forecast": forecast_data,
            "arima_order": [order_p, order_d, order_q],
            "periods_forecasted": steps,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Revenue forecast failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Revenue forecast computation failed: {exc}") from exc


@app.get("/forecast/monte-carlo")
def forecast_monte_carlo(
    n_customers: int = Query(7043, ge=100, le=100000, description="Customer base size"),
    avg_revenue: float = Query(6000.0, ge=100, description="Average annual revenue per customer (₹)"),
    churn_rate_mean: float = Query(0.27, ge=0.01, le=0.99, description="Mean churn rate"),
    churn_rate_std: float = Query(0.05, ge=0.001, le=0.3, description="Churn rate standard deviation"),
    n_simulations: int = Query(5000, ge=100, le=50000, description="Number of Monte Carlo simulations"),
):
    """Run Monte Carlo simulation to quantify revenue risk under churn uncertainty."""
    try:
        sim_revenues = monte_carlo_revenue(
            n_customers=n_customers,
            avg_revenue=avg_revenue,
            churn_rate_mean=churn_rate_mean,
            churn_rate_std=churn_rate_std,
            n_simulations=n_simulations,
        )

        var_5 = float(np.percentile(sim_revenues, 5))
        var_10 = float(np.percentile(sim_revenues, 10))

        logger.info(
            "MONTE_CARLO | n_customers=%d avg_rev=%.0f churn_mean=%.2f sims=%d VaR5=%.0f",
            n_customers, avg_revenue, churn_rate_mean, n_simulations, var_5
        )

        return {
            "simulation_parameters": {
                "n_customers": n_customers,
                "avg_revenue": avg_revenue,
                "churn_rate_mean": churn_rate_mean,
                "churn_rate_std": churn_rate_std,
                "n_simulations": n_simulations,
            },
            "results": {
                "mean_revenue": round(float(sim_revenues.mean()), 2),
                "median_revenue": round(float(np.median(sim_revenues)), 2),
                "std_revenue": round(float(sim_revenues.std()), 2),
                "min_revenue": round(float(sim_revenues.min()), 2),
                "max_revenue": round(float(sim_revenues.max()), 2),
                "value_at_risk_5pct": round(var_5, 2),
                "value_at_risk_10pct": round(var_10, 2),
                "percentiles": {
                    "p5": round(var_5, 2),
                    "p25": round(float(np.percentile(sim_revenues, 25)), 2),
                    "p50": round(float(np.percentile(sim_revenues, 50)), 2),
                    "p75": round(float(np.percentile(sim_revenues, 75)), 2),
                    "p95": round(float(np.percentile(sim_revenues, 95)), 2),
                },
            },
            "histogram_bins": [round(float(x), 2) for x in np.histogram(sim_revenues, bins=30)[1].tolist()],
            "histogram_counts": [int(x) for x in np.histogram(sim_revenues, bins=30)[0].tolist()],
        }
    except Exception as exc:
        logger.error("Monte Carlo simulation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Monte Carlo simulation failed: {exc}") from exc


# ==============================
# Authentication Endpoints
# ==============================
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ProfileUpdateRequest(BaseModel):
    full_name: str
    company: str = ""
    role: str = "analyst"

class SettingsUpdateRequest(BaseModel):
    default_threshold: float = 0.15
    notifications_enabled: bool = True
    dark_mode: bool = True


def _get_current_user(authorization: str = Header(None)):
    """Extract and validate user from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    token = authorization.split(" ", 1)[1]
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    return user


@app.post("/auth/register")
def auth_register(req: RegisterRequest):
    """Create a new user account."""
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    result = register_user(req.email, req.password, req.full_name)
    if "error" in result:
        raise HTTPException(status_code=409, detail=result["error"])
    return {"user": result}


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    """Authenticate and receive a session token."""
    result = login_user(req.email, req.password)
    if "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    return {"user": result}


@app.get("/auth/me")
def auth_me(authorization: str = Header(None)):
    """Get current authenticated user profile."""
    user = _get_current_user(authorization)
    return {"user": user}


@app.put("/auth/profile")
def auth_update_profile(req: ProfileUpdateRequest, authorization: str = Header(None)):
    """Update user profile information."""
    user = _get_current_user(authorization)
    updated = update_profile(user["id"], req.full_name, req.company, req.role)
    return {"user": updated}


@app.put("/auth/settings")
def auth_update_settings(req: SettingsUpdateRequest, authorization: str = Header(None)):
    """Update user preferences and settings."""
    user = _get_current_user(authorization)
    updated = update_settings(user["id"], req.default_threshold, req.notifications_enabled, req.dark_mode)
    return {"user": updated}


@app.post("/auth/logout")
def auth_logout(authorization: str = Header(None)):
    """Invalidate current session token."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        logout_user(token)
    return {"message": "Logged out successfully."}


# ==============================
# Frontend Serving
# ==============================
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend_static")


@app.get("/app")
@app.get("/app/{rest_of_path:path}")
def serve_frontend(rest_of_path: str = ""):
    """Serve the single-page application frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found. Ensure frontend/ directory exists.")