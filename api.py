from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import io
import logging

app = FastAPI()

# ==============================
# 🔥 Load model safely
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

model = joblib.load(MODEL_PATH)

# ==============================
# 🔥 Business constants
# ==============================
THRESHOLD = 0.15
RETENTION_COST = 500
ANNUAL_REVENUE = 6000
SAVE_RATE = 0.6

# ==============================
# 🔥 Logging setup
# ==============================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ==============================
# 🔥 Input schema
# ==============================
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    PhoneService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    InternetService: str
    PaymentMethod: str


# ==============================
# 🔥 Helper: business logic
# ==============================
def get_decision_and_profit(prob):
    if prob >= THRESHOLD:
        decision = "retain"
        expected_profit = (SAVE_RATE * ANNUAL_REVENUE) - RETENTION_COST
    else:
        decision = "no_action"
        expected_profit = 0

    return decision, expected_profit


# ==============================
# 🔥 Single prediction endpoint
# ==============================
@app.post("/predict")
def predict(data: CustomerData):

    df = pd.DataFrame([data.dict()])

    prob = model.predict_proba(df)[:, 1][0]

    decision, expected_profit = get_decision_and_profit(prob)

    # Logging
    logging.info(f"SINGLE | INPUT: {data.dict()} | PROB: {prob} | DECISION: {decision}")

    return {
        "churn_probability": float(prob),
        "decision": decision,
        "threshold": THRESHOLD,
        "expected_profit": float(expected_profit)
    }


# ==============================
# 🔥 Batch prediction endpoint
# ==============================
@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):

    contents = file.file.read()
    df = pd.read_csv(io.BytesIO(contents))

    probs = model.predict_proba(df)[:, 1]
    results = []

    for i, prob in enumerate(probs):
        decision, expected_profit = get_decision_and_profit(prob)

        results.append({
            "churn_probability": float(prob),
            "decision": decision,
            "expected_profit": float(expected_profit)
        })

    # Logging
    logging.info(f"BATCH | rows: {len(df)}")

    return {
        "total_records": len(results),
        "predictions": results
    }