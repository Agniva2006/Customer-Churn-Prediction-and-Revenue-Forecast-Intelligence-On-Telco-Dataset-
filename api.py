from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/xgb_churn_model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[:,1][0]
    return {"churn_probability": float(prob)}