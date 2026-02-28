import streamlit as st
import joblib
import pandas as pd

model = joblib.load("../models/xgb_churn_model.pkl")

st.title("Telecom Churn Prediction")

tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.number_input("Monthly Charges", value=70.0)

if st.button("Predict"):
    input_df = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly]
    })

    prob = model.predict_proba(input_df)[:, 1][0]
    st.write(f"Churn Probability: {prob:.2f}")