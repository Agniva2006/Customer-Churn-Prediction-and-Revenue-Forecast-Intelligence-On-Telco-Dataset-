import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details and get prediction")

# =========================
# 🔹 Inputs
# =========================

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", value=50.0)
total = st.number_input("Total Charges", value=500.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# Optional toggles (defaulted but editable)
phone = st.selectbox("Phone Service", ["Yes", "No"])
online_sec = st.selectbox("Online Security", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
device = st.selectbox("Device Protection", ["Yes", "No"])
tech = st.selectbox("Tech Support", ["Yes", "No"])
tv = st.selectbox("Streaming TV", ["Yes", "No"])
movies = st.selectbox("Streaming Movies", ["Yes", "No"])

# =========================
# 🔹 Predict button
# =========================

if st.button("🚀 Predict"):

    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "PhoneService": phone,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment
    }

    try:
        res = requests.post(API_URL, json=data)
        result = res.json()

        st.subheader("📈 Prediction Result")

        st.metric("Churn Probability", f"{result['churn_probability']:.2f}")

        if result["decision"] == "retain":
            st.error("⚠️ Customer Likely to Churn → RETAIN")
        else:
            st.success("✅ Customer Stable → NO ACTION")

        st.metric("💰 Expected Profit", f"₹ {result['expected_profit']:.2f}")

    except Exception as e:
        st.error(f"API Error: {e}")