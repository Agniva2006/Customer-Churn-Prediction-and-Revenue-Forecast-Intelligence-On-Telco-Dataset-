"""Standalone Streamlit dashboard — runs independently of the FastAPI service.

Loads the production pipeline directly from disk and supports
both single predictions and batch CSV uploads.
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

# ── Resolve model path ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Telecom Churn — Standalone Dashboard")
st.caption("Direct model inference without the FastAPI server")

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error(
        f"Model not found at `{MODEL_PATH}`.\n\n"
        "Please train the model first via the notebooks."
    )
    st.stop()

st.success("✅ Model loaded successfully")

# ── Business constants ─────────────────────────────────────
THRESHOLD = 0.15
RETENTION_COST = 500
ANNUAL_REVENUE = 6000
SAVE_RATE = 0.6

tab_single, tab_batch = st.tabs(["🔎 Single Prediction", "📁 Batch Upload"])

# ===================== SINGLE PREDICTION ===================
with tab_single:
    st.subheader("Enter Customer Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="s_tenure")
        monthly = st.number_input("Monthly Charges (₹)", min_value=0.0, value=70.0, step=5.0, key="s_monthly")
        total = st.number_input("Total Charges (₹)", min_value=0.0, value=840.0, step=50.0, key="s_total")

    with c2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="s_contract")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="s_internet")
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], key="s_payment")
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="s_phone")

    with c3:
        net_opts = ["Yes", "No", "No internet service"]
        online_sec = st.selectbox("Online Security", net_opts, key="s_sec")
        online_backup = st.selectbox("Online Backup", net_opts, key="s_backup")
        device = st.selectbox("Device Protection", net_opts, key="s_device")
        tech = st.selectbox("Tech Support", net_opts, key="s_tech")
        tv = st.selectbox("Streaming TV", net_opts, key="s_tv")
        movies = st.selectbox("Streaming Movies", net_opts, key="s_movies")

    if st.button("🚀 Predict", use_container_width=True, type="primary", key="s_predict"):
        input_df = pd.DataFrame([{
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
            "PaymentMethod": payment,
        }])

        prob = float(model.predict_proba(input_df)[:, 1][0])
        decision = "RETAIN" if prob >= THRESHOLD else "NO ACTION"
        profit = (SAVE_RATE * ANNUAL_REVENUE) - RETENTION_COST if prob >= THRESHOLD else 0

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability", f"{prob:.1%}")
        m2.metric("Decision", decision)
        m3.metric("Expected Profit", f"₹{profit:,.0f}")

        if prob >= 0.6:
            st.error("🔴 HIGH RISK — Immediate retention action recommended")
        elif prob >= 0.3:
            st.warning("🟡 MEDIUM RISK — Monitor closely")
        else:
            st.success("🟢 LOW RISK — Customer appears stable")

        st.progress(min(prob, 1.0))

# ===================== BATCH PREDICTION ====================
with tab_batch:
    st.subheader("Upload a CSV file")
    st.info(
        "The CSV must contain the same columns the model was trained on.\n\n"
        "Required columns: tenure, MonthlyCharges, TotalCharges, PhoneService, "
        "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, "
        "StreamingTV, StreamingMovies, Contract, InternetService, PaymentMethod"
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv", key="b_upload")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"**Uploaded {len(df)} rows**")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary", key="b_predict"):
            with st.spinner("Processing…"):
                probs = model.predict_proba(df)[:, 1]

            results = df.copy()
            results["churn_probability"] = probs.round(4)
            results["risk_level"] = pd.cut(
                probs, bins=[-0.01, 0.3, 0.6, 1.01],
                labels=["low", "medium", "high"],
            )
            results["decision"] = ["RETAIN" if p >= THRESHOLD else "NO ACTION" for p in probs]

            st.divider()
            st.subheader("Results")

            s1, s2, s3 = st.columns(3)
            s1.metric("Total Customers", len(results))
            s2.metric("Retain", (results["decision"] == "RETAIN").sum())
            s3.metric("Avg Churn Prob", f"{probs.mean():.1%}")

            st.dataframe(
                results[["churn_probability", "risk_level", "decision"]].style.background_gradient(
                    subset=["churn_probability"], cmap="RdYlGn_r"
                ),
                use_container_width=True,
            )

            # Download button
            csv_data = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )