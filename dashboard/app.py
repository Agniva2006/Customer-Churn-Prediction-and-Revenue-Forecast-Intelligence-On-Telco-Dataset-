"""Standalone Streamlit dashboard v3.0 — direct unified pipeline inference without API requirement."""

import os
import joblib
import pandas as pd
import streamlit as st

from src.profit_simulation import compute_individualized_profit
from src.explainability import get_top_churn_drivers

# Resolve production model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Telecom Churn Intelligence — Standalone Dashboard")
st.caption("Direct pipeline inference engine (Offline Mode)")

@st.cache_resource
def load_production_pipeline():
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed loading model pipeline: {exc}")
        return None

pipeline = load_production_pipeline()

if pipeline is None:
    st.error(
        f"Production model pipeline not found at `{MODEL_PATH}`.\n\n"
        "Please generate the production pipeline first by running `python train.py`."
    )
    st.stop()

st.success("✅ Production Pipeline Loaded")

DEFAULT_THRESHOLD = 0.15

tab_single, tab_batch = st.tabs(["🔎 Single Customer Inference", "📁 Batch CSV Analytics"])

# ===================== SINGLE PREDICTION ===================
with tab_single:
    st.subheader("Customer Input Attributes")

    c1, c2, c3 = st.columns(3)

    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="dash_tenure")
        monthly = st.number_input("Monthly Charges (₹)", min_value=0.0, value=75.0, step=5.0, key="dash_monthly")
        total = st.number_input("Total Charges (₹)", min_value=0.0, value=900.0, step=50.0, key="dash_total")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="dash_contract")

    with c2:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="dash_internet")
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], key="dash_payment")
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="dash_phone")
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="dash_multi")

    with c3:
        sec_opts = ["Yes", "No", "No internet service"]
        online_sec = st.selectbox("Online Security", sec_opts, key="dash_sec")
        online_backup = st.selectbox("Online Backup", sec_opts, key="dash_backup")
        device = st.selectbox("Device Protection", sec_opts, key="dash_device")
        tech = st.selectbox("Tech Support", sec_opts, key="dash_tech")

    threshold = st.slider("Targeting Threshold", 0.05, 0.50, DEFAULT_THRESHOLD, 0.01, key="dash_thresh")

    if st.button("🚀 Run Inference", type="primary", use_container_width=True, key="dash_predict"):
        input_df = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "PaperlessBilling": "Yes",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
        }])

        prob = float(pipeline.predict_proba(input_df)[:, 1][0])
        profit_info = compute_individualized_profit(monthly, prob, threshold=threshold)
        drivers = get_top_churn_drivers(pipeline, input_df, top_n=3)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Churn Probability", f"{prob:.1%}")
        m2.metric("Risk Priority", profit_info["priority"])
        m3.metric("Customer CLV", f"₹{profit_info['clv']:,.0f}")
        m4.metric("Expected Net Profit", f"₹{profit_info['expected_profit']:,.0f}")

        st.progress(min(prob, 1.0))

        d_col, a_col = st.columns(2)
        with d_col:
            st.subheader("Top Risk Drivers (SHAP)")
            for d in drivers:
                st.info(f"**{d['feature']}**: {d['impact']}")

        with a_col:
            st.subheader("Retention Action Recommendation")
            st.warning(f"**{profit_info['action_quadrant']}** (Decision: {profit_info['decision'].upper()})")

# ===================== BATCH PREDICTION ====================
with tab_batch:
    st.subheader("Upload CSV File for Direct Inference")
    file_up = st.file_uploader("Upload customer CSV dataset", type=["csv"], key="dash_upload")

    if file_up is not None:
        df_batch = pd.read_csv(file_up)
        st.write(f"**Loaded {len(df_batch)} customer records**")
        st.dataframe(df_batch.head(), use_container_width=True)

        if st.button("🚀 Execute Batch Pipeline Prediction", type="primary", key="dash_batch_run"):
            with st.spinner("Executing pipeline inference..."):
                probs = pipeline.predict_proba(df_batch)[:, 1]

            res_df = df_batch.copy()
            res_df["churn_probability"] = probs.round(4)
            res_df["risk_level"] = pd.cut(probs, bins=[-0.01, 0.3, 0.6, 1.01], labels=["low", "medium", "high"])

            decisions = []
            profits = []
            quadrants = []

            for idx, p in enumerate(probs):
                m_val = float(df_batch.iloc[idx]["MonthlyCharges"]) if "MonthlyCharges" in df_batch.columns else 70.0
                info = compute_individualized_profit(m_val, p, threshold=threshold)
                decisions.append(info["decision"])
                profits.append(info["expected_profit"])
                quadrants.append(info["action_quadrant"])

            res_df["decision"] = decisions
            res_df["expected_profit"] = profits
            res_df["action_quadrant"] = quadrants

            st.divider()
            b1, b2, b3 = st.columns(3)
            b1.metric("Total Records", len(res_df))
            b2.metric("Retained Customers", (res_df["decision"] == "retain").sum())
            b3.metric("Total Expected Profit", f"₹{sum(profits):,.0f}")

            st.dataframe(
                res_df[["churn_probability", "risk_level", "decision", "expected_profit", "action_quadrant"]],
                use_container_width=True,
            )

            csv_out = res_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Batch Prediction Results CSV", csv_out, "dashboard_predictions.csv", "text/csv")