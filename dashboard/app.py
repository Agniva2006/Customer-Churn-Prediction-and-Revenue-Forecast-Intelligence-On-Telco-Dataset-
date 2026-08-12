"""Standalone Streamlit dashboard v3.1 — direct unified pipeline inference without API requirement."""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.profit_simulation import compute_individualized_profit, monte_carlo_revenue
from src.explainability import get_top_churn_drivers
from src.forecasting import create_monthly_revenue, arima_forecast

# Resolve production model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Telecom Churn Intelligence — Standalone Dashboard")
st.caption("Direct pipeline inference engine (Offline Mode) v3.1")

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

tab_single, tab_batch, tab_forecast = st.tabs([
    "🔎 Single Customer Inference", "📁 Batch CSV Analytics", "📈 Revenue Forecasting"
])

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

    with st.expander("Additional Options", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        gender = d1.selectbox("Gender", ["Female", "Male"], key="dash_gender")
        senior = d2.selectbox("Senior Citizen", [0, 1], key="dash_senior")
        partner = d3.selectbox("Partner", ["No", "Yes"], key="dash_partner")
        dependents = d4.selectbox("Dependents", ["No", "Yes"], key="dash_dep")
        e1, e2, e3 = st.columns(3)
        paperless = e1.selectbox("Paperless Billing", ["Yes", "No"], key="dash_paperless")
        streaming_tv = e2.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="dash_stv")
        streaming_movies = e3.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="dash_smov")

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
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "PaperlessBilling": paperless,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
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

# ===================== REVENUE FORECASTING ==================
with tab_forecast:
    st.subheader("📈 Revenue Forecasting & Monte Carlo Risk Simulation")

    # Check baseline data availability
    if not os.path.isfile(RAW_DATA_PATH):
        st.error(f"Baseline dataset not found at `{RAW_DATA_PATH}`. Revenue forecasting requires the raw Telco CSV.")
        st.stop()

    @st.cache_data
    def load_baseline():
        return pd.read_csv(RAW_DATA_PATH)

    baseline_df = load_baseline()

    fc1, fc2 = st.columns(2)

    with fc1:
        st.markdown("### 📊 ARIMA Revenue Forecast")

        f_steps = st.slider("Forecast Periods", 1, 12, 6, key="dash_f_steps")
        with st.expander("ARIMA Parameters", expanded=False):
            fp, fd, fq = st.columns(3)
            arima_p = fp.number_input("p (AR)", 0, 5, 1, key="dash_ar_p")
            arima_d = fd.number_input("d (Diff)", 0, 2, 1, key="dash_ar_d")
            arima_q = fq.number_input("q (MA)", 0, 5, 1, key="dash_ar_q")

        if st.button("📊 Generate Forecast", type="primary", use_container_width=True, key="dash_run_arima"):
            with st.spinner("Computing ARIMA forecast..."):
                try:
                    monthly_revenue = create_monthly_revenue(baseline_df)
                    revenue_series = monthly_revenue["Revenue"]
                    predicted, ci = arima_forecast(revenue_series, steps=f_steps, order=(arima_p, arima_d, arima_q))

                    fm1, fm2 = st.columns(2)
                    fm1.metric("Periods Forecasted", f_steps)
                    fm2.metric("Avg Forecast Revenue", f"₹{predicted.mean():,.0f}")

                    # Combined chart
                    chart_df = pd.DataFrame({
                        "Period": list(range(len(revenue_series))) + list(range(len(revenue_series), len(revenue_series) + f_steps)),
                        "Revenue": list(revenue_series) + list(predicted),
                    })
                    st.line_chart(chart_df, x="Period", y="Revenue")

                    # Forecast detail table
                    fc_detail = pd.DataFrame({
                        "Period": range(len(revenue_series), len(revenue_series) + f_steps),
                        "Predicted": predicted.values.round(2),
                        "Lower Bound": ci["lower"].values.round(2),
                        "Upper Bound": ci["upper"].values.round(2),
                    })
                    st.dataframe(fc_detail, use_container_width=True)

                except Exception as e:
                    st.error(f"Forecast failed: {e}")

    with fc2:
        st.markdown("### 🎲 Monte Carlo Revenue Risk")

        with st.expander("Simulation Parameters", expanded=True):
            mc_customers = st.number_input("Customers", 100, 100000, 7043, step=100, key="dash_mc_n")
            mc_revenue = st.number_input("Avg Revenue (₹)", 100.0, 50000.0, 6000.0, step=500.0, key="dash_mc_rev")
            mc_mean = st.slider("Mean Churn Rate", 0.05, 0.80, 0.27, 0.01, key="dash_mc_mean")
            mc_std = st.slider("Churn Std Dev", 0.01, 0.20, 0.05, 0.01, key="dash_mc_std")
            mc_sims = st.select_slider("Simulations", options=[100, 500, 1000, 2000, 5000, 10000], value=5000, key="dash_mc_sims")

        if st.button("🎲 Run Simulation", type="primary", use_container_width=True, key="dash_run_mc"):
            with st.spinner("Running Monte Carlo..."):
                sim_revenues = monte_carlo_revenue(
                    n_customers=mc_customers, avg_revenue=mc_revenue,
                    churn_rate_mean=mc_mean, churn_rate_std=mc_std,
                    n_simulations=mc_sims,
                )

                var5 = float(np.percentile(sim_revenues, 5))
                rm1, rm2 = st.columns(2)
                rm1.metric("Mean Revenue", f"₹{sim_revenues.mean():,.0f}")
                rm2.metric("VaR (5th pct)", f"₹{var5:,.0f}")

                # Histogram
                hist_counts, hist_edges = np.histogram(sim_revenues, bins=25)
                hist_labels = [f"₹{(hist_edges[i]+hist_edges[i+1])/2:,.0f}" for i in range(len(hist_counts))]
                hist_df = pd.DataFrame({"Revenue": hist_labels, "Count": hist_counts})
                st.bar_chart(hist_df, x="Revenue", y="Count")

                # Percentiles
                pcts = [5, 25, 50, 75, 95]
                pct_vals = [float(np.percentile(sim_revenues, p)) for p in pcts]
                pct_df = pd.DataFrame({
                    "Percentile": [f"{p}th" for p in pcts],
                    "Revenue (₹)": [f"₹{v:,.0f}" for v in pct_vals],
                })
                st.dataframe(pct_df, use_container_width=True, hide_index=True)