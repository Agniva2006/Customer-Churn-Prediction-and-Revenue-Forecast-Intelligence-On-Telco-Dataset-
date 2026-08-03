"""Streamlit frontend for the Telecom Churn & Revenue Intelligence API v3.0."""

import os
import streamlit as st
import requests
import pandas as pd

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Telecom Churn & Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Glassmorphism & Custom Design System ───────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
        background: linear-gradient(135deg, #1e1e2f 0%, #0f172a 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.25rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 0.4rem 0 0;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .risk-high { border-left: 6px solid #ef4444 !important; }
    .risk-medium { border-left: 6px solid #f59e0b !important; }
    .risk-low { border-left: 6px solid #10b981 !important; }

    .driver-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #6366f1;
        color: #f8fafc;
    }
    .action-box {
        background: linear-gradient(135deg, #312e81 0%, #1e1b4b 100%);
        border: 1px solid #6366f1;
        border-radius: 14px;
        padding: 1.25rem;
        color: #e0e7ff;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Telecom Churn & Revenue Intelligence Platform</h1>
    <p>Predictive Churn Risk · SHAP Explainability · CLV Action Matrix · Profit Optimization</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: System Health Probe ───────────────────────────
with st.sidebar:
    st.header("⚙️ MLOps System Health")
    try:
        health_res = requests.get(f"{API_URL}/health", timeout=3)
        if health_res.status_code == 200:
            h_json = health_res.json()
            if h_json.get("model_loaded"):
                st.success("🟢 API Online · Model Pipeline Ready")
            else:
                st.warning("🟡 API Online · Model Pipeline NOT Loaded")
        else:
            st.error("🔴 API Degraded")
    except Exception:
        st.error("🔴 API Server Offline")
        st.info("Launch API via terminal:\n```bash\nuvicorn api:app --reload\n```")

    st.divider()
    st.header("🎯 Business Parameters")
    retention_threshold = st.slider("Classification Threshold", 0.05, 0.50, 0.15, 0.01,
                                    help="Optimized profit threshold (Default 0.15)")
    st.caption("Lower threshold prioritizes retention capture for high-value customers.")

# ── Form Inputs ────────────────────────────────────────────
st.subheader("👤 Customer Profile Input")

tab_input, tab_batch_ui, tab_mlops = st.tabs(["🔎 Single Prediction", "📁 Batch CSV Analytics", "⚙️ MLOps & Data Drift"])

with tab_input:
    c1, c2, c3 = st.columns(3)

    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges (₹)", min_value=0.0, value=75.0, step=5.0)
        total = st.number_input("Total Charges (₹)", min_value=0.0, value=900.0, step=50.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with c2:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with c3:
        sec_opts = ["Yes", "No", "No internet service"]
        online_sec = st.selectbox("Online Security", sec_opts)
        online_backup = st.selectbox("Online Backup", sec_opts)
        device = st.selectbox("Device Protection", sec_opts)
        tech = st.selectbox("Tech Support", sec_opts)

    with st.expander("Additional Demographics", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        gender = d1.selectbox("Gender", ["Female", "Male"])
        senior = d2.selectbox("Senior Citizen", [0, 1])
        partner = d3.selectbox("Partner", ["No", "Yes"])
        dependents = d4.selectbox("Dependents", ["No", "Yes"])
        paperless = "Yes"

    st.divider()

    if st.button("🚀 Analyze Churn Risk & Profit Action", type="primary", use_container_width=True):
        payload = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "PaperlessBilling": paperless,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
        }

        with st.spinner("Executing model pipeline & SHAP attribution..."):
            try:
                res = requests.post(f"{API_URL}/predict?threshold={retention_threshold}", json=payload, timeout=10)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.ConnectionError:
                st.error("Connection failed. Please start the FastAPI backend: `uvicorn api:app --reload`")
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        prob = data["churn_probability"]
        risk = data["risk_level"]
        decision = data["decision"]
        clv = data["clv"]
        profit = data["expected_profit"]
        quadrant = data["action_quadrant"]
        priority = data["priority"]
        drivers = data.get("top_churn_drivers", [])

        risk_css = f"risk-{risk}"

        # Metrics Display
        st.subheader("📈 Intelligence Assessment")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f'<div class="metric-card {risk_css}"><h3>{prob:.1%}</h3><p>Churn Probability</p></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card {risk_css}"><h3>{risk.upper()}</h3><p>Risk Priority: {priority}</p></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card {risk_css}"><h3>₹{clv:,.0f}</h3><p>Customer Lifetime Value (CLV)</p></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card {risk_css}"><h3>₹{profit:,.0f}</h3><p>Expected Retention Profit</p></div>', unsafe_allow_html=True)

        st.progress(min(prob, 1.0))

        # SHAP Drivers & Action Quadrant
        r1, r2 = st.columns(2)

        with r1:
            st.subheader("🔍 Top Risk Drivers (SHAP Attribution)")
            if drivers:
                for d in drivers:
                    st.markdown(f'<div class="driver-card"><strong>{d["feature"]}</strong>: {d["impact"]}</div>', unsafe_allow_html=True)
            else:
                st.info("No significant risk drivers identified.")

        with r2:
            st.subheader("🎯 CLV Retention Action Matrix")
            st.markdown(f'''
            <div class="action-box">
                <h4>Recommended Action: {quadrant}</h4>
                <p><strong>System Decision:</strong> {decision.upper()}</p>
                <p><strong>Priority Tier:</strong> {priority}</p>
            </div>
            ''', unsafe_allow_html=True)

# ── Batch Upload Tab ───────────────────────────────────────
with tab_batch_ui:
    st.subheader("📁 Upload CSV for Batch Assessment")
    file_up = st.file_uploader("Select CSV file", type=["csv"], key="streamlit_batch_file")

    if file_up is not None:
        if st.button("🚀 Process Batch CSV", type="primary"):
            files = {"file": (file_up.name, file_up.getvalue(), "text/csv")}
            try:
                b_res = requests.post(f"{API_URL}/predict_batch?threshold={retention_threshold}", files=files, timeout=30)
                b_res.raise_for_status()
                batch_data = b_res.json()

                st.success(f"Successfully processed {batch_data['total_records']} customer records.")
                summary = batch_data["summary"]

                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Total Customers", batch_data['total_records'])
                sm2.metric("Mean Churn Prob", f"{summary['mean_churn_probability']:.1%}")
                sm3.metric("Action Retain Count", summary['retain_count'])
                sm4.metric("Total Net Profit", f"₹{summary['total_expected_net_profit']:,.0f}")

                df_out = pd.DataFrame(batch_data["predictions"])
                st.dataframe(df_out, use_container_width=True)

                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Batch Prediction Results CSV", csv_bytes, "batch_churn_predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Batch processing failed: {exc}")

# ── MLOps & Data Drift Tab ─────────────────────────────────
with tab_mlops:
    st.subheader("⚙️ MLOps Prediction Auditing & Drift Diagnostics")

    m_col1, m_col2 = st.columns([1, 2])

    with m_col1:
        st.markdown("### 🗄️ Audit Database Status")
        try:
            audit_res = requests.get(f"{API_URL}/monitor/recent?limit=1000", timeout=5)
            if audit_res.status_code == 200:
                audit_data = audit_res.json()
                records = audit_data.get("records", [])
                st.metric("Total Logged Predictions (Audit DB)", len(records))
                
                if records:
                    df_audit = pd.DataFrame(records)
                    avg_prob = df_audit["risk_probability"].mean()
                    st.metric("Mean Logged Prediction Risk", f"{avg_prob:.1%}")
                    
                    # Risk breakdown chart
                    risk_counts = df_audit["risk_level"].value_counts()
                    st.write("#### Risk Level Share")
                    st.bar_chart(risk_counts)
                else:
                    st.info("No predictions recorded in audit logs yet. Submit predictions to populate.")
            else:
                st.error("Could not query predictions history from API.")
        except Exception as e:
            st.error(f"Prediction log service connection offline: {e}")

    with m_col2:
        st.markdown("### 📉 Live Data Drift Evaluation")
        st.markdown("Upload a production inference batch CSV file to compare its distribution against baseline training data.")
        
        drift_file = st.file_uploader("Upload production CSV batch to check drift", type=["csv"], key="streamlit_drift_file")
        if drift_file is not None:
            if st.button("🔍 Evaluate Distribution Drift", type="primary"):
                files = {"file": (drift_file.name, drift_file.getvalue(), "text/csv")}
                with st.spinner("Analyzing Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) stats..."):
                    try:
                        drift_res = requests.post(f"{API_URL}/monitor/drift", files=files, timeout=30)
                        drift_res.raise_for_status()
                        report = drift_res.json()
                        
                        overall_drift = report["overall_drift_detected"]
                        if overall_drift:
                            st.error("🚨 WARNING: Significant distribution drift detected! Consider retraining the model.")
                        else:
                            st.success("🟢 STABLE: No major distribution drift detected. Production data matches training data.")
                        
                        # Display feature table
                        rows = []
                        for feat, metrics in report.get("feature_metrics", {}).items():
                            rows.append({
                                "Feature": feat,
                                "PSI Score": metrics["psi"],
                                "KS Stat": metrics["ks_stat"],
                                "P-Value": metrics["p_value"],
                                "Drift Detected": "🔴 DRIFT" if metrics["drift_detected"] else "🟢 STABLE"
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    except Exception as exc:
                        st.error(f"Drift evaluation request failed: {exc}")
                        
        st.markdown("### 📋 Recent Logged Inputs")
        if 'df_audit' in locals() and not df_audit.empty:
            st.dataframe(df_audit.head(10)[["timestamp", "tenure", "monthly_charges", "total_charges", "risk_probability", "risk_level", "action_quadrant"]], use_container_width=True)
        else:
            st.info("No logs available to view.")