"""Streamlit frontend for the Telecom Churn Prediction API."""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2.2rem;
    }
    .metric-card p {
        margin: 0.3rem 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #f5365c 0%, #f56036 100%) !important;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fb6340 0%, #fbb140 100%) !important;
    }
    .risk-low {
        background: linear-gradient(135deg, #2dce89 0%, #2dcecc 100%) !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📊 Telecom Churn Intelligence System")
st.caption("Predict customer churn · Optimise retention · Maximise profit")
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ── Sidebar: API health ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health.get("model_loaded"):
            st.success("🟢 API Online · Model Loaded")
        else:
            st.warning("🟡 API Online · Model NOT loaded")
    except Exception:
        st.error("🔴 API Offline")
        st.info("Start the API with:\n```\nuvicorn api:app --reload\n```")

    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        "This system predicts whether a telecom customer is likely to churn "
        "and recommends a **profit-optimised retention action**.\n\n"
        "**Model:** XGBoost (calibrated)  \n"
        "**Threshold:** 0.15 (profit-optimised)  \n"
        "**Dataset:** Telco Customer Churn (7,043 records)"
    )

# ── Input Form ─────────────────────────────────────────────
st.subheader("🔎 Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12,
                        help="How long the customer has been with the company")
    monthly = st.number_input("Monthly Charges (₹)", min_value=0.0, value=70.0, step=5.0)
    total = st.number_input("Total Charges (₹)", min_value=0.0, value=840.0, step=50.0)

with col2:
    contract = st.selectbox("📄 Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("🌐 Internet Service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("💳 Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])
    phone = st.selectbox("📞 Phone Service", ["Yes", "No"])

with col3:
    net_dep_opts = ["Yes", "No", "No internet service"]
    online_sec = st.selectbox("🔒 Online Security", net_dep_opts)
    online_backup = st.selectbox("💾 Online Backup", net_dep_opts)
    device = st.selectbox("🛡️ Device Protection", net_dep_opts)
    tech = st.selectbox("🔧 Tech Support", net_dep_opts)
    tv = st.selectbox("📺 Streaming TV", net_dep_opts)
    movies = st.selectbox("🎬 Streaming Movies", net_dep_opts)

st.divider()

# ── Prediction ─────────────────────────────────────────────
if st.button("🚀 Predict Churn", use_container_width=True, type="primary"):

    payload = {
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
    }

    with st.spinner("Analysing customer profile…"):
        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            res.raise_for_status()
            result = res.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure `uvicorn api:app` is running.")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    prob = result["churn_probability"]
    risk = result["risk_level"]
    decision = result["decision"]
    profit = result["expected_profit"]

    risk_class = f"risk-{risk}"

    # ── Results ────────────────────────────────────────
    st.subheader("📈 Prediction Result")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(
            f'<div class="metric-card {risk_class}">'
            f'<h2>{prob:.1%}</h2>'
            f'<p>Churn Probability</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with r2:
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk, "⚪")
        st.markdown(
            f'<div class="metric-card {risk_class}">'
            f'<h2>{risk_emoji} {risk.upper()}</h2>'
            f'<p>Risk Level</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with r3:
        action_emoji = "🚨 RETAIN" if decision == "retain" else "✅ NO ACTION"
        st.markdown(
            f'<div class="metric-card {risk_class}">'
            f'<h2>{action_emoji}</h2>'
            f'<p>Expected Profit: ₹{profit:,.0f}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Churn probability bar ─────────────────────────
    st.progress(min(prob, 1.0))

    # ── Customer summary ──────────────────────────────
    with st.expander("📋 Customer Profile Summary", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"**Tenure:** {tenure} months")
            st.markdown(f"**Monthly Charges:** ₹{monthly:,.2f}")
            st.markdown(f"**Total Charges:** ₹{total:,.2f}")
            st.markdown(f"**Contract:** {contract}")
            st.markdown(f"**Internet:** {internet}")
        with s2:
            st.markdown(f"**Payment:** {payment}")
            st.markdown(f"**Phone Service:** {phone}")
            st.markdown(f"**Online Security:** {online_sec}")
            st.markdown(f"**Tech Support:** {tech}")
            services = sum(1 for s in [online_sec, online_backup, device, tech, tv, movies] if s == "Yes")
            st.markdown(f"**Services Subscribed:** {services} / 6")