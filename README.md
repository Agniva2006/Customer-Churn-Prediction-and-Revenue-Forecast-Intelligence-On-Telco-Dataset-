# 📊 Telecom Customer Churn & Revenue Forecast Intelligence System (v3.1 Production Edition)

![CI/CD](https://github.com/Agniva2006/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/actions/workflows/ci-cd.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

<img width="583" height="457" alt="image" src="https://github.com/user-attachments/assets/080c090b-d899-470a-9184-694df9f8956d" />

## 🏢 Enterprise Overview

The **Telecom Customer Churn & Revenue Forecast Intelligence System** is a production-grade machine learning platform and financial decision engine designed to solve subscriber attrition and quantify revenue exposure.

Rather than optimizing raw machine learning metrics (such as Accuracy or F1-Score) in isolation, this system integrates:
- **Stacking Ensemble Pipeline**: Calibrated multi-model stacking (XGBoost + Random Forest + Gradient Boosting) with Logistic Regression meta-learner inside a leak-free scikit-learn pipeline.
- **Cost-Sensitive Threshold Tuning**: Maximizes net retention profit using business parameters.
- **SHAP Model Explainability**: Generates individual customer feature attributions (top churn drivers) for every prediction.
- **CLV Retention Action Matrix**: Maps predicted probability and Customer Lifetime Value (CLV) into individualized intervention strategies.
- **Real-Time Drift Monitoring**: Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics against baseline distributions.
- **ARIMA Revenue Forecasting**: Time-series revenue projections with confidence intervals.
- **Monte Carlo Risk Simulation**: Quantifies 5th-percentile Value-at-Risk (VaR) revenue exposure under churn uncertainty.
- **SQLite Audit Database**: Every prediction is logged for compliance, monitoring, and drift analysis.

---

## 🏗️ Architecture Topology

```
                  ┌─────────────────────────────────────────┐
                  │          Raw Customer Payloads          │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │    Unified Scikit-Learn Pipeline        │
                  │  ┌───────────────────────────────────┐  │
                  │  │ TelcoCleanerTransformer           │  │
                  │  ├───────────────────────────────────┤  │
                  │  │ TelcoFeatureTransformer           │  │
                  │  ├───────────────────────────────────┤  │
                  │  │ ColumnTransformer (Scaler/OHE)    │  │
                  │  ├───────────────────────────────────┤  │
                  │  │ CalibratedClassifierCV            │  │
                  │  │   └─ StackingClassifier           │  │
                  │  │       ├─ XGBClassifier            │  │
                  │  │       ├─ RandomForestClassifier    │  │
                  │  │       ├─ GradientBoostingClassifier│  │
                  │  │       └─ Meta: LogisticRegression  │  │
                  │  └───────────────────────────────────┘  │
                  └────────────────────┬────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  SHAP Attribution    │   │  CLV Action Matrix   │   │ Real-time PSI & KS   │
│  Top Risk Drivers    │   │  Expected Net Profit │   │ Drift Evaluation     │
└───────────┬──────────┘   └───────────┬──────────┘   └───────────┬──────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  ARIMA Revenue       │   │ Monte Carlo VaR      │   │  SQLite Audit DB     │
│  Forecast + CI       │   │ Risk Simulation      │   │  Prediction Logging  │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │   FastAPI REST API & Streamlit UIs      │
                  └─────────────────────────────────────────┘
```

---

## 📦 Key System Modules

| Module | File | Description |
| :--- | :--- | :--- |
| **Pipeline Transformers** | `src/transformers.py` | Custom sklearn `BaseEstimator` & `TransformerMixin` classes for leak-free data cleaning and business feature creation. |
| **Unified Modeling** | `src/modeling.py` | Assembles the complete pipeline with Isotonic `CalibratedClassifierCV` wrapping a `StackingClassifier` ensemble. |
| **Explainability** | `src/explainability.py` | TreeSHAP engine extracting top feature drivers per customer, with heuristic fallback. |
| **Profit & Action Matrix** | `src/profit_simulation.py` | Computes individualized CLV, threshold net profit, retention priority quadrant, and Monte Carlo simulation. |
| **Revenue Forecasting** | `src/forecasting.py` | ARIMA-based time-series revenue forecasting with confidence intervals. |
| **Drift Monitoring** | `src/drift.py` | Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics engine. |
| **Database Audit** | `src/database.py` | SQLite prediction logging for compliance, monitoring, and audit trail. |
| **API Backend** | `api.py` | Enterprise FastAPI service with 8 endpoints (see below). |
| **Streamlit UIs** | `streamlit_app.py`, `dashboard/app.py` | API-integrated frontend and standalone offline dashboard. |
| **Reproducible Training** | `train.py` | CLI training script exporting `models/churn_pipeline.pkl` and `models/model_metadata.json`. |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness/readiness probe with model, baseline, and metadata status |
| `GET` | `/model-info` | Pipeline topology, training metrics, and model versioning metadata |
| `POST` | `/predict` | Single customer churn prediction with SHAP drivers & CLV action |
| `POST` | `/explain` | Detailed SHAP feature attribution for a single customer |
| `POST` | `/predict_batch` | Batch CSV prediction with aggregate profit metrics |
| `POST` | `/monitor/drift` | PSI & KS drift evaluation against baseline training data |
| `GET` | `/monitor/recent` | Retrieve prediction audit log from SQLite database |
| `POST` | `/forecast/revenue` | ARIMA revenue forecast with confidence intervals |
| `GET` | `/forecast/monte-carlo` | Monte Carlo revenue risk simulation with VaR metrics |

---

## 💰 Profit Optimization Formula

Default threshold $0.50$ assumes equal error costs. In telecom retention:
- **False Negative Cost**: Loss of annual customer revenue ($CLV_i$).
- **False Positive Cost**: Low-cost retention offer incentive (₹500).

The financial engine optimizes net profit:

$$\text{Net Profit} = \sum_{i \in \text{Retained}} \Big( \text{Save Rate} \times CLV_i \Big) - \sum_{i \in \text{Targeted}} \text{Retention Cost}$$

Setting threshold to **$0.15$** maximizes net profit by proactively targeting at-risk subscribers.

---

## 🚀 Quick Start & CLI Execution

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Agniva2006/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-.git
cd Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Production Pipeline

```bash
python train.py
```

This generates:
- `models/churn_pipeline.pkl` — serialized stacking ensemble pipeline
- `models/model_metadata.json` — training metrics and version metadata

### 3. Run Automated Pytest Suite

```bash
pytest tests/ -v
```

### 4. Launch Enterprise API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Swagger API documentation available at `http://localhost:8000/docs`.

### 5. Launch Streamlit Dashboards

```bash
# Option A: API-backed Streamlit Frontend (requires API running)
streamlit run streamlit_app.py

# Option B: Standalone Dashboard (Offline mode — no API required)
streamlit run dashboard/app.py
```

### 6. Docker Orchestration

```bash
docker-compose up --build
```

Services:
- **FastAPI API**: `http://localhost:8000` (Swagger: `/docs`)
- **Streamlit Dashboard**: `http://localhost:8501`

---

## ⚙️ Environment Variables

See `.env.example` for all configurable parameters:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `API_URL` | `http://127.0.0.1:8000` | Backend API URL for Streamlit frontends |
| `API_PORT` | `8000` | FastAPI server port |
| `DASHBOARD_PORT` | `8501` | Streamlit dashboard port |
| `DEFAULT_THRESHOLD` | `0.15` | Production classification threshold |

---

## 📄 License

This project is licensed under the MIT License for educational, portfolio, and commercial reference.
