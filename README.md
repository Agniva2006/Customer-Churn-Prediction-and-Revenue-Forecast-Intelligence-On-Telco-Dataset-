# 📊 Telecom Customer Churn & Revenue Forecast Intelligence System (v3.0 Production Edition)

---

<img width="583" height="457" alt="image" src="https://github.com/user-attachments/assets/080c090b-d899-470a-9184-694df9f8956d" />

## 🏢 Enterprise Overview

The **Telecom Customer Churn & Revenue Forecast Intelligence System** is a production-grade machine learning platform and financial decision engine designed to solve subscriber attrition and quantify revenue exposure.

Rather than optimizing raw machine learning metrics (such as Accuracy or F1-Score) in isolation, this system integrates:
- **Unified Leak-Free Pipelines**: End-to-end scikit-learn pipeline encapsulation of data cleaning, feature engineering, column preprocessing, and probability calibration.
- **Cost-Sensitive Threshold Tuning**: Maximizes net retention profit using business parameters.
- **SHAP Model Explainability**: Generates individual customer feature attributions (top churn drivers) for every prediction.
- **CLV Retention Action Matrix**: Maps predicted probability and Customer Lifetime Value (CLV) into individualized intervention strategies.
- **Real-Time Drift Monitoring**: Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics against baseline distributions.
- **ARIMA & Monte Carlo Risk Simulation**: Forecasts 6-month revenue and quantifies 5th-percentile Value-at-Risk (VaR) exposure.

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
                  │  │ CalibratedClassifierCV (XGBoost)  │  │
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
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │   FastAPI REST API & Streamlit UIs      │
                  └─────────────────────────────────────────┘
```

---

## 📦 Key System Modules

| Module | File Link | Description |
| :--- | :--- | :--- |
| **Pipeline Transformers** | [`src/transformers.py`](file:///c:/Users/DELL/Desktop/ntblm/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/src/transformers.py) | Custom sklearn `BaseEstimator` & `TransformerMixin` classes for leak-free data cleaning and business feature creation. |
| **Unified Modeling** | [`src/modeling.py`](file:///c:/Users/DELL/Desktop/ntblm/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/src/modeling.py) | Assembles the complete pipeline with Isotonic `CalibratedClassifierCV` and `XGBClassifier`. |
| **Explainability** | [`src/explainability.py`](file:///c:/Users/DELL/Desktop/ntblm/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/src/explainability.py) | TreeSHAP engine extracting top 3 feature drivers per customer. |
| **Profit & Action Matrix** | [`src/profit_simulation.py`](file:///c:/Users/DELL/Desktop/ntblm/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/src/profit_simulation.py) | Computes individualized CLV, threshold net profit, and retention priority quadrant. |
| **Drift Monitoring** | [`src/drift.py`](file:///c:/Users/DELL/Desktop/ntblm/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/src/drift.py) | Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics engine. |
| **API Backend** | [`api.py`](file:///c:/Users/DELL/Desktop/ntblm/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/api.py) | Enterprise FastAPI service exposing `/health`, `/model-info`, `/predict`, `/explain`, `/predict_batch`, and `/monitor/drift`. |
| **Streamlit UIs** | [`streamlit_app.py`](file:///c:/Users/DELL/Desktop/ntblm/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/streamlit_app.py)<br/>[`dashboard/app.py`](file:///c:/Users/DELL/Desktop/ntblm/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/dashboard/app.py) | API-integrated frontend and standalone offline dashboard. |
| **Reproducible Training** | [`train.py`](file:///c:/Users/DELL/Desktop/ntblm/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/train.py) | End-to-end CLI training script exporting `models/churn_pipeline.pkl`. |

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

### 1. Install Dependencies & Build Pipeline

```bash
pip install -r requirements.txt
python train.py
```

### 2. Run Automated Pytest Suite

```bash
pytest tests/ -v
```

### 3. Launch Enterprise API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Swagger API documentation available at `http://localhost:8000/docs`.

### 4. Launch Streamlit Dashboards

```bash
# Option A: API-backed Streamlit Frontend
streamlit run streamlit_app.py

# Option B: Standalone Dashboard (Offline mode)
streamlit run dashboard/app.py
```

### 5. Docker Orchestration

```bash
docker-compose up --build
```

---

## 📄 License

This project is licensed under the MIT License for educational, portfolio, and commercial reference.
