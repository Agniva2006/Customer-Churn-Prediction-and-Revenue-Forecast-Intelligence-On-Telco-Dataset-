# 📊 Telecom Customer Churn & Revenue Forecast Intelligence System

---

## 🏢 Business Problem

Telecom companies rely heavily on recurring subscription revenue. However, high customer churn directly impacts profitability and increases customer acquisition costs.

This project builds an **end-to-end churn prediction and revenue forecasting system** that:

- Predicts customer churn
- Optimizes retention strategy using profit-based threshold tuning
- Forecasts future revenue under churn uncertainty
- Simulates business risk using Monte Carlo analysis
- Includes model calibration, drift detection, and deployment API

---

## 🎯 Objectives

1. Identify customers likely to churn.
2. Optimize retention strategy using cost-sensitive modeling.
3. Forecast 6-month revenue trends.
4. Quantify revenue risk under churn volatility.
5. Design production-aware ML monitoring and deployment logic.

---

## 📦 Dataset

**Telco Customer Churn Dataset**

- 7,043 customers  
- 21 raw features  
- Mix of demographic, service usage, and financial variables  
- Target variable: `Churn` (Yes/No)

---

## 🔎 Key EDA Insights

- Overall churn rate ≈ 26%
- Month-to-month contracts churn 3x more than long-term contracts
- Customers with tenure < 12 months show highest churn
- Customers without tech support exhibit elevated churn
- Electronic check payment users churn more frequently

These insights guided feature engineering and modeling strategy.

---

## 🧠 Feature Engineering

Advanced business-driven features were created:

- `tenure_group` (customer lifecycle segmentation)
- `service_count` (product dependency proxy)
- `avg_revenue_per_month`
- `contract_risk_score`
- `high_value_customer` flag
- `auto_payment` stability flag

This moved the project beyond basic dummy encoding and introduced lifecycle and financial intelligence into the modeling process.

---

## 🤖 Modeling Approach

### Models Evaluated

| Model               | ROC-AUC |
|--------------------|---------|
| Logistic Regression | ~0.82   |
| Random Forest       | ~0.85   |
| XGBoost             | ~0.87   |

### Final Model: XGBoost

**Why XGBoost?**

- Captures nonlinear interactions
- Handles multicollinearity effectively
- Strong ranking performance for imbalanced datasets
- Better generalization through boosting

### Validation Strategy

- Stratified 5-Fold Cross Validation
- RandomizedSearchCV hyperparameter tuning
- ROC-AUC & Precision-Recall analysis
- Calibration curve evaluation

---

## 🎯 Probability Calibration

Tree-based models often produce poorly calibrated probabilities.

To improve reliability:

- Used `CalibratedClassifierCV` with isotonic regression
- Generated calibration curves

Result:
Improved probability reliability, ensuring predicted churn probabilities better reflect real-world likelihood — critical for financial decisions.

---

## 💰 Profit Optimization Framework

Instead of using the default 0.5 threshold, threshold was optimized using business logic.

### Assumptions

- Retention cost: ₹500 per customer
- Annual revenue per customer: ₹6000
- Save rate: 60%

### Profit Formula
Net Profit = (True Positives * Save Rate * Annual Revenue) - (Predicted Positives * Retention Cost)

The **Profit vs Threshold curve** identified the optimal targeting threshold.

This transforms churn prediction into a **profit-optimized decision engine**, not just a classification model.

---

## 📈 Revenue Forecasting

Revenue was aggregated monthly and modeled using:

- ARIMA time-series modeling
- Confidence interval forecasting
- Residual diagnostics

Additionally:

- 6-month forward revenue forecast generated
- Scenario-based churn impact simulation conducted

This connects customer behavior directly to financial forecasting.

---

## 🎲 Monte Carlo Risk Simulation

To account for churn volatility:

- Simulated 1,000 churn scenarios
- Generated revenue distribution

### Outputs

- Expected revenue
- 5th percentile worst-case revenue
- Revenue distribution histogram

This introduces probabilistic risk modeling rather than deterministic forecasting.

---

## 🔁 Concept Drift Simulation

To simulate behavioral or pricing changes:

- Introduced synthetic feature shift
- Applied Kolmogorov–Smirnov (KS) test on prediction distributions

Purpose:
Detect when model performance may degrade due to changing customer behavior.

---

## 📊 Model Monitoring Logic

Production-level monitoring includes:

- Weekly ROC-AUC tracking
- Probability distribution drift detection
- Feature distribution shift detection
- Churn rate monitoring

Drift triggers retraining pipeline.

---

## 🚀 Deployment

Model deployed using:

- FastAPI REST API
- Swagger documentation
- Model serialization via `joblib`

### Example Endpoint

Returns:

```json
{
  "churn_probability": 0.82
}

telecom-churn-forecast/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_and_profit.ipynb
│   └── 04_revenue_forecast.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── profit_simulation.py
│   ├── forecasting.py
│   └── utils.py
│
├── models/
│   └── xgb_churn_model.pkl
│
├── dashboard/
│   └── app.py
│
├── reports/
│   ├── business_summary.pdf
│   └── figures/
│
├── requirements.txt
└── README.md