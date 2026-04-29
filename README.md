# 📊 Telecom Customer Churn & Revenue Forecast Intelligence System

---
<img width="583" height="457" alt="image" src="https://github.com/user-attachments/assets/080c090b-d899-470a-9184-694df9f8956d" />
<img width="707" height="473" alt="image" src="https://github.com/user-attachments/assets/f4febba9-5e27-4c73-8afc-ed9f2b9886b1" />
<img width="575" height="457" alt="image" src="https://github.com/user-attachments/assets/773f8896-5973-4bae-a245-2b496e13f4f7" />
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/6bb190bf-434a-4627-8b10-af43fd553fab" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/da543890-8f75-4e04-a93f-90fa33f40e5e" />
<img width="560" height="435" alt="image" src="https://github.com/user-attachments/assets/6b260b64-fa7f-4baa-97e2-e1a6e6acf933" />
<img width="575" height="413" alt="image" src="https://github.com/user-attachments/assets/cba2faea-3d7e-41f8-94af-bd4fc962ecc8" />
<img width="569" height="435" alt="image" src="https://github.com/user-attachments/assets/48985023-b98d-424d-b057-407781f699a9" />
<img width="578" height="413" alt="image" src="https://github.com/user-attachments/assets/9740901d-6bb5-4392-b676-0b0169bf4ef4" />
<img width="543" height="435" alt="image" src="https://github.com/user-attachments/assets/bf5960f0-d6d7-42e1-9c5f-755db91d64bf" />

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
