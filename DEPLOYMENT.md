# 🚀 Master Production Deployment & MLOps Architecture Guide

This document provides a comprehensive, end-to-end guide for deploying the **Telecom Customer Churn & Revenue Intelligence Platform** to production web environments (**Render**, **Streamlit Cloud**, **Docker**, and **Linux VPS**) along with full operational details for the **MLOps Auditing & Data Drift Architecture**.

---

## 🏛️ System Architecture Topology

```
                  ┌─────────────────────────────────────────┐
                  │       Streamlit Frontend Client         │
                  │   (Streamlit Cloud / Render Web App)    │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼ (REST API / CORS)
                  ┌─────────────────────────────────────────┐
                  │        FastAPI REST Engine API          │
                  │        (Render Web Service)             │
                  └────────────────────┬────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  Stacking Ensemble   │   │  SQLite Audit Log    │   │ PSI / KS Drift Engine│
│  (XGB + RF + GB + LR)│   │  (logs/predictions.db)│   │ (Baseline Ref Match) │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

---

## 🌐 PART 1: CLOUD PRODUCTION DEPLOYMENT

### 🟦 1. Backend Deployment on Render.com (FastAPI REST Service)

Render provides free hosting for web services with automatic HTTPS and Git integration.

#### Step-by-Step Backend Deployment:
1. Log in to [dashboard.render.com](https://dashboard.render.com/).
2. Click **"New +"** (top right) ➔ Select **"Web Service"**.
3. Connect your GitHub repository: `Agniva2006/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-`.
4. Configure the Web Service settings:
   - **Name**: `telco-churn-api`
   - **Region**: Choose closest region (e.g., Oregon / Singapore / Frankfurt).
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python train.py`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
5. Click **"Create Web Service"**.
6. Render will install dependencies, execute `train.py` to fit the Stacking Ensemble model, and launch Uvicorn.
7. Once live, test endpoints at:
   - **Liveness Probe**: `https://telco-churn-api.onrender.com/health`
   - **Interactive Swagger Docs**: `https://telco-churn-api.onrender.com/docs`

---

### 🎈 2. Frontend Deployment on Streamlit Community Cloud

Streamlit Community Cloud hosts the interactive dashboard UI directly from your GitHub repository.

#### Step-by-Step Frontend Deployment:
1. Log in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account (`Agniva2006`).
2. Click **"New app"** (or **"Create app"**).
3. Fill in the app repository details:
   - **Repository**: `Agniva2006/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Expand **"Advanced settings..."**:
   - Add the Environment Variable pointing to your live Render backend:
     ```env
     API_URL = "https://telco-churn-api.onrender.com"
     ```
5. Click **"Deploy!"**.
6. In 1–2 minutes, your dashboard will be live at `https://<your-app-name>.streamlit.app` with full API integration!

---

### 🌟 3. 1-Click Dual Deployment via Render Blueprint (`render.yaml`)

You can deploy **both** the FastAPI Backend and the Streamlit Frontend in a single click using Render's Infrastructure-as-Code Blueprint file:

1. Log in to [dashboard.render.com](https://dashboard.render.com/).
2. Click **"New +"** ➔ Select **"Blueprint"**.
3. Select your repository `Agniva2006/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-`.
4. Render automatically parses `render.yaml` and provisions:
   - `telco-churn-api` (FastAPI REST Engine)
   - `telco-churn-dashboard` (Streamlit Frontend UI)
5. Click **"Apply"**.

---

## ⚙️ PART 2: MLOPS ARCHITECTURE & MONITORING

### 🗄️ 1. Prediction Auditing & Database Persistence
Every prediction requested through `/predict` or `/predict_batch` is automatically logged to an embedded SQLite database (`logs/predictions.db`).

- **Table Schema**:
  - `id`: INTEGER PRIMARY KEY AUTOINCREMENT
  - `timestamp`: ISO-8601 Timestamp
  - `monthly_charges`: Numeric Monthly Charges
  - `total_charges`: Numeric Total Charges
  - `tenure`: Customer Tenure (months)
  - `contract`: Customer Contract Type
  - `risk_probability`: Calibrated Stacking Model Risk Score
  - `risk_level`: Risk Priority (`low`, `medium`, `high`)
  - `expected_profit`: Net Profit from Retention Intervention (₹)
  - `clv`: Customer Lifetime Value (₹)
  - `action_quadrant`: Priority Retention Quadrant Strategy
- **Audit Retrieval API**: `GET /monitor/recent?limit=100` returns historical logged inference payloads for audit reporting.

---

### 📉 2. Distribution Drift Monitoring (PSI & KS Statistics)
The MLOps engine checks for covariate shift between baseline training data and production inference batches:

1. **Population Stability Index (PSI)**:
   $$\text{PSI} = \sum \Big( \text{Actual}\% - \text{Expected}\% \Big) \times \ln\left( \frac{\text{Actual}\%}{\text{Expected}\%} \right)$$
   - **PSI < 0.10**: Stable distribution (🟢 No Drift).
   - **0.10 ≤ PSI < 0.25**: Moderate shift warning (🟡 Warning).
   - **PSI ≥ 0.25**: Severe drift alert (🔴 Action Required - Model Retraining needed).

2. **Two-Sample Kolmogorov-Smirnov (KS) Test**:
   Computes maximum distance between empirical CDFs of baseline and production batches. Triggers warning when $p\text{-value} < 0.05$.

---

### 🔄 3. Continuous Integration & Continuous Delivery (CI/CD)
The repository includes a GitHub Actions workflow (`.github/workflows/ci-cd.yml`) that triggers automatically on every push or pull request to `main`:
1. Sets up Python 3.10 environment.
2. Installs requirements.
3. Runs `python train.py` to fit the Stacking Ensemble model.
4. Executes `pytest tests/ -v` (12 unit tests verifying transformers, pipeline, database logging, and API endpoints).

---

## 🐳 PART 3: DOCKER & CONTAINER ORCHESTRATION

To run the complete platform inside isolated containers:

### Docker Compose Quickstart:
```bash
# Build and launch both API and Dashboard
docker-compose up -d --build

# Verify running containers
docker-compose ps

# Access endpoints
# FastAPI Swagger: http://localhost:8000/docs
# Streamlit Dashboard: http://localhost:8501
```

---

## 🖥️ PART 4: BARE-METAL / LINUX VM DEPLOYMENT (AWS EC2 / DigitalOcean)

### Step-by-Step Linux VPS Setup:
1. **Connect via SSH**:
   ```bash
   ssh ubuntu@<your-server-ip>
   ```
2. **Setup Dependencies & Environment**:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx
   git clone https://github.com/Agniva2006/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-.git
   cd Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   python train.py
   ```
3. **Configure Systemd Daemons**:
   - Save to `/etc/systemd/system/churn-api.service`:
     ```ini
     [Unit]
     Description=Telecom Churn FastAPI Service
     After=network.target

     [Service]
     User=ubuntu
     WorkingDirectory=/home/ubuntu/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-
     ExecStart=/home/ubuntu/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 --workers 4
     Restart=always

     [Install]
     WantedBy=multi-user.target
     ```
   - Save to `/etc/systemd/system/churn-dashboard.service`:
     ```ini
     [Unit]
     Description=Telecom Churn Streamlit Dashboard
     After=network.target

     [Service]
     User=ubuntu
     WorkingDirectory=/home/ubuntu/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-
     ExecStart=/home/ubuntu/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/venv/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
     Restart=always

     [Install]
     WantedBy=multi-user.target
     ```
4. **Enable Services & SSL**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now churn-api churn-dashboard
   sudo certbot --nginx
   ```
