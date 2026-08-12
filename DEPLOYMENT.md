# 🚀 Master Production Deployment & MLOps Architecture Guide (v3.1 Unified Edition)

This document provides a comprehensive, end-to-end guide for deploying the **ChurnGuard AI — Telecom Customer Churn & Revenue Forecast Platform** to production environments (**Render**, **Docker**, and **Linux VPS**).

---

## 🏛️ System Architecture Topology

By integrating the premium Web App directly into the FastAPI service, we have unified the frontend and backend. This completely eliminates CORS issues and simplifies orchestration down to **one single service**.

```
                         ┌─────────────────────────────┐
                         │   Web Browser (Client UI)   │
                         │      (Served at /app)       │
                         └──────────────┬──────────────┘
                                        │
                                        ▼ (API Requests)
                         ┌─────────────────────────────┐
                         │   FastAPI REST Engine API   │
                         │    (Gunicorn / Uvicorn)     │
                         └──────────────┬──────────────┘
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

### 🟦 Option A: 1-Click Deploy on Render.com

Render provides free hosting for web services with automatic HTTPS, Git integration, and managed build scripts.

#### Step-by-Step Render Deployment:
1. Log in to [dashboard.render.com](https://dashboard.render.com/).
2. Click **"New +"** (top right) ➔ Select **"Web Service"**.
3. Connect your GitHub repository.
4. Configure the Web Service settings:
   - **Name**: `churnguard-ai`
   - **Region**: Choose the closest region to your user base.
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python train.py`
   - **Start Command**: `gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
   - **Instance Type**: `Free`
5. Click **"Create Web Service"**.
6. Once deployed:
   - **App Interface**: `https://churnguard-ai.onrender.com/app`
   - **Swagger API Docs**: `https://churnguard-ai.onrender.com/docs`
   - **Liveness Probe**: `https://churnguard-ai.onrender.com/health`

---

## 🐳 PART 2: DOCKER & CONTAINER ORCHESTRATION

To run the complete platform inside isolated containers using the secure, hardened Docker setup (non-root privileges and health checks):

### Docker Compose Quickstart:
1. Launch all services in detached mode:
   ```bash
   docker-compose up -d --build
   ```
2. Verify running containers:
   ```bash
   docker-compose ps
   ```
3. Access endpoints:
   - **Web App UI**: `http://localhost:8000/app`
   - **Swagger Docs**: `http://localhost:8000/docs`
   - **Health Check**: `http://localhost:8000/health`

---

## 🖥️ PART 3: BARE-METAL / LINUX VM DEPLOYMENT (AWS EC2 / DigitalOcean)

For hosting on an Ubuntu virtual machine behind Nginx with an SSL certificate:

### Step-by-Step VPS Setup:
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
3. **Configure Systemd Daemon**:
   Save the following systemd definition to `/etc/systemd/system/churnguard.service`:
   ```ini
   [Unit]
   Description=ChurnGuard AI Unified Service
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-
   ExecStart=/home/ubuntu/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-/venv/bin/gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
4. **Configure Nginx Reverse Proxy with Free SSL**:
   Create a server block configuration under `/etc/nginx/sites-available/churnguard`:
   ```nginx
   server {
       server_name churnguard.yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```
   Enable the site and obtain SSL via Let's Encrypt:
   ```bash
   sudo ln -s /etc/nginx/sites-available/churnguard /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   sudo systemctl daemon-reload
   sudo systemctl enable --now churnguard
   sudo certbot --nginx -d churnguard.yourdomain.com
   ```

---

## ⚙️ PART 4: MLOPS ARCHITECTURE & MONITORING

### 🗄️ 1. Prediction Auditing & Database Persistence
Every prediction requested through `/predict` or `/predict_batch` is automatically logged to an embedded SQLite database (`logs/predictions.db`).
- **Table Schema**: Includes `timestamp`, billing rates, tenure, contract type, model calibrated probability, risk level (`low`, `medium`, `high`), CLV, expected net profit, and action quadrant.

### 📉 2. Distribution Drift Monitoring (PSI & KS Statistics)
The MLOps engine checks for covariate shift between baseline training data and production inference batches:
1. **Population Stability Index (PSI)**:
   - **PSI < 0.10**: Stable distribution (🟢 No Drift).
   - **0.10 ≤ PSI < 0.25**: Moderate shift warning (🟡 Warning).
   - **PSI ≥ 0.25**: Severe drift alert (🔴 Action Required - Retrain model).
2. **Two-Sample Kolmogorov-Smirnov (KS) Test**: Triggers warning when $p\text{-value} < 0.05$.

### 🔄 3. Continuous Integration & Continuous Delivery (CI/CD)
The repository includes a GitHub Actions workflow (`.github/workflows/ci-cd.yml`) that triggers automatically on every push or pull request to `main`:
1. Sets up Python 3.10 environment.
2. Runs static code linting via `flake8`.
3. Installs requirements.
4. Runs `python train.py` to fit the Stacking Ensemble model.
5. Verifies model metadata file creation.
6. Executes `pytest tests/ -v` (23 tests verifying transformers, pipeline, database logging, forecasting ARIMA bounds, and API endpoints).
7. Verifies `Docker build` correctness.
