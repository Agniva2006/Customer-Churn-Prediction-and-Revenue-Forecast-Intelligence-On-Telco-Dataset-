# 🚀 Enterprise Deployment Guide — Telecom Churn Intelligence System

This document outlines the step-by-step instructions to deploy the **Telecom Customer Churn & Revenue Intelligence System** across free cloud services (Render, Streamlit Cloud) or enterprise infrastructure (Docker, AWS/GCP Linux VMs).

---

## 🌟 Method 1: Render.com 1-Click Blueprint (Recommended for API & Dashboard)

Render hosts both the **FastAPI REST Service** (`api.py`) and the **Streamlit Web Dashboard** (`streamlit_app.py`) simultaneously using the `render.yaml` specification.

### Instructions:
1. Ensure your latest changes are pushed to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete production deployment build"
   git push origin main
   ```
2. Log in to [dashboard.render.com](https://dashboard.render.com/).
3. Click **"New +"** (top right) ➔ Select **"Blueprint"**.
4. Connect your GitHub repository: `Agniva2006/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-`.
5. Render will detect `render.yaml` and configure 2 web services:
   - `telco-churn-api` (FastAPI REST Service)
   - `telco-churn-dashboard` (Streamlit Dashboard)
6. Click **"Apply"** to deploy.
7. Once completed:
   - Swagger API Docs: `https://telco-churn-api.onrender.com/docs`
   - Streamlit Web Dashboard: `https://telco-churn-dashboard.onrender.com`

---

## 🎈 Method 2: Streamlit Community Cloud (Free 1-Click UI Hosting)

Streamlit Cloud hosts the Streamlit UI directly from your GitHub repository for free.

### Instructions:
1. Log in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account (`Agniva2006`).
2. Click **"New app"**.
3. Select your deployment options:
   - **Repository**: `Agniva2006/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Click **"Deploy!"**.
5. Your dashboard will be live at `https://<your-custom-app-name>.streamlit.app` in under 2 minutes.

---

## 🐳 Method 3: Local or Cloud Docker Container Setup

Deploy the entire stack locally or on any server supporting Docker Desktop or Docker Engine.

### Instructions:
1. Build and launch services in detached mode:
   ```bash
   docker-compose up -d --build
   ```
2. Verify running services:
   ```bash
   docker-compose ps
   ```
3. Access endpoints:
   - **FastAPI Backend**: `http://localhost:8000/docs`
   - **Streamlit Frontend**: `http://localhost:8501`
4. Stop containers when done:
   ```bash
   docker-compose down
   ```

---

## 🖥️ Method 4: Production Linux Server (AWS EC2 / DigitalOcean Droplet)

Deploy on an Ubuntu/Debian server using Nginx as a reverse proxy and Certbot for free HTTPS.

### Instructions:
1. **Connect via SSH**:
   ```bash
   ssh ubuntu@<your-server-ip>
   ```
2. **Install Python, Nginx, and Certbot**:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx
   ```
3. **Clone Repository & Build Model**:
   ```bash
   git clone https://github.com/Agniva2006/Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-.git
   cd Customer-Churn-Prediction-and-Revenue-Forecast-Intelligence-On-Telco-Dataset-
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   python train.py
   ```
4. **Create Systemd Services**:
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
5. **Enable & Enable SSL**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now churn-api churn-dashboard
   sudo certbot --nginx
   ```
