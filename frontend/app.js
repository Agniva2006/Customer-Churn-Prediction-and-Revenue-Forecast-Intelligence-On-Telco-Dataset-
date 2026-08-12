/* ═══════════════════════════════════════════
   ChurnGuard AI — Premium Frontend Application
   ═══════════════════════════════════════════ */

const API_BASE = ""; // Relative paths since FastAPI serves both API and static files

// Application State
const state = {
    token: localStorage.getItem("token") || null,
    user: null,
    activePage: "dashboard",
    health: { status: "offline", model_loaded: false },
    recentPredictions: [],
    forecastData: null,
    monteCarloData: null
};

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
    initAuthTabs();
    initForms();
    initNav();
    initMobileMenu();
    
    // Check if we are already logged in
    if (state.token) {
        checkAuthSession();
    } else {
        showScreen("auth-screen");
    }

    // Start background health checking
    pollSystemHealth();
    setInterval(pollSystemHealth, 30000);
});

// Toast Notifications
function showToast(message, type = "info") {
    const container = document.getElementById("toast-container") || createToastContainer();
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(100px)";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function createToastContainer() {
    const container = document.createElement("div");
    container.id = "toast-container";
    container.className = "toast-container";
    document.body.appendChild(container);
    return container;
}

// Screen Toggle
function showScreen(screenId) {
    document.getElementById("auth-screen").classList.add("hidden");
    document.getElementById("app-screen").classList.add("hidden");
    document.getElementById(screenId).classList.remove("hidden");
}

// Authentication Session & Login Handlers
function initAuthTabs() {
    const tabs = document.querySelectorAll(".auth-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            
            const formType = tab.dataset.tab;
            if (formType === "login") {
                document.getElementById("login-form").classList.remove("hidden");
                document.getElementById("register-form").classList.add("hidden");
            } else {
                document.getElementById("login-form").classList.add("hidden");
                document.getElementById("register-form").classList.remove("hidden");
            }
        });
    });
}

async function checkAuthSession() {
    try {
        const res = await fetch(`${API_BASE}/auth/me`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        if (res.ok) {
            const data = await res.json();
            state.user = data.user;
            updateUserUI();
            showScreen("app-screen");
            navigateTo(state.activePage);
        } else {
            handleLogout();
        }
    } catch (e) {
        console.error("Auth check failed:", e);
        showScreen("auth-screen");
    }
}

function initForms() {
    // Login Form Submit
    document.getElementById("login-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const email = document.getElementById("login-email").value;
        const password = document.getElementById("login-password").value;
        const submitBtn = e.target.querySelector("button[type='submit']");
        
        setBtnLoading(submitBtn, true);
        document.getElementById("login-error").classList.add("hidden");
        
        try {
            const res = await fetch(`${API_BASE}/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();
            if (res.ok) {
                state.token = data.user.token;
                state.user = data.user;
                localStorage.setItem("token", state.token);
                updateUserUI();
                showScreen("app-screen");
                navigateTo("dashboard");
                showToast(`Welcome back, ${state.user.full_name}!`, "success");
            } else {
                document.getElementById("login-error").textContent = data.detail || "Authentication failed.";
                document.getElementById("login-error").classList.remove("hidden");
            }
        } catch (err) {
            document.getElementById("login-error").textContent = "Connection to server failed.";
            document.getElementById("login-error").classList.remove("hidden");
        } finally {
            setBtnLoading(submitBtn, false);
        }
    });

    // Register Form Submit
    document.getElementById("register-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const full_name = document.getElementById("reg-name").value;
        const email = document.getElementById("reg-email").value;
        const password = document.getElementById("reg-password").value;
        const submitBtn = e.target.querySelector("button[type='submit']");
        
        setBtnLoading(submitBtn, true);
        document.getElementById("register-error").classList.add("hidden");
        
        try {
            const res = await fetch(`${API_BASE}/auth/register`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password, full_name })
            });
            const data = await res.json();
            if (res.ok) {
                state.token = data.user.token;
                state.user = data.user;
                localStorage.setItem("token", state.token);
                updateUserUI();
                showScreen("app-screen");
                navigateTo("dashboard");
                showToast("Account created successfully!", "success");
            } else {
                document.getElementById("register-error").textContent = data.detail || "Registration failed.";
                document.getElementById("register-error").classList.remove("hidden");
            }
        } catch (err) {
            document.getElementById("register-error").textContent = "Connection to server failed.";
            document.getElementById("register-error").classList.remove("hidden");
        } finally {
            setBtnLoading(submitBtn, false);
        }
    });

    // Logout Click
    document.getElementById("logout-btn").addEventListener("click", handleLogout);
}

function handleLogout() {
    if (state.token) {
        fetch(`${API_BASE}/auth/logout`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${state.token}` }
        }).catch(err => console.log("Logout request error:", err));
    }
    state.token = null;
    state.user = null;
    localStorage.removeItem("token");
    showScreen("auth-screen");
    showToast("Logged out successfully.", "info");
}

function setBtnLoading(btn, isLoading) {
    const textSpan = btn.querySelector(".btn-text");
    const loaderSpan = btn.querySelector(".btn-loader");
    if (isLoading) {
        btn.disabled = true;
        textSpan.classList.add("hidden");
        loaderSpan.classList.remove("hidden");
    } else {
        btn.disabled = false;
        textSpan.classList.remove("hidden");
        loaderSpan.classList.add("hidden");
    }
}

function updateUserUI() {
    if (!state.user) return;
    
    // Sidebar info
    const init = state.user.full_name.split(" ").map(w => w[0]).join("").substring(0, 2).toUpperCase();
    document.getElementById("sidebar-avatar").textContent = init;
    document.getElementById("sidebar-avatar").style.backgroundColor = state.user.avatar_color || "#6366f1";
    document.getElementById("sidebar-name").textContent = state.user.full_name;
    document.getElementById("sidebar-role").textContent = state.user.role || "Analyst";

    // Dark/Light Theme Settings sync
    if (state.user.dark_mode) {
        document.documentElement.style.setProperty("--bg-base", "#0a0a1a");
        document.documentElement.style.setProperty("--bg-deep", "#050510");
        document.documentElement.style.setProperty("--text", "#f1f5f9");
    } else {
        // Simple elegant light mode variables overriding dark mode base theme
        document.documentElement.style.setProperty("--bg-base", "#f8fafc");
        document.documentElement.style.setProperty("--bg-deep", "#f1f5f9");
        document.documentElement.style.setProperty("--text", "#0f172a");
    }
}

// Navigation & SPA Routing
function initNav() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            navItems.forEach(i => i.classList.remove("active"));
            item.classList.add("active");
            navigateTo(item.dataset.page);
        });
    });
}

function initMobileMenu() {
    const toggle = document.getElementById("mobile-menu-toggle");
    const sidebar = document.getElementById("sidebar");
    
    toggle.addEventListener("click", (e) => {
        e.stopPropagation();
        sidebar.classList.toggle("open");
    });
    
    document.addEventListener("click", () => {
        sidebar.classList.remove("open");
    });
}

function navigateTo(pageId) {
    state.activePage = pageId;
    const pageTitle = pageId.charAt(0).toUpperCase() + pageId.slice(1);
    document.getElementById("page-title").textContent = pageTitle;
    
    const container = document.getElementById("page-container");
    container.innerHTML = `<div class="page-loader"><div class="spinner"></div><p>Loading ${pageTitle}...</p></div>`;
    
    switch (pageId) {
        case "dashboard":
            renderDashboard(container);
            break;
        case "predict":
            renderPredict(container);
            break;
        case "batch":
            renderBatch(container);
            break;
        case "forecast":
            renderForecast(container);
            break;
        case "monitor":
            renderMonitor(container);
            break;
        case "profile":
            renderProfile(container);
            break;
        case "settings":
            renderSettings(container);
            break;
        default:
            container.innerHTML = `<div class="empty-state"><h3>Page not found</h3></div>`;
    }
}

// System Health Checks
async function pollSystemHealth() {
    const statusDiv = document.getElementById("system-status");
    if (!statusDiv) return;

    const dot = statusDiv.querySelector(".status-dot");
    const text = statusDiv.querySelector(".status-text");

    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            const data = await res.json();
            state.health = data;
            if (data.status === "healthy") {
                dot.className = "status-dot status-ok";
                text.textContent = "API & Model Online";
            } else {
                dot.className = "status-dot status-loading";
                text.textContent = "Degraded State";
            }
        } else {
            dot.className = "status-dot status-error";
            text.textContent = "Service Error";
        }
    } catch (e) {
        dot.className = "status-dot status-error";
        text.textContent = "API Server Offline";
    }
}

// Page Renderers

// 1. Dashboard View
async function renderDashboard(container) {
    try {
        // Fetch recent predictions logged in SQLite
        const res = await fetch(`${API_BASE}/monitor/recent?limit=100`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        
        let count = 0;
        let avgRisk = 0;
        let records = [];
        
        if (res.ok) {
            const data = await res.json();
            records = data.records || [];
            count = records.length;
            if (count > 0) {
                const totalRisk = records.reduce((acc, row) => acc + (row.risk_probability || 0), 0);
                avgRisk = totalRisk / count;
            }
        }
        
        container.innerHTML = `
            <div class="metric-grid">
                <div class="metric-card metric-primary">
                    <span class="metric-label">Logged Audit Requests</span>
                    <h3 class="metric-value" id="dash-count">${count}</h3>
                    <p class="metric-change">Stored in SQLite audit trail</p>
                    <div class="metric-icon">📁</div>
                </div>
                <div class="metric-card metric-warning">
                    <span class="metric-label">Average Risk Rate</span>
                    <h3 class="metric-value" id="dash-avg-risk">${(avgRisk * 100).toFixed(1)}%</h3>
                    <p class="metric-change">Rolling cohort average</p>
                    <div class="metric-icon">📈</div>
                </div>
                <div class="metric-card metric-success">
                    <span class="metric-label">Production Threshold</span>
                    <h3 class="metric-value">${state.user ? state.user.default_threshold : 0.15}</h3>
                    <p class="metric-change">Customizable parameter</p>
                    <div class="metric-icon">🎯</div>
                </div>
            </div>
            
            <div class="grid-2">
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">System Status Overview</h3>
                    </div>
                    <div style="display:flex; flex-direction:column; gap: 1rem;">
                        <div class="setting-item" style="padding:0.5rem 0;">
                            <span>Model Loaded</span>
                            <span class="risk-tag ${state.health.model_loaded ? 'low' : 'high'}">
                                ${state.health.model_loaded ? 'Ready' : 'Not Loaded'}
                            </span>
                        </div>
                        <div class="setting-item" style="padding:0.5rem 0;">
                            <span>Baseline Reference</span>
                            <span class="risk-tag ${state.health.baseline_loaded ? 'low' : 'high'}">
                                ${state.health.baseline_loaded ? 'Available' : 'Missing'}
                            </span>
                        </div>
                        <div class="setting-item" style="padding:0.5rem 0;">
                            <span>Platform Version</span>
                            <code style="color: var(--primary); font-weight:600;">v3.1.0 (PROD)</code>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Audit Prediction History</h3>
                        <span class="card-subtitle">Last 5 prediction requests</span>
                    </div>
                    <div class="table-wrapper">
                        <table>
                            <thead>
                                <tr>
                                    <th>Contract</th>
                                    <th>Monthly</th>
                                    <th>Churn Risk</th>
                                    <th>Action</th>
                                </tr>
                            </thead>
                            <tbody id="dash-history-body">
                                ${records.length === 0 ? '<tr><td colspan="4" style="text-align:center;">No predictions logged yet. Run some predictions first!</td></tr>' : 
                                  records.slice(0, 5).map(row => `
                                    <tr>
                                        <td>${row.contract || "N/A"}</td>
                                        <td>₹${row.monthly_charges || 0}</td>
                                        <td class="${row.risk_level === 'high' ? 'risk-high' : row.risk_level === 'medium' ? 'risk-medium' : 'risk-low'}" style="font-weight:600;">
                                            ${((row.risk_probability || 0) * 100).toFixed(1)}%
                                        </td>
                                        <td><span class="risk-tag ${row.risk_level === 'high' ? 'high' : row.risk_level === 'medium' ? 'medium' : 'low'}">${row.action_quadrant ? row.action_quadrant.split(" ")[0] : "None"}</span></td>
                                    </tr>
                                  `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    } catch(err) {
        container.innerHTML = `<div class="empty-state"><h3>Failed to load dashboard data.</h3></div>`;
    }
}

// 2. Churn Prediction Form
function renderPredict(container) {
    container.innerHTML = `
        <div class="card" style="margin-bottom:1.5rem;">
            <div class="card-header">
                <h3 class="card-title">Run Subscriber Analysis</h3>
                <span class="card-subtitle">Enter customer contract and usage variables to get prediction insights</span>
            </div>
            <form id="predict-customer-form" style="display:flex; flex-direction:column; gap:1.25rem;">
                <div class="form-row">
                    <div class="form-group">
                        <label>Tenure (months)</label>
                        <input type="number" id="tenure" min="0" max="72" value="12" required>
                    </div>
                    <div class="form-group">
                        <label>Monthly Charges (₹)</label>
                        <input type="number" id="MonthlyCharges" step="0.01" value="75.00" required>
                    </div>
                    <div class="form-group">
                        <label>Total Charges (₹)</label>
                        <input type="number" id="TotalCharges" step="0.01" value="900.00" required>
                    </div>
                    <div class="form-group">
                        <label>Contract Type</label>
                        <select id="Contract">
                            <option value="Month-to-month">Month-to-month</option>
                            <option value="One year">One year</option>
                            <option value="Two year">Two year</option>
                        </select>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>Internet Service</label>
                        <select id="InternetService">
                            <option value="Fiber optic">Fiber optic</option>
                            <option value="DSL">DSL</option>
                            <option value="No">No Internet Service</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Payment Method</label>
                        <select id="PaymentMethod">
                            <option value="Electronic check">Electronic check</option>
                            <option value="Mailed check">Mailed check</option>
                            <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                            <option value="Credit card (automatic)">Credit card (automatic)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Phone Service</label>
                        <select id="PhoneService">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Multiple Lines</label>
                        <select id="MultipleLines">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                            <option value="No phone service">No phone service</option>
                        </select>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>Online Security</label>
                        <select id="OnlineSecurity">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                            <option value="No internet service">No internet service</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Online Backup</label>
                        <select id="OnlineBackup">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                            <option value="No internet service">No internet service</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Device Protection</label>
                        <select id="DeviceProtection">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                            <option value="No internet service">No internet service</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Tech Support</label>
                        <select id="TechSupport">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                            <option value="No internet service">No internet service</option>
                        </select>
                    </div>
                </div>

                <div style="border-top: 1px solid var(--border); padding-top: 1.25rem;">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Gender</label>
                            <select id="gender">
                                <option value="Female">Female</option>
                                <option value="Male">Male</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Senior Citizen</label>
                            <select id="SeniorCitizen">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Partner</label>
                            <select id="Partner">
                                <option value="No">No</option>
                                <option value="Yes">Yes</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Dependents</label>
                            <select id="Dependents">
                                <option value="No">No</option>
                                <option value="Yes">Yes</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>Paperless Billing</label>
                        <select id="PaperlessBilling">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Streaming TV</label>
                        <select id="StreamingTV">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                            <option value="No internet service">No internet service</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Streaming Movies</label>
                        <select id="StreamingMovies">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                            <option value="No internet service">No internet service</option>
                        </select>
                    </div>
                </div>

                <button type="submit" class="btn btn-primary" style="align-self: flex-start;">
                    <span class="btn-text">🚀 Run Churn Analysis</span>
                    <span class="btn-loader hidden"></span>
                </button>
            </form>
        </div>

        <div id="prediction-result-container" class="hidden">
            <div class="metric-grid">
                <div class="metric-card" id="res-prob-card">
                    <span class="metric-label">Churn Probability</span>
                    <h3 class="metric-value" id="res-prob">0.0%</h3>
                    <p class="metric-change" id="res-risk-level">LOW RISK</p>
                    <div class="metric-icon">🤖</div>
                </div>
                <div class="metric-card metric-primary">
                    <span class="metric-label">Customer Lifetime Value</span>
                    <h3 class="metric-value" id="res-clv">₹0</h3>
                    <p class="metric-change">Annual Contract Value</p>
                    <div class="metric-icon">💎</div>
                </div>
                <div class="metric-card metric-success">
                    <span class="metric-label">Expected Net Profit</span>
                    <h3 class="metric-value" id="res-profit">₹0</h3>
                    <p class="metric-change">With retention simulation</p>
                    <div class="metric-icon">💰</div>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Top Risk Drivers (SHAP)</h3>
                        <span class="card-subtitle">Highest attribution drivers for this subscriber</span>
                    </div>
                    <div class="driver-list" id="res-drivers">
                        <!-- Rendered by JS -->
                    </div>
                </div>

                <div class="card" style="display:flex; flex-direction:column; justify-content:center;">
                    <div class="action-box">
                        <h3 id="res-quadrant">VIP Concierge Outreach</h3>
                        <p class="action-decision" style="margin-bottom:1rem;">Recommended Action Plan</p>
                        <div style="display:flex; justify-content:space-around; font-size:0.85rem;">
                            <div>
                                <span style="display:block; color:var(--text-muted);">SYSTEM DECISION</span>
                                <strong id="res-decision" style="color:var(--text); text-transform:uppercase;">RETAIN</strong>
                            </div>
                            <div>
                                <span style="display:block; color:var(--text-muted);">PRIORITY TIER</span>
                                <strong id="res-priority" style="color:var(--text);">P1</strong>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.getElementById("predict-customer-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const submitBtn = e.target.querySelector("button[type='submit']");
        setBtnLoading(submitBtn, true);
        
        const payload = {
            tenure: parseInt(document.getElementById("tenure").value),
            MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
            TotalCharges: parseFloat(document.getElementById("TotalCharges").value),
            Contract: document.getElementById("Contract").value,
            InternetService: document.getElementById("InternetService").value,
            PaymentMethod: document.getElementById("PaymentMethod").value,
            PhoneService: document.getElementById("PhoneService").value,
            MultipleLines: document.getElementById("MultipleLines").value,
            OnlineSecurity: document.getElementById("OnlineSecurity").value,
            OnlineBackup: document.getElementById("OnlineBackup").value,
            DeviceProtection: document.getElementById("DeviceProtection").value,
            TechSupport: document.getElementById("TechSupport").value,
            StreamingTV: document.getElementById("StreamingTV").value,
            StreamingMovies: document.getElementById("StreamingMovies").value,
            PaperlessBilling: document.getElementById("PaperlessBilling").value,
            gender: document.getElementById("gender").value,
            SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
            Partner: document.getElementById("Partner").value,
            Dependents: document.getElementById("Dependents").value
        };

        const threshold = state.user ? state.user.default_threshold : 0.15;

        try {
            const res = await fetch(`${API_BASE}/predict?threshold=${threshold}`, {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${state.token}`
                },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (res.ok) {
                // Populate results
                document.getElementById("res-prob").textContent = `${(data.churn_probability * 100).toFixed(1)}%`;
                document.getElementById("res-clv").textContent = `₹${data.clv.toLocaleString()}`;
                document.getElementById("res-profit").textContent = `₹${data.expected_profit.toLocaleString()}`;
                document.getElementById("res-quadrant").textContent = data.action_quadrant;
                document.getElementById("res-decision").textContent = data.decision;
                document.getElementById("res-priority").textContent = data.priority;

                const card = document.getElementById("res-prob-card");
                const riskLevel = document.getElementById("res-risk-level");
                card.className = "metric-card";
                
                if (data.risk_level === "high") {
                    card.classList.add("metric-danger");
                    riskLevel.textContent = "HIGH RISK";
                    riskLevel.className = "metric-change risk-high";
                } else if (data.risk_level === "medium") {
                    card.classList.add("metric-warning");
                    riskLevel.textContent = "MEDIUM RISK";
                    riskLevel.className = "metric-change risk-medium";
                } else {
                    card.classList.add("metric-success");
                    riskLevel.textContent = "LOW RISK";
                    riskLevel.className = "metric-change risk-low";
                }

                // Render Drivers
                const driversDiv = document.getElementById("res-drivers");
                driversDiv.innerHTML = "";
                if (data.top_churn_drivers && data.top_churn_drivers.length > 0) {
                    data.top_churn_drivers.forEach(d => {
                        const item = document.createElement("div");
                        item.className = "driver-item";
                        item.innerHTML = `
                            <div style="flex:1;">
                                <div class="driver-name">${d.feature}</div>
                                <div class="driver-impact">${d.impact}</div>
                            </div>
                            <div style="font-weight:600; font-size:0.8rem; color:var(--primary);">${d.shap_value ? d.shap_value.toFixed(4) : "N/A"}</div>
                        `;
                        driversDiv.appendChild(item);
                    });
                } else {
                    driversDiv.innerHTML = '<div class="empty-state">No significant risk drivers identified.</div>';
                }

                document.getElementById("prediction-result-container").classList.remove("hidden");
                showToast("Prediction completed successfully.", "success");
            } else {
                showToast(data.detail || "Prediction request failed.", "error");
            }
        } catch (err) {
            showToast("Server connection error during prediction request.", "error");
        } finally {
            setBtnLoading(submitBtn, false);
        }
    });
}

// 3. Batch Analytics
function renderBatch(container) {
    container.innerHTML = `
        <div class="card" style="margin-bottom:1.5rem;">
            <div class="card-header">
                <h3 class="card-title">Bulk CSV Process Engine</h3>
                <span class="card-subtitle">Upload a list of subscribers to get batch action metrics</span>
            </div>
            
            <div class="upload-zone" id="drop-zone">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Drag & drop your CSV file here</div>
                <div class="upload-hint">or click to browse from system</div>
                <input type="file" id="batch-file" accept=".csv">
            </div>
        </div>

        <div id="batch-result-container" class="hidden">
            <div class="metric-grid">
                <div class="metric-card metric-primary">
                    <span class="metric-label">Subscribers Processed</span>
                    <h3 class="metric-value" id="batch-total">0</h3>
                    <p class="metric-change">Total dataset size</p>
                    <div class="metric-icon">👥</div>
                </div>
                <div class="metric-card metric-warning">
                    <span class="metric-label">Mean Churn Risk</span>
                    <h3 class="metric-value" id="batch-mean-risk">0.0%</h3>
                    <p class="metric-change">Average risk value</p>
                    <div class="metric-icon">📈</div>
                </div>
                <div class="metric-card metric-success">
                    <span class="metric-label">Expected Net Profit</span>
                    <h3 class="metric-value" id="batch-net-profit">₹0</h3>
                    <p class="metric-change">Optimized retention return</p>
                    <div class="metric-icon">💰</div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Processed Batch Output</h3>
                    <button class="btn btn-primary btn-sm" id="btn-download-batch">📥 Download CSV</button>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Index</th>
                                <th>Risk Probability</th>
                                <th>Risk Level</th>
                                <th>System Action</th>
                                <th>Expected Net Profit</th>
                            </tr>
                        </thead>
                        <tbody id="batch-table-body">
                            <!-- Rendered dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;

    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("batch-file");
    let batchPredictions = null;

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            processBatchFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            processBatchFile(e.target.files[0]);
        }
    });

    async function processBatchFile(file) {
        showToast(`Uploading ${file.name}...`, "info");
        const formData = new FormData();
        formData.append("file", file);

        const threshold = state.user ? state.user.default_threshold : 0.15;

        try {
            const res = await fetch(`${API_BASE}/predict_batch?threshold=${threshold}`, {
                method: "POST",
                headers: { 
                    "Authorization": `Bearer ${state.token}`
                },
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                batchPredictions = data;
                
                // Set metrics
                document.getElementById("batch-total").textContent = data.total_records;
                document.getElementById("batch-mean-risk").textContent = `${(data.summary.mean_churn_probability * 100).toFixed(1)}%`;
                document.getElementById("batch-net-profit").textContent = `₹${data.summary.total_expected_net_profit.toLocaleString()}`;

                // Set table rows
                const tbody = document.getElementById("batch-table-body");
                tbody.innerHTML = "";
                data.predictions.slice(0, 10).forEach(row => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td>#${row.record_index + 1}</td>
                        <td class="${row.risk_level === 'high' ? 'risk-high' : row.risk_level === 'medium' ? 'risk-medium' : 'risk-low'}" style="font-weight:600;">
                            ${(row.churn_probability * 100).toFixed(1)}%
                        </td>
                        <td><span class="risk-tag ${row.risk_level === 'high' ? 'high' : row.risk_level === 'medium' ? 'medium' : 'low'}">${row.risk_level.toUpperCase()}</span></td>
                        <td style="text-transform: capitalize;">${row.decision}</td>
                        <td>₹${row.expected_profit}</td>
                    `;
                    tbody.appendChild(tr);
                });

                document.getElementById("batch-result-container").classList.remove("hidden");
                showToast("Batch processing complete.", "success");
            } else {
                showToast(data.detail || "Failed to process batch CSV file.", "error");
            }
        } catch(err) {
            showToast("Server connection error during upload.", "error");
        }
    }

    document.getElementById("batch-result-container").addEventListener("click", (e) => {
        if (e.target && e.target.id === "btn-download-batch" && batchPredictions) {
            // Reconstruct a simple downloadable CSV file
            let csvContent = "data:text/csv;charset=utf-8,Index,Churn Probability,Risk Level,System Decision,Expected Profit,Action Quadrant\n";
            batchPredictions.predictions.forEach(p => {
                csvContent += `${p.record_index + 1},${p.churn_probability},${p.risk_level},${p.decision},${p.expected_profit},"${p.action_quadrant || ''}"\n`;
            });
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "batch_churn_predictions.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });
}

// 4. Revenue Forecasting View
function renderForecast(container) {
    container.innerHTML = `
        <div class="grid-2">
            <!-- ARIMA Forecast Input & Plot -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">ARIMA Revenue Time-Series Forecast</h3>
                    <span class="card-subtitle">Generate a 6-month historical/future revenue model</span>
                </div>
                <div style="display:flex; flex-direction:column; gap:1.25rem;">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Forecast Periods (Months)</label>
                            <input type="number" id="fc-steps" min="1" max="12" value="6">
                        </div>
                        <div class="form-group">
                            <label>ARIMA Model Order (p,d,q)</label>
                            <div style="display:flex; gap:0.5rem;">
                                <input type="number" id="fc-p" value="1" min="0" max="3" style="padding:0.5rem;">
                                <input type="number" id="fc-d" value="1" min="0" max="2" style="padding:0.5rem;">
                                <input type="number" id="fc-q" value="1" min="0" max="3" style="padding:0.5rem;">
                            </div>
                        </div>
                    </div>
                    <button class="btn btn-primary" id="btn-run-forecast">📊 Generate Forecast</button>
                    
                    <div id="forecast-output-panel" class="hidden">
                        <div class="chart-container" id="forecast-chart-container">
                            <!-- Custom SVG plot generated by JS -->
                        </div>
                        <div class="table-wrapper" style="margin-top:1.25rem;">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Period</th>
                                        <th>Predicted Revenue</th>
                                        <th>95% Lower Bound</th>
                                        <th>95% Upper Bound</th>
                                    </tr>
                                </thead>
                                <tbody id="forecast-table-body"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Monte Carlo Simulation Panel -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Monte Carlo Revenue Risk Simulator</h3>
                    <span class="card-subtitle">Stochastic simulation of churn rate variance on revenue</span>
                </div>
                <div style="display:flex; flex-direction:column; gap:1.25rem;">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Customer Base Size</label>
                            <input type="number" id="mc-base" value="7043">
                        </div>
                        <div class="form-group">
                            <label>Avg Customer ARPU (₹)</label>
                            <input type="number" id="mc-arpu" value="6000">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Mean Churn Rate</label>
                            <input type="number" id="mc-rate" value="0.27" step="0.01">
                        </div>
                        <div class="form-group">
                            <label>Churn Std Dev</label>
                            <input type="number" id="mc-std" value="0.05" step="0.01">
                        </div>
                    </div>
                    <button class="btn btn-primary" id="btn-run-mc">🎲 Execute Monte Carlo (5000 Sims)</button>

                    <div id="mc-output-panel" class="hidden">
                        <div style="display:flex; justify-content:space-between; margin-bottom:1rem;">
                            <div class="forecast-card" style="flex:1; margin-right:0.5rem;">
                                <h3 id="mc-mean-rev">₹0</h3>
                                <p>Simulated Mean Revenue</p>
                            </div>
                            <div class="forecast-card" style="flex:1; margin-left:0.5rem; background: linear-gradient(135deg, var(--danger-bg) 0%, #1a0505 100%); border-color: var(--danger);">
                                <h3 id="mc-var-rev" style="color:var(--danger);">₹0</h3>
                                <p>Value-at-Risk (5th pct)</p>
                            </div>
                        </div>
                        <h4 style="font-size:0.8rem; text-transform:uppercase; color:var(--text-secondary); margin-bottom:0.5rem;">Simulated Churn Frequency Distribution</h4>
                        <div class="bar-chart" id="mc-chart-container">
                            <!-- SVG/Div bars generated by JS -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.getElementById("btn-run-forecast").addEventListener("click", async () => {
        const steps = document.getElementById("fc-steps").value;
        const p = document.getElementById("fc-p").value;
        const d = document.getElementById("fc-d").value;
        const q = document.getElementById("fc-q").value;
        
        try {
            const res = await fetch(`${API_BASE}/forecast/revenue?steps=${steps}&order_p=${p}&order_d=${d}&order_q=${q}`, {
                method: "POST",
                headers: { "Authorization": `Bearer ${state.token}` }
            });
            const data = await res.json();
            
            if (res.ok) {
                renderForecastChart(data);
                
                const tbody = document.getElementById("forecast-table-body");
                tbody.innerHTML = "";
                data.forecast.forEach(row => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td>Month ${row.period}</td>
                        <td style="font-weight:600;">₹${row.predicted_revenue.toLocaleString()}</td>
                        <td style="color:var(--text-secondary);">₹${row.lower_bound.toLocaleString()}</td>
                        <td style="color:var(--text-secondary);">₹${row.upper_bound.toLocaleString()}</td>
                    `;
                    tbody.appendChild(tr);
                });
                
                document.getElementById("forecast-output-panel").classList.remove("hidden");
                showToast("ARIMA forecast computed.", "success");
            } else {
                showToast(data.detail || "Forecasting service failed.", "error");
            }
        } catch (e) {
            showToast("Server connection error during forecasting.", "error");
        }
    });

    document.getElementById("btn-run-mc").addEventListener("click", async () => {
        const base = document.getElementById("mc-base").value;
        const arpu = document.getElementById("mc-arpu").value;
        const rate = document.getElementById("mc-rate").value;
        const std = document.getElementById("mc-std").value;

        try {
            const res = await fetch(`${API_BASE}/forecast/monte-carlo?n_customers=${base}&avg_revenue=${arpu}&churn_rate_mean=${rate}&churn_rate_std=${std}&n_simulations=5000`, {
                headers: { "Authorization": `Bearer ${state.token}` }
            });
            const data = await res.json();
            
            if (res.ok) {
                document.getElementById("mc-mean-rev").textContent = `₹${data.results.mean_revenue.toLocaleString(undefined, {maximumFractionDigits:0})}`;
                document.getElementById("mc-var-rev").textContent = `₹${data.results.value_at_risk_5pct.toLocaleString(undefined, {maximumFractionDigits:0})}`;

                // Plot custom bar chart
                const chart = document.getElementById("mc-chart-container");
                chart.innerHTML = "";
                const counts = data.histogram_counts;
                const max = Math.max(...counts);
                
                counts.forEach((c, idx) => {
                    const hPercent = (c / max) * 100;
                    const bar = document.createElement("div");
                    bar.className = "bar";
                    bar.style.height = `${hPercent}%`;
                    bar.title = `${c} simulations`;
                    chart.appendChild(bar);
                });

                document.getElementById("mc-output-panel").classList.remove("hidden");
                showToast("Monte Carlo simulation executed.", "success");
            } else {
                showToast("Monte Carlo simulation request failed.", "error");
            }
        } catch (e) {
            showToast("Server connection error during Monte Carlo.", "error");
        }
    });
}

function renderForecastChart(data) {
    const container = document.getElementById("forecast-chart-container");
    container.innerHTML = "";
    
    const hist = data.historical_monthly_revenue || [];
    const fc = data.forecast || [];
    const all = [...hist.map(h => h.revenue), ...fc.map(f => f.predicted_revenue)];
    
    const minVal = Math.min(...all) * 0.95;
    const maxVal = Math.max(...all) * 1.05;
    const valRange = maxVal - minVal;
    
    const width = container.clientWidth || 400;
    const height = 180;
    const pointsCount = all.length;
    const xStep = width / (pointsCount - 1);
    
    let pathD = "";
    let fcPathD = "";
    
    // Draw historical line path
    hist.forEach((h, idx) => {
        const x = idx * xStep;
        const y = height - ((h.revenue - minVal) / valRange) * height;
        if (idx === 0) {
            pathD += `M ${x} ${y}`;
        } else {
            pathD += ` L ${x} ${y}`;
        }
    });

    // Draw forecast line path
    fc.forEach((f, idx) => {
        const globalIdx = hist.length + idx - 1;
        const x = globalIdx * xStep;
        const y = height - ((f.predicted_revenue - minVal) / valRange) * height;
        if (idx === 0) {
            const lastHist = hist[hist.length - 1];
            const startY = height - ((lastHist.revenue - minVal) / valRange) * height;
            fcPathD += `M ${x} ${startY} L ${x} ${y}`;
        } else {
            fcPathD += ` L ${x} ${y}`;
        }
    });

    const svg = `
        <svg viewBox="0 0 ${width} ${height}" style="width:100%; height:100%;">
            <defs>
                <linearGradient id="chart-grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="var(--primary)" stop-opacity="0.3"/>
                    <stop offset="100%" stop-color="var(--primary)" stop-opacity="0"/>
                </linearGradient>
            </defs>
            <path d="${pathD}" fill="none" stroke="var(--text-muted)" stroke-width="2" stroke-dasharray="4"/>
            <path d="${fcPathD}" fill="none" stroke="var(--primary)" stroke-width="3"/>
            ${fc.map((f, idx) => {
                const globalIdx = hist.length + idx - 1;
                const x = globalIdx * xStep;
                const y = height - ((f.predicted_revenue - minVal) / valRange) * height;
                return `<circle cx="${x}" cy="${y}" r="4" fill="var(--primary)" />`;
            }).join('')}
        </svg>
    `;
    container.innerHTML = svg;
}

// 5. MLOps / Drift Monitoring View
function renderMonitor(container) {
    container.innerHTML = `
        <div class="card" style="margin-bottom:1.5rem;">
            <div class="card-header">
                <h3 class="card-title">Live Distribution Drift Diagnostics</h3>
                <span class="card-subtitle">Upload a validation batch CSV to check Kolmogorov-Smirnov (KS) & Population Stability Index (PSI) drift</span>
            </div>

            <div class="upload-zone" id="drift-drop-zone">
                <div class="upload-icon">📉</div>
                <div class="upload-text">Upload production CSV cohort batch to check drift</div>
                <div class="upload-hint">Drag & drop or click to upload validation batch</div>
                <input type="file" id="drift-file" accept=".csv">
            </div>
        </div>

        <div id="drift-result-container" class="hidden">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Feature Drift Analysis Details</h3>
                    <span id="drift-status-badge" class="risk-tag">Stable</span>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>PSI Score</th>
                                <th>KS Statistic</th>
                                <th>P-Value</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="drift-table-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    `;

    const dropZone = document.getElementById("drift-drop-zone");
    const fileInput = document.getElementById("drift-file");

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            evaluateDrift(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            evaluateDrift(e.target.files[0]);
        }
    });

    async function evaluateDrift(file) {
        showToast(`Analyzing drift for ${file.name}...`, "info");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${API_BASE}/monitor/drift`, {
                method: "POST",
                headers: { "Authorization": `Bearer ${state.token}` },
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                // Populate results
                const badge = document.getElementById("drift-status-badge");
                if (data.overall_drift_detected) {
                    badge.textContent = "DRIFT DETECTED";
                    badge.className = "risk-tag high";
                    showToast("🚨 Warning: Significant population distribution drift detected!", "error");
                } else {
                    badge.textContent = "POPULATION STABLE";
                    badge.className = "risk-tag low";
                    showToast("🟢 Stable: No significant population distribution drift detected.", "success");
                }

                const tbody = document.getElementById("drift-table-body");
                tbody.innerHTML = "";
                
                Object.entries(data.feature_metrics).forEach(([feat, m]) => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td style="font-weight:600;">${feat}</td>
                        <td>${m.psi.toFixed(4)}</td>
                        <td>${m.ks_stat.toFixed(4)}</td>
                        <td>${m.p_value.toFixed(6)}</td>
                        <td>
                            <span class="risk-tag ${m.drift_detected ? 'high' : 'low'}">
                                ${m.drift_detected ? 'Drift' : 'Stable'}
                            </span>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });

                document.getElementById("drift-result-container").classList.remove("hidden");
            } else {
                showToast(data.detail || "Failed to analyze drift.", "error");
            }
        } catch(err) {
            showToast("Server connection error during drift check.", "error");
        }
    }
}

// 6. User Profile View
function renderProfile(container) {
    if (!state.user) return;
    
    const init = state.user.full_name.split(" ").map(w => w[0]).join("").substring(0, 2).toUpperCase();

    container.innerHTML = `
        <div class="profile-header">
            <div class="profile-avatar" style="background-color: ${state.user.avatar_color || '#6366f1'};">${init}</div>
            <div class="profile-details">
                <h2>${state.user.full_name}</h2>
                <p>${state.user.email}</p>
                <div class="profile-meta">
                    <span>Role: <strong style="text-transform: capitalize; color:var(--text);">${state.user.role}</strong></span>
                    <span>Company: <strong style="color:var(--text);">${state.user.company || "N/A"}</strong></span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Edit User Credentials</h3>
                <span class="card-subtitle">Keep your organizational credentials and contact detail profile up-to-date</span>
            </div>
            <form id="profile-edit-form" style="display:flex; flex-direction:column; gap:1.25rem;">
                <div class="form-row">
                    <div class="form-group">
                        <label>Full Name</label>
                        <input type="text" id="prof-name" value="${state.user.full_name}" required>
                    </div>
                    <div class="form-group">
                        <label>Company / Organization</label>
                        <input type="text" id="prof-company" value="${state.user.company || ''}">
                    </div>
                    <div class="form-group">
                        <label>Functional Role</label>
                        <select id="prof-role">
                            <option value="analyst" ${state.user.role === 'analyst' ? 'selected' : ''}>Analyst</option>
                            <option value="manager" ${state.user.role === 'manager' ? 'selected' : ''}>Manager</option>
                            <option value="admin" ${state.user.role === 'admin' ? 'selected' : ''}>Administrator</option>
                            <option value="developer" ${state.user.role === 'developer' ? 'selected' : ''}>Developer</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="btn btn-primary" style="align-self: flex-start;">Update Profile Details</button>
            </form>
        </div>
    `;

    document.getElementById("profile-edit-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const full_name = document.getElementById("prof-name").value;
        const company = document.getElementById("prof-company").value;
        const role = document.getElementById("prof-role").value;

        try {
            const res = await fetch(`${API_BASE}/auth/profile`, {
                method: "PUT",
                headers: { 
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${state.token}`
                },
                body: JSON.stringify({ full_name, company, role })
            });
            const data = await res.json();
            
            if (res.ok) {
                state.user = data.user;
                updateUserUI();
                navigateTo("profile");
                showToast("Profile details updated successfully.", "success");
            } else {
                showToast(data.detail || "Failed to update profile.", "error");
            }
        } catch(err) {
            showToast("Server connection error during profile update.", "error");
        }
    });
}

// 7. System Settings View
function renderSettings(container) {
    if (!state.user) return;

    container.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Platform Preferences</h3>
                <span class="card-subtitle">Manage classification sensitivity, interface skin, and alerting notification hooks</span>
            </div>
            
            <div style="display:flex; flex-direction:column; gap:1.25rem;">
                <div class="setting-item">
                    <div class="setting-label">
                        <h4>Default Classification Threshold</h4>
                        <p>Lowering increases retention capture targets (P1/P2 sensitivity)</p>
                    </div>
                    <div style="display:flex; align-items:center; gap:1rem; width: 300px;">
                        <input type="range" id="sett-thresh" min="0.05" max="0.50" step="0.01" value="${state.user.default_threshold}" style="flex:1;">
                        <span id="sett-thresh-val" style="font-weight:700; width:40px; text-align:right;">${state.user.default_threshold.toFixed(2)}</span>
                    </div>
                </div>

                <div class="setting-item">
                    <div class="setting-label">
                        <h4>Dark Theme Interface Skin</h4>
                        <p>Toggle high-contrast premium dark space colors</p>
                    </div>
                    <button class="toggle ${state.user.dark_mode ? 'active' : ''}" id="sett-dark-mode" type="button"></button>
                </div>

                <div class="setting-item">
                    <div class="setting-label">
                        <h4>Real-Time Drift Webhook Trigger</h4>
                        <p>Receive immediate alerts on server-side population shift flags</p>
                    </div>
                    <button class="toggle ${state.user.notifications_enabled ? 'active' : ''}" id="sett-notif" type="button"></button>
                </div>

                <button class="btn btn-primary" id="btn-save-settings" style="align-self: flex-start; margin-top:1rem;">Save Preferences</button>
            </div>
        </div>
    `;

    // Sliders
    const slider = document.getElementById("sett-thresh");
    const val = document.getElementById("sett-thresh-val");
    slider.addEventListener("input", (e) => {
        val.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Toggles
    const toggles = document.querySelectorAll(".toggle");
    toggles.forEach(t => {
        t.addEventListener("click", () => {
            t.classList.toggle("active");
        });
    });

    // Save Settings Event
    document.getElementById("btn-save-settings").addEventListener("click", async () => {
        const threshold = parseFloat(slider.value);
        const darkMode = document.getElementById("sett-dark-mode").classList.contains("active");
        const notifications = document.getElementById("sett-notif").classList.contains("active");

        try {
            const res = await fetch(`${API_BASE}/auth/settings`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${state.token}`
                },
                body: JSON.stringify({
                    default_threshold: threshold,
                    dark_mode: darkMode,
                    notifications_enabled: notifications
                })
            });
            const data = await res.json();
            
            if (res.ok) {
                state.user = data.user;
                updateUserUI();
                navigateTo("settings");
                showToast("Platform configurations saved.", "success");
            } else {
                showToast(data.detail || "Failed to update configurations.", "error");
            }
        } catch(err) {
            showToast("Server connection error during settings save.", "error");
        }
    });
}
