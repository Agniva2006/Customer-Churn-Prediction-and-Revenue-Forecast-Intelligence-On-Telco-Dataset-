"""Central configuration for the Telecom Churn & Revenue Intelligence system.

All business parameters and runtime settings are sourced from environment
variables (see ``.env.example``) with sensible production defaults, so no
magic numbers are hardcoded in the API or training code.
"""

import os

# ==============================
# Business Constants
# ==============================
# Production classification threshold (probability above which we act).
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.15"))

# Cost of one retention offer (e.g., discount / concierge package).
RETENTION_COST = float(os.environ.get("RETENTION_COST", "500"))

# Fraction of churned revenue saved by a successful retention offer.
SAVE_RATE = float(os.environ.get("SAVE_RATE", "0.6"))

# Default Customer Lifetime Value used by causal engines when not supplied.
DEFAULT_CLV = float(os.environ.get("DEFAULT_CLV", "850"))

# ==============================
# Monte Carlo Defaults
# ==============================
MC_N_CUSTOMERS = int(os.environ.get("MC_N_CUSTOMERS", "7043"))
MC_AVG_REVENUE = float(os.environ.get("MC_AVG_REVENUE", "6000"))
MC_CHURN_MEAN = float(os.environ.get("MC_CHURN_MEAN", "0.27"))
MC_CHURN_STD = float(os.environ.get("MC_CHURN_STD", "0.05"))
MC_N_SIMULATIONS = int(os.environ.get("MC_N_SIMULATIONS", "5000"))

# ==============================
# Auth
# ==============================
AUTH_TOKEN_TTL_HOURS = int(os.environ.get("AUTH_TOKEN_TTL_HOURS", "72"))
AUTH_PBKDF2_ITERATIONS = int(os.environ.get("AUTH_PBKDF2_ITERATIONS", "200000"))

# ==============================
# App
# ==============================
API_VERSION = "3.2.0"
