# Multi-stage production Dockerfile for Telecom Churn & Revenue Intelligence Platform
FROM python:3.10-slim

# Create non-root user for security
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --create-home appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . .

# Ensure logs and models directories exist
RUN mkdir -p /app/logs /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command uses gunicorn with uvicorn workers
CMD ["gunicorn", "api:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
