# =============================================================================
# Dockerfile: ES Training for Beam Steering Optimization
# =============================================================================
# Base image: Python 3.11 slim (minimal, suitable for AWS Batch)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (minimal for slim image)
# procps provides pgrep/pkill needed by joblib for parallel processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy project code
COPY . .

# Default entrypoint: Can be overridden by AWS Batch job definition
# Usage: docker run cs229-trainer python src/scripts/train_90deg.py
CMD ["python3", "-u", "src/scripts/train_90deg.py"]
