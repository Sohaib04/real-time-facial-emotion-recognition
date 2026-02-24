# ──────────────────────────────────────────────
# Multi-stage Docker build for Emotion Detection
# ──────────────────────────────────────────────
FROM python:3.10-slim AS base

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create required directories
RUN mkdir -p logs screenshots model/weights data

# Default: training mode
# Override with: docker run <image> python app/main.py
CMD ["python", "model/train.py"]
