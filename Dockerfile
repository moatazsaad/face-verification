FROM python:3.11-slim

WORKDIR /app

# System packages commonly needed by Pillow / InsightFace / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency file first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project
COPY . /app

# Create artifact directory in case your code expects it
RUN mkdir -p /app/artifacts

# Default command runs your CLI
ENTRYPOINT ["python", "-m", "src.run_inference_cli"]