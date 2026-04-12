FROM python:3.10

WORKDIR /app

COPY . .

# Install system dependencies 
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Install python deps
RUN uv sync

# CMD ["python", "-m", "src.run_inference_cli"]
ENTRYPOINT ["/app/.venv/bin/python", "-m", "src.run_inference_cli"]