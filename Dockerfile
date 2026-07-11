# Removed --platform=linux/amd64 from FROM to fix the warning
FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first. 
# This prevents requirements.txt from accidentally pulling the massive 2GB+ GPU version.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-dotenv requests

# Copy source code
COPY . .

# CRITICAL: We are skipping the local model (Tier 1) to guarantee speed on 2 vCPU.
# Because we skip it, we DO NOT need to download the model weights!
# This keeps the image small and the build fast.
ENV THRIFT_SKIP_TIER_1=true
ENV PYTHONUNBUFFERED=1

# Entry point
CMD ["python", "run.py"]