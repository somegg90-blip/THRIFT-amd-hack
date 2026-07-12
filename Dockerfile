FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (smaller than GPU version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment variables - CHANGE THESE TO TEST LOCAL MODEL
ENV THRIFT_SKIP_TIER_1=true
ENV THRIFT_USE_4BIT=true
ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]