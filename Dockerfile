# Multi-stage build for production-ready backend with GPU support
# ============================================
# Base CPU stage
# ============================================
FROM python:3.12-slim as base-cpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Base GPU stage
# ============================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as base-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    curl \
    wget \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Development CPU stage
# ============================================
FROM base-cpu as development-cpu

ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]

# ============================================
# Development GPU stage
# ============================================
FROM base-gpu as development-gpu

ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# Install Python dependencies with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]

# ============================================
# Production CPU stage
# ============================================
FROM base-cpu as production-cpu

ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==21.2.0

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/media /app/staticfiles && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# Use gunicorn for production
CMD ["sh", "-c", "python manage.py collectstatic --noinput && python manage.py migrate && gunicorn image_labeling_backend.wsgi:application --bind 0.0.0.0:8000 --workers 4 --timeout 120 --access-logfile - --error-logfile -"]

# ============================================
# Production GPU stage
# ============================================
FROM base-gpu as production-gpu

ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install Python dependencies with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==21.2.0

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/media /app/staticfiles && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# Use gunicorn for production
CMD ["sh", "-c", "python manage.py collectstatic --noinput && python manage.py migrate && gunicorn image_labeling_backend.wsgi:application --bind 0.0.0.0:8000 --workers 4 --timeout 120 --access-logfile - --error-logfile -"]

# ============================================
# Default aliases for backward compatibility
# ============================================
FROM development-cpu as development
FROM production-cpu as production
