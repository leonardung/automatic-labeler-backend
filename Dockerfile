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
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Base GPU stage
# ============================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base-gpu

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
    python3-setuptools \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    cmake \
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

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Copy application code
COPY . .

RUN pip install -r submodules/PaddleOCR/requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]

# ============================================
# Development GPU stage
# ============================================
FROM base-gpu as development-gpu

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Copy application code
COPY . .

RUN pip install -r submodules/PaddleOCR/requirements.txt

RUN pip install -r requirements_no_version.txt

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]

# ============================================
# Production CPU stage
# ============================================
FROM base-cpu as production-cpu

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Copy application code
COPY . .

RUN pip install -r submodules/PaddleOCR/requirements.txt

COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install gunicorn==21.2.0

# Create necessary directories with proper permissions
RUN mkdir -p /app/media /app/staticfiles && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /opt/venv

# Switch to non-root user
USER appuser

EXPOSE 8000

# Use gunicorn for production
CMD ["sh", "-c", "python manage.py collectstatic --noinput && python manage.py migrate && gunicorn image_labeling_backend.wsgi:application --bind 0.0.0.0:8000 --workers 4 --timeout 120 --access-logfile - --error-logfile -"]

# ============================================
# Production GPU stage
# ============================================
FROM base-gpu as production-gpu

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install Python dependencies with CUDA support
RUN pip install --upgrade pip && \
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Copy application code
COPY . .

RUN pip install -r submodules/PaddleOCR/requirements.txt

COPY requirements_no_version.txt .
RUN pip install -r requirements_no_version.txt && \
    pip install gunicorn==21.2.0

# Create necessary directories with proper permissions
RUN mkdir -p /app/media /app/staticfiles && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /opt/venv

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
