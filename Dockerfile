# Multi-stage build for production-ready backend with GPU support
# ============================================
# Base CPU stage - Using official PaddlePaddle image
# ============================================
FROM paddlepaddle/paddle:3.0.0 as base-cpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install additional system dependencies needed for the application
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    cmake \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Base GPU stage - Using official PaddlePaddle image
# ============================================
FROM paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 as base-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install additional system dependencies needed for the application
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    cmake \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Development CPU stage
# ============================================
FROM base-cpu as development-cpu

# PaddlePaddle CPU is already installed in the base image

# Copy application code
COPY . .

# Install PaddleOCR and other dependencies
RUN pip install --upgrade pip && \
    pip install -r submodules/PaddleOCR/requirements.txt && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]

# ============================================
# Development GPU stage
# ============================================
FROM base-gpu as development-gpu

# PaddlePaddle GPU is already installed in the base image

# Copy application code
COPY . .

# Install PaddleOCR and other dependencies
RUN pip install --upgrade pip && \
    pip install -r submodules/PaddleOCR/requirements.txt && \
    pip install -r requirements_no_version.txt

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]

# ============================================
# Production CPU stage
# ============================================
FROM base-cpu as production-cpu

# PaddlePaddle CPU is already installed in the base image

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r submodules/PaddleOCR/requirements.txt && \
    pip install -r requirements.txt && \
    pip install gunicorn==21.2.0

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

# PaddlePaddle GPU is already installed in the base image

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r submodules/PaddleOCR/requirements.txt && \
    pip install -r requirements_no_version.txt && \
    pip install gunicorn==21.2.0

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
