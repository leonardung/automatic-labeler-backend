# Django Backend Dockerfile WITHOUT PaddleOCR
# PaddleOCR operations are now handled by the GPU microservice

# ============================================
# Base stage
# ============================================
FROM python:3.12-slim as base

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
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Development stage
# ============================================
FROM base as development

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install additional development tools
RUN pip install ipython ipdb

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run migrations and start development server
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]

# ============================================
# Production stage
# ============================================
FROM base as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn==21.2.0

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/media /app/staticfiles /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

# Collect static files, run migrations, and start with gunicorn
CMD ["sh", "-c", "python manage.py collectstatic --noinput && python manage.py migrate && gunicorn image_labeling_backend.wsgi:application --bind 0.0.0.0:8000 --workers 4 --timeout 120 --access-logfile - --error-logfile -"]
