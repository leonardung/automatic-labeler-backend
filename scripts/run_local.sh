#!/bin/bash

# Local Development Runner for Automatic Labeler Backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Starting Automatic Labeler Backend (Local)"
echo "========================================="

cd "$PROJECT_DIR"

# Load local environment variables
if [ -f .env.local ]; then
    echo "Loading .env.local..."
    export $(cat .env.local | grep -v '^#' | xargs)
else
    echo "Warning: .env.local not found, using defaults"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run migrations
echo "Running database migrations..."
python manage.py migrate

# Create superuser if needed (optional)
# python manage.py createsuperuser --noinput || true

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Start development server
echo "Starting development server on port ${PORT:-8000}..."
python manage.py runserver "0.0.0.0:${PORT:-8000}"
