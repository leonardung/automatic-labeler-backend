#!/bin/bash

# Docker Development Runner for Automatic Labeler Backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Starting Automatic Labeler Backend (Docker)"
echo "========================================="

cd "$PROJECT_DIR"

# Check if .env.docker exists, copy to .env if it does
if [ -f .env.docker ]; then
    echo "Copying .env.docker to .env..."
    cp .env.docker .env
else
    echo "Warning: .env.docker not found"
fi

# Build and start Docker containers
echo "Building and starting Docker containers..."
docker compose up --build

# Note: Use docker compose down to stop containers
# Note: Use docker compose up -d to run in detached mode
