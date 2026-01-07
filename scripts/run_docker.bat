@echo off
REM Docker Development Runner for Automatic Labeler Backend (Windows)

echo =========================================
echo Starting Automatic Labeler Backend (Docker)
echo =========================================

cd /d "%~dp0\.."

REM Check if .env.docker exists, copy to .env if it does
if exist .env.docker (
    echo Copying .env.docker to .env...
    copy /Y .env.docker .env
) else (
    echo Warning: .env.docker not found
)

REM Build and start Docker containers
echo Building and starting Docker containers...
docker compose up --build

REM Note: Use docker compose down to stop containers
REM Note: Use docker compose up -d to run in detached mode
