@echo off
REM Local Development Runner for Automatic Labeler Backend (Windows)

setlocal enabledelayedexpansion

echo =========================================
echo Starting Automatic Labeler Backend (Local)
echo =========================================

cd /d "%~dp0\.."

REM Load environment variables from .env.local
if exist .env.local (
    echo Loading .env.local...
    for /f "usebackq tokens=*" %%a in (".env.local") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            set "%%a"
        )
    )
) else (
    echo Warning: .env.local not found, using defaults
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run migrations
echo Running database migrations...
python manage.py migrate

REM Collect static files
echo Collecting static files...
python manage.py collectstatic --noinput

REM Start development server
if "%PORT%"=="" set PORT=8000
echo Starting development server on port %PORT%...
python manage.py runserver 0.0.0.0:%PORT%

endlocal
