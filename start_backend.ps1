# ==============================================================================
# VoiceShield Backend Startup Script (FastAPI + Uvicorn)
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [VoiceShield] Initializing Backend Service...           " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ScriptDir "backend"
$VenvDir = Join-Path $BackendDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ReqFile = Join-Path $BackendDir "requirements.txt"

# 1. Verify Python availability
$SystemPython = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $SystemPython -and -not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Python was not found in PATH. Please install Python 3.10+ and add it to your PATH." -ForegroundColor Red
    exit 1
}

# 2. Check / Create Virtual Environment
if (-not (Test-Path $VenvPython)) {
    Write-Host "[INFO] Python virtual environment not found. Creating .venv in backend/..." -ForegroundColor Yellow
    try {
        Set-Location $BackendDir
        python -m venv .venv
        if (-not (Test-Path $VenvPython)) {
            throw "Virtual environment creation failed. $VenvPython does not exist."
        }
        Write-Host "[SUCCESS] Virtual environment created." -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Failed to create virtual environment: $_" -ForegroundColor Red
        exit 1
    }

    # Install requirements in new venv
    Write-Host "[INFO] Installing backend dependencies from requirements.txt..." -ForegroundColor Yellow
    try {
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r $ReqFile
        Write-Host "[SUCCESS] Dependencies successfully installed." -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Failed to install requirements: $_" -ForegroundColor Red
        exit 1
    }
}

# 3. Start FastAPI Application
Set-Location $BackendDir
Write-Host "[INFO] Using Python: $VenvPython" -ForegroundColor Cyan
Write-Host "[INFO] Starting VoiceShield API on http://0.0.0.0:8000" -ForegroundColor Green
Write-Host "[INFO] Swagger Docs: http://localhost:8000/docs" -ForegroundColor Yellow

try {
    & $VenvPython -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}
catch {
    Write-Host "[ERROR] Backend server exited with an error: $_" -ForegroundColor Red
    exit 1
}
