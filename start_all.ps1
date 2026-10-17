# ==============================================================================
# VoiceShield Full-Stack Startup Script
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [VoiceShield] Launching Full-Stack Services...          " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start Backend Process
Write-Host "[1/2] Launching Backend Service (FastAPI)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-File", (Join-Path $ScriptDir "start_backend.ps1")

# Start Frontend Process
Write-Host "[2/2] Launching Frontend Service (Vite)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-File", (Join-Path $ScriptDir "start_frontend.ps1")

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " VoiceShield Services Successfully Dispatched!            " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " • Frontend UI:   http://localhost:3000" -ForegroundColor Cyan
Write-Host " • Backend API:   http://localhost:8000" -ForegroundColor Cyan
Write-Host " • Swagger Docs:  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host " • OpenAPI Spec:  http://localhost:8000/openapi.json" -ForegroundColor Cyan
Write-Host "==========================================================`n" -ForegroundColor Green
