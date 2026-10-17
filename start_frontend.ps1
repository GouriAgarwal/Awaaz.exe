# ==============================================================================
# VoiceShield Frontend Startup Script (React + Vite)
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [VoiceShield] Initializing Frontend Application...      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendDir = Join-Path $ScriptDir "frontend"
$PortableNode = Join-Path $ScriptDir "..\nodejs"

# 1. Ensure Node / npm is on PATH
$NodeCmd = (Get-Command node -ErrorAction SilentlyContinue)
if (-not $NodeCmd) {
    if (Test-Path (Join-Path $PortableNode "node.exe")) {
        Write-Host "[INFO] Detected local Node.js environment in workspace. Adding to PATH..." -ForegroundColor Yellow
        $env:PATH = "$PortableNode;" + $env:PATH
    } else {
        Write-Host "[ERROR] Node.js is not installed or not found in PATH." -ForegroundColor Red
        exit 1
    }
}

# 2. Check Node modules
Set-Location $FrontendDir
if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Write-Host "[INFO] node_modules not found. Running npm install..." -ForegroundColor Yellow
    npm install
}

# 3. Start Vite Development Server
Write-Host "[INFO] Starting Vite Frontend on http://0.0.0.0:3000" -ForegroundColor Green
Write-Host "[INFO] Access URL: http://localhost:3000" -ForegroundColor Yellow

try {
    npm run dev -- --host 0.0.0.0 --port 3000
}
catch {
    Write-Host "[ERROR] Frontend server exited with an error: $_" -ForegroundColor Red
    exit 1
}
