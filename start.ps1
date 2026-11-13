<#
Starts the Agent Configurator on this machine:

  Langflow 1.13.2 with the CoworkAI blocks   http://127.0.0.1:7860
  Hand-off service and page                  http://127.0.0.1:8003
  Agent Evaluator API and UI (if the repo is next to this one)
                                             http://127.0.0.1:8002 and http://127.0.0.1:5174

The LLM Configurator (coworkAI, port 8001) is not started or changed by this script.
Start it yourself, or pass -Mock to run a stand-in that needs no provider keys.

  .\start.ps1                 use the real LLM Configurator on LLM_CONFIGURATOR_URL
  .\start.ps1 -Mock           use the stand-in configurator on port 8001
  .\start.ps1 -NoEvaluator    do not start the Agent Evaluator
  .\start.ps1 -NoBlockPolicy  leave Langflow's own model blocks visible

Press Ctrl+C to stop everything this script started.
#>
param(
    [switch]$Mock,
    [switch]$NoEvaluator,
    [switch]$NoBlockPolicy
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$langflow = Join-Path $root ".venv\Scripts\langflow.exe"
$logs = Join-Path $root ".langflow\logs"
$evaluatorRoot = Join-Path (Split-Path -Parent $root) "Agent-Evaluation"
$started = @()

function Write-Step($text) { Write-Host $text -ForegroundColor Cyan }

function Test-Url($url) {
    try {
        Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        # A reply with an error status still means something is listening.
        return ($null -ne $_.Exception.Response)
    }
}

function Wait-Url($url, $name, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Url $url) { return }
        Start-Sleep -Seconds 2
    }
    throw "$name did not answer on $url within $seconds seconds. See the logs in $logs."
}

function Start-Background($name, $file, $arguments, $workingDirectory) {
    $out = Join-Path $logs "$name.log"
    $err = Join-Path $logs "$name.err.log"
    $process = Start-Process -FilePath $file -ArgumentList $arguments -WorkingDirectory $workingDirectory `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    $script:started += $process
    return $process
}

function Stop-Started {
    foreach ($process in $script:started) {
        if ($process -and -not $process.HasExited) {
            # /T also stops child processes (Langflow and npm start workers).
            & taskkill /F /T /PID $process.Id 2>$null | Out-Null
        }
    }
}

try {
    New-Item -ItemType Directory -Force -Path $logs | Out-Null

    # 1. Python environment
    if (-not (Test-Path -LiteralPath $langflow)) {
        Write-Step "Installing Langflow 1.13.2 and the blocks (first run only, takes several minutes)..."
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            & uv venv --python 3.12 (Join-Path $root ".venv")
            & uv pip install --python (Join-Path $root ".venv") -r (Join-Path $root "requirements.txt")
        } else {
            & py -3.12 -m venv (Join-Path $root ".venv")
            Push-Location $root
            try { & $python -m pip install -r requirements.txt } finally { Pop-Location }
        }
        if (-not (Test-Path -LiteralPath $langflow)) { throw "Installing Langflow failed. See the output above." }
    }

    # 2. Settings
    $envFile = Join-Path $root ".env"
    if (-not (Test-Path -LiteralPath $envFile)) {
        Copy-Item -LiteralPath (Join-Path $root ".env.example") -Destination $envFile
        Write-Host "Created .env from .env.example." -ForegroundColor Yellow
    }
    Get-Content -LiteralPath $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim(), "Process")
        }
    }
    if ($Mock) { $env:LLM_CONFIGURATOR_URL = "http://127.0.0.1:8001" }
    if (-not $env:LLM_CONFIGURATOR_URL) { $env:LLM_CONFIGURATOR_URL = "http://127.0.0.1:8001" }
    if (-not $env:LANGFLOW_URL) { $env:LANGFLOW_URL = "http://127.0.0.1:7860" }
    if (-not $env:EVALUATOR_API_URL) { $env:EVALUATOR_API_URL = "http://127.0.0.1:8002/api" }
    if (-not $env:EVALUATOR_UI_URL) { $env:EVALUATOR_UI_URL = "http://127.0.0.1:5174" }

    # 3. LLM Configurator: the real one is never started or changed from here.
    if ($Mock) {
        if (Test-Url "$($env:LLM_CONFIGURATOR_URL)/configs") {
            throw "Port 8001 is already in use. Stop what is running there, or start without -Mock to use it."
        }
        Write-Step "Starting the stand-in LLM Configurator on port 8001 (scripted replies, no provider keys)..."
        Start-Background "mock-configurator" $python @("-m", "uvicorn", "tests.mock_configurator:app", "--host", "127.0.0.1", "--port", "8001", "--log-level", "warning") $root | Out-Null
        Wait-Url "$($env:LLM_CONFIGURATOR_URL)/configs" "The stand-in LLM Configurator" 60
    } elseif (-not (Test-Url "$($env:LLM_CONFIGURATOR_URL)/configs")) {
        Write-Host "The LLM Configurator is not answering on $($env:LLM_CONFIGURATOR_URL)." -ForegroundColor Yellow
        Write-Host "  Start it from coworkAI\llm-configurator\backend with:  uvicorn main:app --port 8001" -ForegroundColor Yellow
        Write-Host "  (the coworkAI core platform must have run once to create ENCRYPTION_KEY), or use  .\start.ps1 -Mock" -ForegroundColor Yellow
        Write-Host "  Continuing: the blocks will show an error until it is running." -ForegroundColor Yellow
    }

    # 4. Langflow
    if (Test-Url "$($env:LANGFLOW_URL)/health") {
        Write-Host "Langflow is already running on $($env:LANGFLOW_URL). Using it. Restart it to pick up changed blocks." -ForegroundColor Yellow
    } else {
        Write-Step "Starting Langflow (the first start runs database migrations and takes a few minutes)..."
        $env:LANGFLOW_AUTO_LOGIN = "true"
        $env:LANGFLOW_CONFIG_DIR = Join-Path $root ".langflow"
        $env:LANGFLOW_COMPONENTS_PATH = Join-Path $root "components"
        $env:LANGFLOW_DO_NOT_TRACK = "true"
        $env:DO_NOT_TRACK = "true"
        $env:PYTHONUTF8 = "1"
        $env:PYTHONIOENCODING = "utf-8"
        Start-Background "langflow" $langflow @("run", "--host", "127.0.0.1", "--port", "7860", "--no-open-browser") $root | Out-Null
        Wait-Url "$($env:LANGFLOW_URL)/health" "Langflow" 900
    }

    # 5. Langflow API key (kept in .env) and block policy
    & $python (Join-Path $root "scripts\create_langflow_key.py") --url $env:LANGFLOW_URL --env $envFile
    if ($LASTEXITCODE -ne 0) { throw "Could not create a Langflow API key." }
    if (-not $NoBlockPolicy) {
        & $python (Join-Path $root "scripts\set_block_policy.py") --url $env:LANGFLOW_URL --env $envFile
    }
    # The hand-off service reads LANGFLOW_API_KEY from .env itself; do not keep an older value in this process.
    Remove-Item Env:LANGFLOW_API_KEY -ErrorAction SilentlyContinue

    # 6. Agent Evaluator (a separate repo next to this one)
    if (-not $NoEvaluator) {
        $evaluatorPython = Join-Path $evaluatorRoot "backend\.venv\Scripts\python.exe"
        if (Test-Url "http://127.0.0.1:8002/health") {
            Write-Host "Agent Evaluator API is already running on port 8002. Using it." -ForegroundColor Yellow
        } elseif (Test-Path -LiteralPath $evaluatorPython) {
            Write-Step "Starting the Agent Evaluator API on port 8002..."
            Start-Background "evaluator-api" $evaluatorPython @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8002") (Join-Path $evaluatorRoot "backend") | Out-Null
            Wait-Url "http://127.0.0.1:8002/health" "The Agent Evaluator API" 120
        } else {
            Write-Host "Agent Evaluator is not set up at $evaluatorRoot. Run its start.ps1 once, then start this again." -ForegroundColor Yellow
        }
        $evaluatorFrontend = Join-Path $evaluatorRoot "frontend"
        if (Test-Url $env:EVALUATOR_UI_URL) {
            Write-Host "Agent Evaluator UI is already running on $($env:EVALUATOR_UI_URL). Using it." -ForegroundColor Yellow
        } elseif (Test-Path -LiteralPath (Join-Path $evaluatorFrontend "node_modules")) {
            # 5174, because the coworkAI frontend needs 5173 (the LLM Configurator only accepts that origin).
            Write-Step "Starting the Agent Evaluator UI on port 5174..."
            Start-Background "evaluator-ui" "cmd.exe" @("/c", "npm", "run", "dev", "--", "--port", "5174", "--strictPort") $evaluatorFrontend | Out-Null
        } elseif (Test-Path -LiteralPath $evaluatorFrontend) {
            Write-Host "Agent Evaluator UI is not installed (no node_modules). Run 'npm install' in $evaluatorFrontend." -ForegroundColor Yellow
        }
    }

    # 7. Hand-off page: build it once
    $dist = Join-Path $root "frontend\dist\index.html"
    if (-not (Test-Path -LiteralPath $dist)) {
        if (Get-Command npm -ErrorAction SilentlyContinue) {
            Write-Step "Building the hand-off page..."
            Push-Location (Join-Path $root "frontend")
            try {
                if (-not (Test-Path -LiteralPath "node_modules")) { & cmd /c npm install --no-audit --no-fund }
                & cmd /c npm run build
            } finally { Pop-Location }
        } else {
            Write-Host "npm was not found, so the hand-off page is not built. Its API still works on port 8003." -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "Build agents:        $($env:LANGFLOW_URL)   (blocks are under 'Coworkai' in the sidebar)" -ForegroundColor Green
    Write-Host "Evaluate an agent:   http://127.0.0.1:8003" -ForegroundColor Green
    Write-Host "Agent Evaluator:     $($env:EVALUATOR_UI_URL)" -ForegroundColor Green
    Write-Host "LLM Configurator:    $($env:LLM_CONFIGURATOR_URL)$(if ($Mock) { '   (stand-in)' })" -ForegroundColor Green
    Write-Host "Starter templates:   $root\templates  (import them from Langflow's flow list)" -ForegroundColor Green
    Write-Host "Logs:                $logs" -ForegroundColor Green
    Write-Host ""
    Write-Host "Press Ctrl+C to stop." -ForegroundColor Green

    # 8. Hand-off service in the foreground
    Push-Location $root
    try {
        & $python -m uvicorn agent_configurator.handoff.app:app --host 127.0.0.1 --port 8003 --log-level warning
    } finally { Pop-Location }
} finally {
    Stop-Started
}
