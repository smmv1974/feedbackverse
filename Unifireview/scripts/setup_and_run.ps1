# ========================================
# Objective 2 - Setup and Run Helper
# ========================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Section($title) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host $title -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Info($message)  { Write-Host ("   [INFO] {0}" -f $message) -ForegroundColor Cyan }
function Write-Ok($message)    { Write-Host ("   [ OK ] {0}" -f $message) -ForegroundColor Green }
function Write-Warn($message)  { Write-Host ("   [WARN] {0}" -f $message) -ForegroundColor Yellow }
function Write-Fail($message)  { Write-Host ("   [FAIL] {0}" -f $message) -ForegroundColor Red }

Write-Section "Objective 2 Pipeline - Setup & Run"

# Step 1: GPU availability
Write-Host "Step 1: Checking GPU..." -ForegroundColor Yellow
$useGPU = $false
try {
    $gpu = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($gpu) {
        Write-Ok ("Detected GPU: {0}" -f $gpu.Trim())
        $useGPU = $true
    } else {
        Write-Warn "No GPU reported by nvidia-smi. Falling back to cloud models."
    }
}
catch {
    Write-Warn "nvidia-smi not available. Falling back to cloud models."
}

# Step 2: Ollama CLI
Write-Host "`nStep 2: Checking Ollama CLI..." -ForegroundColor Yellow
try {
    $ollamaVersion = ollama --version 2>$null
    if (-not $ollamaVersion) {
        throw "ollama command returned no output."
    }
    Write-Ok ("Ollama CLI found: {0}" -f $ollamaVersion.Trim())
}
catch {
    Write-Fail "Ollama is not installed or not on PATH. Install from https://ollama.com then rerun."
    exit 1
}

# Step 3: Ensure Ollama server is running
Write-Host "`nStep 3: Checking Ollama server..." -ForegroundColor Yellow
function Test-OllamaServer {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get -TimeoutSec 5 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

if (Test-OllamaServer) {
    Write-Ok "Ollama server is responding."
}
else {
    Write-Warn "Ollama server not reachable. Starting a local instance..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 3
    if (Test-OllamaServer) {
        Write-Ok "Ollama server started."
    } else {
        Write-Fail "Unable to reach Ollama server at http://127.0.0.1:11434."
        exit 1
    }
}

# Step 4: Select and test model
Write-Host "`nStep 4: Selecting model..." -ForegroundColor Yellow
$modelName = $null

if ($useGPU) {
    $modelName = "qwen2.5:3b-instruct"
    Write-Info ("GPU detected; targeting local model {0}" -f $modelName)
} else {
    $modelName = "deepseek-v3.1:671b-cloud"
    Write-Warn ("Running in cloud fallback mode: {0}" -f $modelName)
}

function Ensure-ModelPresent {
    param([string]$Name)
    $modelExists = ollama list 2>$null | Select-String -Pattern ([regex]::Escape($Name))
    if (-not $modelExists) {
        Write-Info ("Pulling model {0}..." -f $Name)
        ollama pull $Name
    } else {
        Write-Ok ("Model {0} already available." -f $Name)
    }
}

function Test-Model {
    param([string]$Name)
    try {
        $null = ollama run $Name "Hello" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "ollama run returned exit code $LASTEXITCODE"
        }
        return $true
    }
    catch {
        Write-Warn ("Model {0} test failed: {1}" -f $Name, $_)
        return $false
    }
}

if ($modelName -like "*cloud") {
    Write-Info "Skipping local pull for cloud model."
} else {
    Ensure-ModelPresent -Name $modelName
    if (-not (Test-Model -Name $modelName)) {
        Write-Warn "Switching to reliable cloud model because local GPU test failed."
        $useGPU = $false
        $modelName = "deepseek-v3.1:671b-cloud"
    }
}

Write-Ok ("Selected model: {0}" -f $modelName)

# Step 5: Update config
Write-Host "`nStep 5: Updating config/config.yaml..." -ForegroundColor Yellow
$configPath = "config\config.yaml"
if (-not (Test-Path $configPath)) {
    Write-Fail ("Cannot find {0}. Aborting." -f $configPath)
    exit 1
}

$config = Get-Content $configPath -Raw
$patterns = @(
    'name: "[^"]*"  # (summarization|sentiment|themes|alt_small)',
    'name: "heuristic"',
    'name: "llama3\.2:[^"]*"',
    'name: "phi3:mini"',
    'name: "gpt-oss:[^"]*"',
    'name: "deepseek-v3\.1:[^"]*"',
    'name: "qwen2\.5:[^"]*"'
)
foreach ($pattern in $patterns) {
    $config = $config -replace $pattern, ("name: `"{0}`"" -f $modelName)
}

if ($useGPU) {
    $config = $config -replace 'batch_size: \d+', 'batch_size: 50'
    $config = $config -replace 'delay_between_batches: \d+', 'delay_between_batches: 0'
} else {
    $config = $config -replace 'batch_size: \d+', 'batch_size: 10'
    $config = $config -replace 'delay_between_batches: \d+', 'delay_between_batches: 15'
}

Set-Content -Path $configPath -Value $config -Encoding UTF8
Write-Ok "Configuration updated."

# Step 6: Activate virtual environment
Write-Host "`nStep 6: Activating virtual environment..." -ForegroundColor Yellow
$venvActivate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
    Write-Ok "Virtual environment activated."
} else {
    Write-Warn "Virtual environment not found; using global Python."
}

# Step 7: Run pipeline
Write-Host "`nStep 7: Running pipeline..." -ForegroundColor Yellow
Write-Info "This step may take 10-45 minutes depending on the model."
$pipelineStart = Get-Date
try {
    python run_pipeline.py
}
catch {
    Write-Fail "Pipeline execution threw an exception: $_"
    exit 1
}
$pipelineEnd = Get-Date
Write-Ok ("Pipeline finished in {0:N1} minutes." -f ($pipelineEnd - $pipelineStart).TotalMinutes)

if ($LASTEXITCODE -ne 0) {
    Write-Fail "Pipeline exited with non-zero status. Check logs above."
    exit 1
}

# Step 8: Run validation and report metrics
Write-Host "`nStep 8: Running validation..." -ForegroundColor Yellow
try {
    python validate.py
}
catch {
    Write-Fail "Validation script failed: $_"
    exit 1
}

if ($LASTEXITCODE -ne 0) {
    Write-Fail "Validation exited with non-zero status."
    exit 1
}

Write-Section "Validation Metrics"
$metricsPath = "data\results\validation_metrics.json"
if (Test-Path $metricsPath) {
    try {
        $metrics = Get-Content $metricsPath | ConvertFrom-Json
        Write-Ok ("Cohen's kappa (binned): {0}" -f $metrics.kappa_binned)
        Write-Ok ("MAE: {0}" -f $metrics.mae)
        Write-Ok ("Pearson r: {0}" -f $metrics.pearson_r)
        Write-Ok ("ANOVA F: {0}" -f $metrics.anova_F)
        Write-Ok ("ANOVA p: {0}" -f $metrics.anova_p)
    }
    catch {
        Write-Warn "Unable to parse validation metrics JSON."
    }
} else {
    Write-Warn "Metrics file not found at $metricsPath."
}

Write-Info "Review Streamlit dashboard with: streamlit run streamlit_app.py"
Write-Info "Raw outputs: data\results\analysis_results.json and validation_metrics.json"
Write-Ok "Setup and run sequence complete."
