@echo off
echo ================================================================================
echo OLLAMA CLOUD PIPELINE - SIMPLE RUN
echo ================================================================================
echo.
echo Using: 120B Ollama Cloud Models
echo Config: config_ollama_cloud.yaml (CORRECT CONFIG!)
echo Expected: 30-40 minutes for 300 reviews
echo.
echo ================================================================================
echo.

REM Check if API key is set
if not defined OLLAMA_API_KEY (
    echo [ERROR] OLLAMA_API_KEY environment variable not set
    echo [ERROR] Please set it first: set OLLAMA_API_KEY=your-key-here
    pause
    exit /b 1
)

echo [INFO] API Key detected
echo [INFO] Starting pipeline...
echo.

python scripts/run_pipeline.py --config config/config_ollama_cloud.yaml

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo SUCCESS! Pipeline completed
    echo ================================================================================
    echo.
    echo Results in: data/results/
    echo.
) else (
    echo.
    echo ================================================================================
    echo ERROR! Exit code: %ERRORLEVEL%
    echo ================================================================================
    echo.
)

pause
