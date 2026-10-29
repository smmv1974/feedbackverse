@echo off
echo ================================================================================
echo RUNNING FULL 300-REVIEW VALIDATION WITH OLLAMA CLOUD
echo ================================================================================
echo.
echo Expected time: 30-40 minutes
echo Using models: gpt-oss:20b-cloud (sentiment) + gpt-oss:120b-cloud (themes)
echo.
echo Starting pipeline...
echo.

python scripts/validate_complete.py --reviews 300

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo SUCCESS! Pipeline completed
    echo ================================================================================
    echo.
    echo Results saved to: data/results/
    echo.
    echo View results:
    echo   python -m json.tool data/results/validation_metrics.json
    echo.
) else (
    echo.
    echo ================================================================================
    echo ERROR! Pipeline failed with exit code %ERRORLEVEL%
    echo ================================================================================
    echo.
    echo Check logs/pipeline_ollama_cloud.log for details
)

pause
