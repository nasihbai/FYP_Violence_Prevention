@echo off
echo Starting Violence Detection Web Dashboard...
echo.
echo Once started, open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python run_detection.py --web

pause
