@echo off
echo Starting Violence Detection (Webcam Mode)...
echo Press 'q' in the video window to quit
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python run_detection.py

pause
