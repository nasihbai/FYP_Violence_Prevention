@echo off
echo Starting Original Violence Detection Script...
echo Press 'q' in the video window to quit
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python pose_lstm_realtime.py

pause
