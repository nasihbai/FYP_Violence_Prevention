@echo off
echo Starting Model Training...
echo This may take a while depending on your hardware.
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python train_model_enhanced.py

pause
