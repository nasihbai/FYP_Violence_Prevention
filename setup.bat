@echo off
echo ============================================
echo Violence Detection System - Auto Setup
echo ============================================
echo.

cd /d "%~dp0"

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python is installed and in PATH
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [4/5] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install some dependencies
    echo Trying to continue anyway...
)
echo.

echo [5/5] Verifying installation...
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')"
python -c "from ultralytics import YOLO; print('YOLO: OK')"
echo.

echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To run the detection system:
echo   1. Open Command Prompt (cmd.exe)
echo   2. Navigate to this folder
echo   3. Run: venv\Scripts\activate.bat
echo   4. Run: python run_detection.py
echo.
echo Or use the run scripts created for you:
echo   - run_webcam.bat      (simple webcam detection)
echo   - run_web_dashboard.bat (web interface)
echo.
pause
