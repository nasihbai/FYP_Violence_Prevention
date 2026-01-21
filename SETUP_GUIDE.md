# Violence Detection System - Complete Setup & Run Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [What's New (Improvements Made)](#whats-new)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Detailed Usage Guide](#detailed-usage-guide)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

This is an enhanced **LSTM-based Violence Detection System** that uses:
- **YOLOv8** for multi-person detection
- **MediaPipe** for pose estimation
- **LSTM with Attention** for temporal action recognition
- **Real-time alerts** (sound, email, webhooks)
- **Web dashboard** for monitoring

---

## What's New (Improvements Made)

### 🎯 Major Enhancements

| Feature | Before | After |
|---------|--------|-------|
| **Person Detection** | Single person only | Multi-person with YOLO |
| **Model Architecture** | Basic 4-layer LSTM | Enhanced LSTM with Attention + TCN |
| **Thread Safety** | Unsafe global variables | Proper locks and queues |
| **Data Handling** | No augmentation | Full augmentation + balancing |
| **Alerts** | None (unused alert.wav) | Sound + Email + Webhooks |
| **Evaluation** | No metrics | Full confusion matrix, ROC, PR curves |
| **Configuration** | Hard-coded values | Centralized config system |
| **Interface** | OpenCV window only | Web dashboard + OpenCV |

### 📁 New Files Created

```
FYP_Violence_Prevention/
├── config/
│   ├── __init__.py
│   └── settings.py              # Centralized configuration
├── core/
│   ├── __init__.py
│   ├── yolo_detector.py         # YOLO multi-person detection
│   ├── lstm_model.py            # Enhanced LSTM with attention
│   ├── pose_extractor.py        # Thread-safe pose extraction
│   └── detection_engine.py      # Main detection pipeline
├── utils/
│   ├── __init__.py
│   ├── data_augmentation.py     # Data augmentation utilities
│   └── evaluation.py            # Model evaluation framework
├── alerts/
│   ├── __init__.py
│   └── alert_system.py          # Alert notification system
├── web/
│   ├── __init__.py
│   ├── app.py                   # Flask web dashboard
│   └── templates/
│       └── index.html           # Dashboard UI
├── run_detection.py             # Main entry point
├── train_model_enhanced.py      # Enhanced training script
├── collect_data.py              # Data collection utility
├── requirements.txt             # Updated dependencies
└── SETUP_GUIDE.md              # This guide
```

---

## System Requirements

### Hardware
- **CPU**: Intel i5/AMD Ryzen 5 or better
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA (optional, but recommended)
- **Webcam**: Any USB webcam or IP camera

### Software
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.8 - 3.11 (3.10 recommended)
- **CUDA**: 11.2+ (for GPU acceleration)

---

## Installation

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd c:\Users\User\Desktop\Sih_Project\FYP_Violence_Prevention

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
# source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Step 3: Download YOLO Model (Automatic)

The YOLOv8 model will be downloaded automatically on first run. If you want to pre-download:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Step 4: Verify Installation

```bash
python -c "
import tensorflow as tf
import mediapipe as mp
import cv2
from ultralytics import YOLO

print(f'TensorFlow: {tf.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'MediaPipe: {mp.__version__}')
print('YOLO: OK')
print('All dependencies installed successfully!')
"
```

---

## Project Structure

```
FYP_Violence_Prevention/
│
├── 📁 config/                    # Configuration management
│   └── settings.py               # All settings in one place
│
├── 📁 core/                      # Core detection modules
│   ├── yolo_detector.py          # Multi-person detection
│   ├── lstm_model.py             # Violence classification
│   ├── pose_extractor.py         # Pose landmark extraction
│   └── detection_engine.py       # Main processing pipeline
│
├── 📁 utils/                     # Utility functions
│   ├── data_augmentation.py      # Training data augmentation
│   └── evaluation.py             # Model evaluation metrics
│
├── 📁 alerts/                    # Alert system
│   └── alert_system.py           # Sound, email, webhook alerts
│
├── 📁 web/                       # Web dashboard
│   ├── app.py                    # Flask application
│   └── templates/                # HTML templates
│
├── 📁 models/                    # Saved models (created on training)
├── 📁 data/                      # Training data
├── 📁 logs/                      # Log files
├── 📁 alerts/screenshots/        # Alert screenshots
│
├── 📄 run_detection.py           # ⭐ Main entry point
├── 📄 train_model_enhanced.py    # Training script
├── 📄 collect_data.py            # Data collection
├── 📄 requirements.txt           # Dependencies
│
└── 📄 Original files (preserved):
    ├── pose_lstm_realtime.py     # Original detection script
    ├── train_model.py            # Original training script
    ├── lstm-model.h5             # Original trained model
    └── *.txt                     # Original training data
```

---

## Quick Start

### Option 1: Run with Webcam (Simplest)

```bash
python run_detection.py
```

### Option 2: Run with Web Dashboard

```bash
python run_detection.py --web
# Open browser: http://localhost:5000
```

### Option 3: Run Original Script (Backward Compatible)

```bash
python pose_lstm_realtime.py
```

---

## Detailed Usage Guide

### 1. Data Collection

Collect training data for new actions:

```bash
# Collect neutral (normal) behavior - 1000 frames
python collect_data.py --label neutral --frames 1000

# Collect violent behavior - 500 frames
python collect_data.py --label violent --frames 500

# Collect from specific camera
python collect_data.py --label neutral --frames 1000 --camera 1
```

**Tips for data collection:**
- Ensure good lighting
- Vary positions and angles
- For violence: simulate pushing, punching motions (safely!)
- Collect at least 500 frames per class

### 2. Training the Model

Train with the enhanced model:

```bash
# Basic training
python train_model_enhanced.py

# With custom settings
python train_model_enhanced.py --epochs 150 --batch-size 64 --augmentation-factor 5

# Train simple model (faster, less accurate)
python train_model_enhanced.py --simple-model
```

**Training outputs:**
- `models/violence_lstm_enhanced.h5` - Trained model
- `models/evaluation/` - Evaluation plots and metrics
- `models/checkpoints/` - Training checkpoints

### 3. Running Detection

#### Basic Detection (OpenCV Window)

```bash
# Webcam (camera 0)
python run_detection.py

# Specific camera
python run_detection.py --source 1

# Video file
python run_detection.py --source path/to/video.mp4

# RTSP stream (IP camera)
python run_detection.py --source "rtsp://admin:password@192.168.1.100:554/stream"

# HTTP stream
python run_detection.py --source "http://example.com/stream.mp4"
```

#### Detection Options

```bash
# Adjust violence threshold (default: 0.6)
python run_detection.py --threshold 0.7

# Disable YOLO (single person mode, faster)
python run_detection.py --no-yolo

# Disable sound alerts
python run_detection.py --no-sound

# Custom model
python run_detection.py --model path/to/custom_model.h5
```

#### Web Dashboard Mode

```bash
# Start web dashboard
python run_detection.py --web

# Custom port
python run_detection.py --web --port 8080

# Allow external access
python run_detection.py --web --host 0.0.0.0 --port 5000
```

Then open your browser to `http://localhost:5000`

### 4. Using the Web Dashboard

The web dashboard provides:

| Feature | Description |
|---------|-------------|
| **Live Feed** | Real-time video with detection overlay |
| **Statistics** | Frame count, FPS, violence detections |
| **Alert History** | Recent violence alerts with timestamps |
| **Configuration** | Adjust threshold in real-time |
| **Controls** | Start/Stop/Reset detection |
| **Snapshot** | Capture current frame |

### 5. Alert Configuration

Edit `config/settings.py` to configure alerts:

```python
class AlertConfig:
    # Sound alerts
    SOUND_ENABLED = True
    SOUND_FILE = BASE_DIR / "alert.wav"

    # Email alerts (configure with your SMTP)
    EMAIL_ENABLED = True
    EMAIL_SMTP_SERVER = "smtp.gmail.com"
    EMAIL_SMTP_PORT = 587
    EMAIL_SENDER = "your-email@gmail.com"
    EMAIL_PASSWORD = "your-app-password"
    EMAIL_RECIPIENTS = ["recipient@example.com"]

    # Webhook alerts (Slack, Discord, etc.)
    WEBHOOK_ENABLED = True
    WEBHOOK_URL = "https://hooks.slack.com/services/xxx/yyy/zzz"
```

---

## Configuration

### Main Configuration File: `config/settings.py`

#### Video Settings
```python
class VideoConfig:
    SOURCE = 0                    # Camera index or path
    RTSP_BUFFER_SIZE = 1          # Lower = less latency
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
```

#### Model Settings
```python
class ModelConfig:
    YOLO_MODEL = "yolov8n.pt"     # Options: yolov8n, yolov8s, yolov8m
    VIOLENCE_THRESHOLD = 0.6      # 0-1, higher = more strict
    LSTM_SEQUENCE_LENGTH = 20     # Frames per prediction
```

#### Detection Settings
```python
class DetectionConfig:
    WARMUP_FRAMES = 30            # Skip initial frames
    MAX_PERSONS = 10              # Max people to track
    SMOOTHING_WINDOW = 5          # Prediction averaging
```

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'ultralytics'"
```bash
pip install ultralytics
```

#### 2. "CUDA out of memory"
- Use smaller YOLO model: Edit `config/settings.py`
  ```python
  YOLO_MODEL = "yolov8n.pt"  # nano version
  ```
- Or disable YOLO:
  ```bash
  python run_detection.py --no-yolo
  ```

#### 3. "Camera not found"
- Check camera index:
  ```python
  import cv2
  for i in range(5):
      cap = cv2.VideoCapture(i)
      if cap.isOpened():
          print(f"Camera {i}: Available")
          cap.release()
  ```

#### 4. "Model not found"
- The system will use the original `lstm-model.h5` if no enhanced model exists
- Train a new model:
  ```bash
  python train_model_enhanced.py
  ```

#### 5. TensorFlow Warnings
- These are usually harmless. To suppress:
  ```python
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  ```

#### 6. Slow Performance
1. Use smaller YOLO model (`yolov8n.pt`)
2. Reduce frame resolution in `config/settings.py`
3. Disable YOLO for single-person mode
4. Use GPU if available

### Getting Help

1. Check the logs in `logs/` directory
2. Run with debug mode:
   ```bash
   python run_detection.py --debug
   ```
3. Check GitHub issues or create a new one

---

## Example Workflows

### Workflow 1: Fresh Start (No Existing Model)

```bash
# 1. Collect neutral data
python collect_data.py --label neutral --frames 1000

# 2. Collect violent data
python collect_data.py --label violent --frames 500

# 3. Train model
python train_model_enhanced.py

# 4. Run detection
python run_detection.py
```

### Workflow 2: Use Existing Data

```bash
# Train with existing neutral.txt and violent.txt
python train_model_enhanced.py --data-dir .

# Run detection
python run_detection.py
```

### Workflow 3: Production Deployment

```bash
# Run with web dashboard on network
python run_detection.py --web --host 0.0.0.0 --port 8080 --threshold 0.7
```

---

## API Reference (for Integration)

### REST API Endpoints (Web Mode)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard page |
| `/video_feed` | GET | MJPEG video stream |
| `/api/stats` | GET | Current statistics |
| `/api/alerts` | GET | Alert history |
| `/api/config` | GET/POST | Configuration |
| `/api/start` | POST | Start detection |
| `/api/stop` | POST | Stop detection |
| `/api/reset` | POST | Reset statistics |
| `/api/snapshot` | GET | Current frame |
| `/health` | GET | Health check |

### Python API

```python
from core.detection_engine import ThreadSafeDetector, VideoProcessor

# Initialize detector
detector = ThreadSafeDetector(
    lstm_model_path='models/violence_lstm_enhanced.h5',
    use_yolo=True,
    violence_threshold=0.6
)

# Start workers
detector.start()

# Process frame
result = detector.process_frame(frame)

# Check for violence
if result.has_violence:
    for det in result.detections:
        print(f"Person {det.person_id}: {det.class_name} ({det.confidence:.2%})")
```

---

## License & Credits

- Original project: FYP Violence Prevention (LSTM Actions Recognition)
- Enhancements: Multi-person detection, attention mechanism, web dashboard
- Libraries: TensorFlow, MediaPipe, YOLOv8 (Ultralytics), Flask

---

**Happy Detecting! 🎯**
