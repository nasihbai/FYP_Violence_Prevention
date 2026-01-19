# Quick Start Guide

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, mediapipe, tensorflow; print('✓ All dependencies installed')"
```

---

## 📹 Choose Your Video Source

Edit `pose_lstm_realtime.py` around **line 12-23** and uncomment ONE option:

### Option 1: Local Video File
```python
VIDEO_PATH = "path/to/video.mp4"
```

### Option 2: HTTP/HTTPS URL
```python
VIDEO_PATH = "https://example.com/video.mp4"
```

### Option 3: RTSP Stream (IP Camera)
```python
VIDEO_PATH = "rtsp://admin:password@192.168.1.100:554/stream"
```

### Option 4: Webcam
```python
VIDEO_PATH = 0  # 0, 1, 2, etc.
```

---

## ▶️ Run

```bash
python pose_lstm_realtime.py
```

**Press 'q' to quit**

---

## ⚙️ RTSP Configuration (Optional)

```python
RTSP_BUFFER_SIZE = 1    # 1=low latency, 10=more stable
RTSP_TRANSPORT = "tcp"  # "tcp" or "udp"
```

---

## 🎯 What You'll See

- **Green skeleton** over detected person
- **Green box** around person
- **Label** at top-left:
  - "neutral" in green (normal behavior)
  - "violent" in red (aggressive action)

---

## 🐛 Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| File not found | Check path, use absolute path |
| RTSP won't connect | Test with VLC first |
| High latency | Set `RTSP_BUFFER_SIZE = 1` |
| No detection | Ensure full body visible |

---

## 📚 Full Documentation

- [VIDEO_INPUT_GUIDE.md](VIDEO_INPUT_GUIDE.md) - Detailed configuration guide
- [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md) - How the system works
- [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) - Complete test suite
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details

---

## ✅ Requirements

- Python 3.7+
- OpenCV with FFMPEG (for RTSP)
- Model file: `lstm-model.h5` (must exist in project folder)

---

## 🎥 RTSP Examples

```python
# No authentication
VIDEO_PATH = "rtsp://192.168.1.100:554/stream1"

# With username/password
VIDEO_PATH = "rtsp://admin:password@192.168.1.100:554/stream"

# Hikvision
VIDEO_PATH = "rtsp://admin:password@192.168.1.100:554/h264/ch1/main/av_stream"

# Dahua
VIDEO_PATH = "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
```

**Test RTSP with VLC before using in script!**

---

## 💡 Pro Tips

1. **Lower latency**: `RTSP_BUFFER_SIZE = 1`, use TCP
2. **More stable**: `RTSP_BUFFER_SIZE = 3-10`, use TCP
3. **Faster processing**: Close other applications
4. **Best accuracy**: Ensure full body in frame
5. **Debug mode**: Set `VERBOSE_PREDICTION = True` (line 31)

---

**That's it! You're ready to go.** 🎉
