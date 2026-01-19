# Video Input Guide for pose_lstm_realtime.py

## Overview

The `pose_lstm_realtime.py` script has been modified to support **four types of video input**:
1. **Local MP4 files**
2. **MP4 video URLs** (direct links)
3. **RTSP streams** (IP cameras, streaming servers)
4. **Camera devices** (original functionality)

## How to Configure Video Source

Open [pose_lstm_realtime.py](pose_lstm_realtime.py) and look for the **VIDEO SOURCE CONFIGURATION** section (around line 12-32).

### Option 1: Use Local MP4 File

```python
# Uncomment and modify this line:
VIDEO_PATH = "path/to/your/video.mp4"

# Examples:
VIDEO_PATH = "videos/sample.mp4"
VIDEO_PATH = "/Users/username/Desktop/test_video.mp4"
VIDEO_PATH = "C:/Users/username/Videos/action.mp4"  # Windows
```

### Option 2: Use MP4 Video URL

```python
# Uncomment and modify this line:
VIDEO_PATH = "https://example.com/your-video.mp4"

# Example with real URL:
VIDEO_PATH = "https://www.example.com/samples/violent_detection_test.mp4"
```

**Note:** The URL must be a **direct link** to an MP4 file, not a webpage containing a video player.

### Option 3: Use RTSP Stream (IP Camera)

```python
# Uncomment and modify this line:
VIDEO_PATH = "rtsp://username:password@192.168.1.100:554/stream"

# Examples:
# Without authentication:
VIDEO_PATH = "rtsp://192.168.1.100:554/stream1"

# With authentication:
VIDEO_PATH = "rtsp://admin:password123@192.168.1.100:554/live/main"

# Popular IP camera RTSP formats:
VIDEO_PATH = "rtsp://admin:password@192.168.1.100:554/h264/ch1/main/av_stream"  # Hikvision
VIDEO_PATH = "rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"  # Dahua
VIDEO_PATH = "rtsp://192.168.1.100:554/axis-media/media.amp"  # Axis
VIDEO_PATH = "rtsp://192.168.1.100:8554/unicast"  # Generic
```

**RTSP Configuration Options:**

In the script, you can adjust these settings for better performance:

```python
RTSP_BUFFER_SIZE = 1  # Lower value = less latency (default: 1)
RTSP_TRANSPORT = "tcp"  # Use "tcp" or "udp" (tcp is more reliable)
```

**Note:**
- RTSP streams provide real-time video from IP cameras
- Lower buffer size reduces latency but may cause frame drops
- TCP transport is more reliable but slightly slower than UDP
- Make sure your camera supports RTSP protocol

### Option 4: Use Camera Device (Original)

```python
# Uncomment and modify this line:
VIDEO_PATH = 0  # or 1, 6, etc.

# Examples:
VIDEO_PATH = 0   # Built-in webcam
VIDEO_PATH = 1   # First external USB camera
VIDEO_PATH = 6   # Sixth camera device (original default)
```

## Features Added

### 1. **RTSP Stream Support**
Full support for RTSP protocol with:
- Automatic reconnection on stream failure
- Configurable buffer size for latency control
- TCP/UDP transport selection
- Authentication support

### 2. **Video Looping**
When using MP4 files, the video will automatically loop back to the start when it reaches the end. This allows continuous testing without manually restarting.

### 3. **Smart Reconnection**
For live streams (RTSP, camera, live URLs):
- Automatically attempts to reconnect on connection loss
- 2-second delay between reconnection attempts
- Resets detection state on reconnection

### 4. **Error Handling**
- Validates that local files exist before attempting to open them
- Checks if video source opened successfully
- Provides clear error messages if something goes wrong
- Graceful degradation on stream failures

### 5. **Status Messages**
The script prints which video source is being used:
```
Loading RTSP stream: rtsp://192.168.1.100:554/stream1
Loading video from URL: https://example.com/video.mp4
Loading video from local file: /path/to/video.mp4
Using camera device: 0
Video opened successfully!
Video Info - FPS: 30.0, Frames: 1500, Resolution: 1920x1080
```

### 6. **Performance Optimizations**
- Silent prediction mode (no verbose TensorFlow output)
- Thread safety with list copying
- Optional verbose prediction logging
- Optimized RTSP buffer management

## Usage Examples

### Example 1: Test with Local Video

```python
# In pose_lstm_realtime.py, set:
VIDEO_PATH = "test_videos/violent_action.mp4"
```

Then run:
```bash
python pose_lstm_realtime.py
```

### Example 2: Test with URL

```python
# In pose_lstm_realtime.py, set:
VIDEO_PATH = "https://storage.googleapis.com/sample-videos/action.mp4"
```

Then run:
```bash
python pose_lstm_realtime.py
```

### Example 3: Use RTSP IP Camera

```python
# In pose_lstm_realtime.py, set:
VIDEO_PATH = "rtsp://admin:password@192.168.1.100:554/stream1"
```

Then run:
```bash
python pose_lstm_realtime.py
```

### Example 4: Use Webcam

```python
# In pose_lstm_realtime.py, set:
VIDEO_PATH = 0
```

Then run:
```bash
python pose_lstm_realtime.py
```

## Keyboard Controls

- **Press 'q'** - Quit the application

## Troubleshooting

### Issue: "Failed to open video source"

**Causes:**
1. Local file doesn't exist or path is incorrect
2. URL is not a direct MP4 link
3. Network connection issue for URLs/RTSP
4. Camera device index is incorrect
5. RTSP authentication failed
6. RTSP port blocked by firewall

**Solutions:**
- Verify the file path is correct and file exists
- Test the URL in a browser to ensure it's a direct video link
- Try different camera indices (0, 1, 2, etc.)
- Check your internet connection for URL-based videos
- Verify RTSP credentials (username/password)
- Test RTSP stream with VLC media player first
- Check firewall settings for RTSP ports (usually 554)
- Try changing `RTSP_TRANSPORT` from "tcp" to "udp" or vice versa

### Issue: Video is too slow or choppy

**Causes:**
- High-resolution video
- CPU-intensive LSTM predictions
- Network bandwidth issues (for URLs/RTSP)
- High RTSP latency

**Solutions:**
- Use lower resolution videos
- Reduce `warm_up_frames` value (currently 60)
- Download the video locally instead of streaming from URL
- Close other applications to free up CPU
- For RTSP: Set `RTSP_BUFFER_SIZE = 1` for lower latency
- For RTSP: Try switching transport protocol (tcp/udp)
- Request lower resolution stream from IP camera if available

### Issue: RTSP stream keeps disconnecting

**Causes:**
- Network instability
- Camera timeout settings
- Bandwidth limitations
- Firewall interruptions

**Solutions:**
- Check network stability and signal strength
- Increase camera's RTSP timeout setting
- Use wired connection instead of WiFi if possible
- Reduce video quality/bitrate on camera settings
- The script has auto-reconnection - it will try to reconnect automatically

### Issue: "FileNotFoundError"

**Solution:**
- Check if the file path is correct
- Use absolute paths instead of relative paths
- Verify file extension is `.mp4`

### Issue: Predictions are incorrect

**Causes:**
- Video content doesn't match training data
- Person not fully visible in frame
- Low-quality video

**Solutions:**
- Ensure the person's full body is visible
- Use videos similar to training data
- Check that pose landmarks are being detected (you'll see skeleton overlay)

## Technical Details

### Supported Video Sources

The script supports multiple input types:

**Video Files:**
- MP4 (recommended)
- AVI
- MOV
- MKV

**Streaming Protocols:**
- RTSP (Real-Time Streaming Protocol)
- HTTP/HTTPS video URLs
- Live camera devices

### Video Processing Pipeline

```
Video Source → Frame Capture → MediaPipe Pose Detection →
Landmark Extraction → Buffer (20 frames) → LSTM Model →
Prediction (violent/neutral) → Display with Overlay
```

### Performance Considerations

1. **Local files** - Fastest, no network latency
2. **URLs** - Depends on network speed and video hosting
3. **RTSP streams** - Real-time with low latency (1-3 seconds), network dependent
4. **Camera devices** - Real-time, depends on camera quality

### RTSP-Specific Details

**Buffer Size Impact:**
- `RTSP_BUFFER_SIZE = 1` → Lowest latency, may drop frames on slow network
- `RTSP_BUFFER_SIZE = 3` → More stable, slightly higher latency
- `RTSP_BUFFER_SIZE = 10` → Very stable, higher latency (5-10 seconds)

**Transport Protocol:**
- **TCP** (default): More reliable, handles packet loss, slightly slower
- **UDP**: Faster but may lose packets on poor networks

**Common RTSP Ports:**
- Port 554: Standard RTSP port
- Port 8554: Alternative RTSP port
- Port 88: Some Hikvision cameras

### Warm-up Period

The first 60 frames are skipped to allow MediaPipe to stabilize detection. This is normal behavior.

## Code Changes Summary

### Changes Made to `pose_lstm_realtime.py`:

1. **Added imports**: `import os`, `import time`
2. **Added VIDEO_PATH configuration** with 4 input options
3. **Added RTSP support** with buffer and transport configuration
4. **Added video source detection logic** (auto-detects RTSP/URL/file/camera)
5. **Added smart reconnection** for live streams
6. **Added video loop handling** for local files
7. **Added silent prediction mode** (verbose=0)
8. **Added thread safety** with list copying
9. **Added safety checks** for empty coordinate lists
10. **Added video info display** (FPS, resolution, frame count)
11. **Added performance configuration** (VERBOSE_PREDICTION flag)

### Original Behavior Preserved

The original functionality with camera devices is fully preserved. Setting `VIDEO_PATH = 6` will work exactly like the original code.

## How to Test RTSP Stream

Before using an RTSP stream in this script, test it first:

### Method 1: Using VLC Media Player
1. Open VLC
2. Media → Open Network Stream
3. Enter your RTSP URL: `rtsp://username:password@192.168.1.100:554/stream`
4. Click Play
5. If video plays, the RTSP URL is correct

### Method 2: Using FFmpeg
```bash
ffplay rtsp://admin:password@192.168.1.100:554/stream1
```

### Method 3: Using Python Test Script
```python
import cv2

rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream1"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if cap.isOpened():
    print("RTSP stream opened successfully!")
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured: {frame.shape}")
    else:
        print("Failed to read frame")
else:
    print("Failed to open RTSP stream")

cap.release()
```

## Next Steps

1. Choose your video source type
2. Edit the `VIDEO_PATH` variable in [pose_lstm_realtime.py](pose_lstm_realtime.py)
3. Run the script
4. Press 'q' to quit when done

## Example Video Sources for Testing

You can test with these sample video types:
- Walking/standing people (should detect as "neutral")
- Fighting/aggressive movements (should detect as "violent")
- Mixed actions for comprehensive testing

Make sure the video shows the full body of the person for accurate pose detection.
