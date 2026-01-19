# Testing Checklist for LSTM Actions Recognition

## Pre-Testing Requirements

### Environment Setup
- [ ] Python 3.7+ installed
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Model file `lstm-model.h5` exists in project directory
- [ ] OpenCV compiled with FFMPEG support (for RTSP)

### Verify Installation
```bash
python -c "import cv2; print(cv2.__version__)"
python -c "import mediapipe; print(mediapipe.__version__)"
python -c "import tensorflow; print(tensorflow.__version__)"
```

---

## Test Suite 1: Local Video File Input

### Test 1.1: Basic MP4 File
- [ ] Prepare a test MP4 file with clear full-body human poses
- [ ] Set `VIDEO_PATH = "path/to/test.mp4"` in pose_lstm_realtime.py
- [ ] Run: `python pose_lstm_realtime.py`
- [ ] **Expected**: Video opens, displays window with pose landmarks
- [ ] **Expected**: Label displays "neutral" or "violent" based on actions
- [ ] **Expected**: Video loops when it ends
- [ ] Press 'q' to quit
- [ ] **Expected**: Script exits cleanly

### Test 1.2: Non-existent File
- [ ] Set `VIDEO_PATH = "nonexistent.mp4"`
- [ ] Run script
- [ ] **Expected**: `FileNotFoundError` with clear message
- [ ] **Expected**: Script exits without crash

### Test 1.3: Large Video File
- [ ] Use a video file > 100MB
- [ ] Run script
- [ ] **Expected**: Smooth playback without memory issues
- [ ] Monitor CPU/memory usage

### Test 1.4: Different Video Formats
- [ ] Test with `.avi` format
- [ ] Test with `.mov` format
- [ ] **Expected**: All formats load and play correctly

---

## Test Suite 2: HTTP/HTTPS URL Input

### Test 2.1: Direct MP4 URL
- [ ] Find a direct MP4 URL (e.g., sample video hosting)
- [ ] Set `VIDEO_PATH = "https://example.com/video.mp4"`
- [ ] Run script
- [ ] **Expected**: Video downloads and plays
- [ ] **Expected**: Detection works normally

### Test 2.2: Invalid URL
- [ ] Set `VIDEO_PATH = "https://invalid-url-12345.com/video.mp4"`
- [ ] Run script
- [ ] **Expected**: Clear error message about failed connection
- [ ] **Expected**: Script exits gracefully

### Test 2.3: Slow Network
- [ ] Test with slow internet connection
- [ ] **Expected**: Video may buffer but continues to play
- [ ] **Expected**: Auto-reconnection works if connection drops

---

## Test Suite 3: RTSP Stream Input

### Test 3.1: RTSP Stream (No Authentication)
- [ ] Set `VIDEO_PATH = "rtsp://192.168.1.100:554/stream1"`
- [ ] Run script
- [ ] **Expected**: "Loading RTSP stream" message appears
- [ ] **Expected**: Stream opens successfully
- [ ] **Expected**: Real-time detection works
- [ ] **Expected**: FPS is reasonable (15-30 FPS)

### Test 3.2: RTSP Stream (With Authentication)
- [ ] Set `VIDEO_PATH = "rtsp://username:password@192.168.1.100:554/stream"`
- [ ] Run script
- [ ] **Expected**: Authentication succeeds
- [ ] **Expected**: Stream opens and displays

### Test 3.3: Invalid RTSP Credentials
- [ ] Use wrong username/password
- [ ] Run script
- [ ] **Expected**: Clear authentication error
- [ ] **Expected**: Script exits gracefully

### Test 3.4: RTSP Reconnection
- [ ] Start script with valid RTSP stream
- [ ] During runtime, disconnect network cable or camera
- [ ] Wait 5 seconds
- [ ] Reconnect network/camera
- [ ] **Expected**: "Stream connection lost" message
- [ ] **Expected**: "Attempting to reconnect..." message
- [ ] **Expected**: "Reconnected successfully!" after 2-3 seconds
- [ ] **Expected**: Detection continues normally

### Test 3.5: TCP vs UDP Transport
- [ ] Test with `RTSP_TRANSPORT = "tcp"`
- [ ] Test with `RTSP_TRANSPORT = "udp"`
- [ ] **Expected**: Both work, TCP more stable, UDP slightly faster
- [ ] Note any differences in stability/latency

### Test 3.6: Buffer Size Impact
- [ ] Test with `RTSP_BUFFER_SIZE = 1`
- [ ] Test with `RTSP_BUFFER_SIZE = 3`
- [ ] Test with `RTSP_BUFFER_SIZE = 10`
- [ ] **Expected**: Lower buffer = lower latency but may drop frames
- [ ] **Expected**: Higher buffer = more stable but higher latency
- [ ] Measure latency for each setting

### Test 3.7: Common IP Camera Brands
Test with real IP cameras:
- [ ] Hikvision camera
- [ ] Dahua camera
- [ ] Axis camera
- [ ] Generic ONVIF camera
- [ ] **Expected**: All compatible cameras work

---

## Test Suite 4: Camera Device Input

### Test 4.1: Built-in Webcam
- [ ] Set `VIDEO_PATH = 0`
- [ ] Run script
- [ ] **Expected**: Built-in webcam activates
- [ ] **Expected**: Real-time detection works
- [ ] Perform violent/neutral actions
- [ ] **Expected**: Label changes appropriately

### Test 4.2: External USB Camera
- [ ] Connect external USB camera
- [ ] Try `VIDEO_PATH = 1`, then 2, 3, etc.
- [ ] **Expected**: Correct index opens external camera
- [ ] **Expected**: Detection works normally

### Test 4.3: Multiple Cameras
- [ ] Connect multiple cameras
- [ ] Test each device index
- [ ] **Expected**: Each index opens correct camera
- [ ] **Expected**: No conflicts or crashes

### Test 4.4: Camera Disconnection
- [ ] Start with `VIDEO_PATH = 0`
- [ ] During runtime, disconnect USB camera
- [ ] **Expected**: Reconnection logic attempts to recover
- [ ] **Expected**: Clear error if recovery fails

---

## Test Suite 5: Detection Accuracy

### Test 5.1: Neutral Pose Detection
- [ ] Person standing still
- [ ] Person walking normally
- [ ] Person sitting
- [ ] **Expected**: Label shows "neutral" (green text)

### Test 5.2: Violent Action Detection
- [ ] Person punching
- [ ] Person kicking
- [ ] Person in aggressive stance
- [ ] **Expected**: Label shows "violent" (red text)

### Test 5.3: Edge Cases
- [ ] No person in frame
- [ ] **Expected**: No landmarks drawn, label persists last state
- [ ] Multiple people in frame
- [ ] **Expected**: MediaPipe detects one person (first detected)
- [ ] Person partially out of frame
- [ ] **Expected**: Detection may be inconsistent (acceptable)

### Test 5.4: Lighting Conditions
- [ ] Bright lighting
- [ ] Dim lighting
- [ ] Backlit person
- [ ] **Expected**: MediaPipe handles various conditions reasonably

---

## Test Suite 6: Performance Testing

### Test 6.1: CPU Usage
- [ ] Monitor CPU usage during operation
- [ ] **Expected**: 30-70% CPU usage (depends on hardware)
- [ ] **Expected**: No CPU spikes causing lag

### Test 6.2: Memory Usage
- [ ] Monitor memory usage over 5 minutes
- [ ] **Expected**: Stable memory usage (~500MB-1GB)
- [ ] **Expected**: No memory leaks (usage doesn't continuously grow)

### Test 6.3: Frame Rate
- [ ] Count frames processed per second
- [ ] **Expected**: 20-30 FPS on modern hardware
- [ ] **Expected**: Consistent frame rate, no major drops

### Test 6.4: Warm-up Period
- [ ] Count frames until detection starts
- [ ] **Expected**: Detection starts after 60 frames (~2 seconds at 30 FPS)
- [ ] **Expected**: Warm-up message or no crashes during warm-up

### Test 6.5: Long-term Stability
- [ ] Run script for 1 hour continuously
- [ ] **Expected**: No crashes or freezes
- [ ] **Expected**: Memory usage remains stable
- [ ] **Expected**: Detection accuracy doesn't degrade

---

## Test Suite 7: Error Handling & Edge Cases

### Test 7.1: Missing Model File
- [ ] Rename or remove `lstm-model.h5`
- [ ] Run script
- [ ] **Expected**: Clear error about missing model file
- [ ] **Expected**: Script exits gracefully

### Test 7.2: Corrupted Video File
- [ ] Use a corrupted or incomplete MP4 file
- [ ] Run script
- [ ] **Expected**: Error message about invalid file
- [ ] **Expected**: Script exits without crash

### Test 7.3: Network Interruption (URL)
- [ ] Start with HTTP video URL
- [ ] Disconnect internet during playback
- [ ] **Expected**: Connection lost message
- [ ] **Expected**: Attempt to reconnect
- [ ] Reconnect internet
- [ ] **Expected**: Stream resumes

### Test 7.4: Keyboard Interrupt
- [ ] Run script
- [ ] Press Ctrl+C
- [ ] **Expected**: Script exits cleanly
- [ ] **Expected**: Window closes
- [ ] **Expected**: Resources released

### Test 7.5: Window Closed Manually
- [ ] Run script
- [ ] Click the X button on the video window
- [ ] **Expected**: Script detects window closure
- [ ] **Expected**: Script exits cleanly

---

## Test Suite 8: Configuration Options

### Test 8.1: Verbose Prediction Mode
- [ ] Set `VERBOSE_PREDICTION = True`
- [ ] Run script
- [ ] **Expected**: Console shows prediction percentages
- [ ] **Expected**: Performance not significantly impacted

### Test 8.2: Silent Mode (Default)
- [ ] Set `VERBOSE_PREDICTION = False`
- [ ] Run script
- [ ] **Expected**: No prediction percentages printed
- [ ] **Expected**: Clean console output

### Test 8.3: Different Warm-up Frames
- [ ] Set `warm_up_frames = 30`
- [ ] Run script
- [ ] **Expected**: Detection starts earlier
- [ ] Set `warm_up_frames = 120`
- [ ] **Expected**: Longer warm-up period

---

## Test Suite 9: Threading & Concurrency

### Test 9.1: Thread Safety
- [ ] Run script with fast video playback
- [ ] **Expected**: No threading errors in console
- [ ] **Expected**: Predictions don't corrupt each other

### Test 9.2: Multiple Predictions Overlap
- [ ] Monitor if new predictions start before previous finish
- [ ] **Expected**: System handles overlapping predictions
- [ ] **Expected**: No crashes or race conditions

---

## Test Suite 10: Output & Visualization

### Test 10.1: Landmark Visualization
- [ ] Run script with person in frame
- [ ] **Expected**: Green skeleton overlay on person
- [ ] **Expected**: 33 pose landmarks visible
- [ ] **Expected**: Landmarks track movement smoothly

### Test 10.2: Bounding Box
- [ ] Verify green rectangle around detected person
- [ ] **Expected**: Box adapts to person's position
- [ ] **Expected**: Box size adjusts to distance from camera

### Test 10.3: Label Display
- [ ] Check label text in top-left corner
- [ ] **Expected**: "neutral" in green when no violence
- [ ] **Expected**: "violent" in red when violence detected
- [ ] **Expected**: Text clearly readable

### Test 10.4: Window Resizing
- [ ] Manually resize the video window
- [ ] **Expected**: Video scales appropriately
- [ ] **Expected**: No distortion or crashes

---

## Test Suite 11: Cross-Platform Testing

### Test 11.1: Windows
- [ ] Run on Windows 10/11
- [ ] **Expected**: All features work correctly
- [ ] Note any platform-specific issues

### Test 11.2: macOS
- [ ] Run on macOS (Intel and Apple Silicon)
- [ ] **Expected**: All features work correctly
- [ ] Note any platform-specific issues

### Test 11.3: Linux
- [ ] Run on Ubuntu/Debian
- [ ] **Expected**: All features work correctly
- [ ] Note any platform-specific issues

---

## Test Suite 12: Documentation Verification

### Test 12.1: README Instructions
- [ ] Follow instructions in README.md
- [ ] **Expected**: All steps work as described
- [ ] Note any unclear or outdated instructions

### Test 12.2: VIDEO_INPUT_GUIDE
- [ ] Follow examples in VIDEO_INPUT_GUIDE.md
- [ ] Test each configuration option
- [ ] **Expected**: All examples work correctly

### Test 12.3: PROJECT_ANALYSIS
- [ ] Verify information accuracy in PROJECT_ANALYSIS.md
- [ ] **Expected**: Descriptions match actual behavior

---

## Performance Benchmarks

Record these metrics during testing:

| Test Scenario | FPS | CPU % | RAM (MB) | Latency (ms) | Notes |
|---------------|-----|-------|----------|---------------|-------|
| Local MP4     |     |       |          |               |       |
| HTTP URL      |     |       |          |               |       |
| RTSP (TCP)    |     |       |          |               |       |
| RTSP (UDP)    |     |       |          |               |       |
| Webcam        |     |       |          |               |       |
| USB Camera    |     |       |          |               |       |

---

## Known Limitations (Document During Testing)

- Maximum resolution tested: _______
- Maximum video length tested: _______
- RTSP streams tested: _______
- Camera brands tested: _______
- OS versions tested: _______

---

## Bug Report Template

If you find issues, document them using this format:

```
**Issue Title**: [Brief description]

**Severity**: Critical / High / Medium / Low

**Steps to Reproduce**:
1.
2.
3.

**Expected Behavior**:

**Actual Behavior**:

**Environment**:
- OS:
- Python Version:
- OpenCV Version:
- Video Source Type:

**Error Messages/Logs**:
```

**Screenshots/Videos**: [If applicable]

---

## Sign-off Checklist

Before marking testing as complete:

- [ ] All critical tests passed
- [ ] No major bugs found
- [ ] Performance meets expectations
- [ ] Documentation is accurate
- [ ] Cross-platform compatibility verified
- [ ] RTSP support fully functional
- [ ] Error handling works correctly
- [ ] Long-term stability confirmed

**Tested By**: _________________
**Date**: _________________
**Overall Status**: Pass / Fail / Needs Revision
**Notes**: _________________
