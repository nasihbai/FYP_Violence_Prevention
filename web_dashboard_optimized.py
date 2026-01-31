"""
Optimized Web Dashboard for Violence Detection
===============================================
High-performance web dashboard with:
- TFLite inference for 3-5x speedup
- Reduced latency (15-frame sequences)
- Real-time FPS and latency display
- Alert system for violence events
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.optimized_detector import OptimizedDetector, DetectorConfig, FrameResult

# ===== CONFIGURATION =====
VIDEO_SOURCE = 0  # Webcam (change to file path or RTSP URL)
HOST = '0.0.0.0'
PORT = 5000

# Create Flask app
app = Flask(__name__)

# Global state
detector = None
latest_result: FrameResult = None
stats = {
    'fps': 0.0,
    'inference_ms': 0.0,
    'violence_events': 0,
    'status': 'Initializing...',
    'uptime': 0,
    'frames_processed': 0,
    'model_type': 'Unknown'
}
start_time = time.time()
stats_lock = threading.Lock()


def init_detector():
    """Initialize the optimized detector."""
    global detector, stats

    config = DetectorConfig(
        use_tflite=True,
        sequence_length=15,
        prediction_stride=5,
        warmup_frames=20,
        smoothing_window=3,
        violence_threshold=0.6,
        early_detection_threshold=0.9,
        pose_model_complexity=0,  # Lite model for speed
        use_features=False,  # Disable for now until feature model is trained
    )

    detector = OptimizedDetector(config)

    with stats_lock:
        stats['model_type'] = 'TFLite' if detector.use_tflite else 'Keras'
        stats['status'] = 'Ready'

    print(f"Detector initialized. Using {'TFLite' if detector.use_tflite else 'Keras'} inference.")


def generate_frames():
    """Generate video frames with detection overlay."""
    global latest_result, stats

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    print(f"Video source opened: {VIDEO_SOURCE}")

    while True:
        ret, frame = cap.read()

        if not ret:
            # Loop video file
            if isinstance(VIDEO_SOURCE, str) and not VIDEO_SOURCE.startswith(('rtsp://', 'http://')):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                detector.reset()
                continue
            time.sleep(0.1)
            continue

        # Process frame
        result = detector.process_frame(frame)
        latest_result = result

        # Update stats
        with stats_lock:
            stats['fps'] = result.fps
            stats['inference_ms'] = result.inference_time_ms
            stats['frames_processed'] = detector.frame_count
            stats['uptime'] = int(time.time() - start_time)

            if result.has_violence:
                stats['violence_events'] += 1
                stats['status'] = 'VIOLENCE DETECTED'
            elif detector.frame_count <= detector.config.warmup_frames:
                stats['status'] = f'Warming up ({detector.frame_count}/{detector.config.warmup_frames})'
            else:
                stats['status'] = 'Monitoring'

        # Draw results
        output = detector.draw_results(frame, result)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Violence Detection - Optimized Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
        }
        .header {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #00ff88;
        }
        .header h1 {
            font-size: 2em;
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        .header .subtitle {
            color: #888;
            margin-top: 5px;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            padding: 20px;
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .video-section {
            flex: 2;
            min-width: 600px;
        }
        .video-container {
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        .video-container img {
            width: 100%;
            display: block;
        }
        .stats-section {
            flex: 1;
            min-width: 300px;
        }
        .stats-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stats-card h3 {
            color: #00ff88;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-label { color: #888; }
        .stat-value {
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }
        .status-indicator {
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .status-safe {
            background: linear-gradient(135deg, #00c853, #00e676);
            color: #000;
        }
        .status-warning {
            background: linear-gradient(135deg, #ff6d00, #ff9100);
            color: #000;
            animation: pulse 1s infinite;
        }
        .status-danger {
            background: linear-gradient(135deg, #ff1744, #ff5252);
            color: #fff;
            animation: pulse 0.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .performance-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin-top: 5px;
            overflow: hidden;
        }
        .performance-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .fps-bar { background: linear-gradient(90deg, #00ff88, #00e676); }
        .latency-bar { background: linear-gradient(90deg, #2196f3, #00bcd4); }
        .optimizations {
            font-size: 0.85em;
            color: #888;
        }
        .optimizations li {
            margin: 5px 0;
            list-style: none;
        }
        .optimizations li:before {
            content: "✓ ";
            color: #00ff88;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Violence Detection System</h1>
        <div class="subtitle">Optimized Real-Time Monitoring</div>
    </div>

    <div class="container">
        <div class="video-section">
            <div class="video-container">
                <img src="/video_feed" alt="Video Feed">
            </div>
        </div>

        <div class="stats-section">
            <div id="status-indicator" class="status-indicator status-safe">
                Initializing...
            </div>

            <div class="stats-card">
                <h3>Performance Metrics</h3>
                <div class="stat-row">
                    <span class="stat-label">FPS</span>
                    <span class="stat-value" id="fps">0.0</span>
                </div>
                <div class="performance-bar">
                    <div class="performance-fill fps-bar" id="fps-bar" style="width: 0%"></div>
                </div>

                <div class="stat-row" style="margin-top: 15px">
                    <span class="stat-label">Inference Time</span>
                    <span class="stat-value" id="inference-ms">0 ms</span>
                </div>
                <div class="performance-bar">
                    <div class="performance-fill latency-bar" id="latency-bar" style="width: 0%"></div>
                </div>
            </div>

            <div class="stats-card">
                <h3>Detection Stats</h3>
                <div class="stat-row">
                    <span class="stat-label">Violence Events</span>
                    <span class="stat-value" id="violence-events">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Frames Processed</span>
                    <span class="stat-value" id="frames-processed">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Uptime</span>
                    <span class="stat-value" id="uptime">0s</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Model Type</span>
                    <span class="stat-value" id="model-type">Unknown</span>
                </div>
            </div>

            <div class="stats-card">
                <h3>Active Optimizations</h3>
                <ul class="optimizations">
                    <li>TFLite Inference (3-5x faster)</li>
                    <li>Reduced Sequence (15 frames)</li>
                    <li>Prediction Stride (every 5 frames)</li>
                    <li>Exponential Smoothing</li>
                    <li>Early Detection (>90% confidence)</li>
                    <li>Lite Pose Model</li>
                </ul>
            </div>
        </div>
    </div>

    <footer>
        Violence Detection System - Optimized Edition
    </footer>

    <script>
        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    // Update values
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('inference-ms').textContent = data.inference_ms.toFixed(1) + ' ms';
                    document.getElementById('violence-events').textContent = data.violence_events;
                    document.getElementById('frames-processed').textContent = data.frames_processed;
                    document.getElementById('uptime').textContent = data.uptime + 's';
                    document.getElementById('model-type').textContent = data.model_type;

                    // Update FPS bar (target: 30 FPS = 100%)
                    const fpsPercent = Math.min((data.fps / 30) * 100, 100);
                    document.getElementById('fps-bar').style.width = fpsPercent + '%';

                    // Update latency bar (target: <50ms = 100%, >200ms = 0%)
                    const latencyPercent = Math.max(0, Math.min(100, (1 - (data.inference_ms - 10) / 190) * 100));
                    document.getElementById('latency-bar').style.width = latencyPercent + '%';

                    // Update status indicator
                    const statusEl = document.getElementById('status-indicator');
                    statusEl.textContent = data.status;

                    if (data.status.includes('VIOLENCE')) {
                        statusEl.className = 'status-indicator status-danger';
                    } else if (data.status.includes('Warming')) {
                        statusEl.className = 'status-indicator status-warning';
                    } else {
                        statusEl.className = 'status-indicator status-safe';
                    }
                })
                .catch(err => console.error('Stats update error:', err));
        }

        // Update stats every 500ms
        setInterval(updateStats, 500);
        updateStats();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve main dashboard."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stats')
def api_stats():
    """API endpoint for stats."""
    with stats_lock:
        return jsonify(stats)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Optimized Violence Detection Dashboard")
    print("=" * 60)
    print(f"Video source: {VIDEO_SOURCE}")
    print(f"Server: http://localhost:{PORT}")
    print("=" * 60)

    # Initialize detector
    init_detector()

    # Start Flask
    print(f"\nStarting web server at http://localhost:{PORT}")
    print("Open this URL in your browser to view the dashboard")
    print("Press Ctrl+C to stop")

    app.run(host=HOST, port=PORT, threaded=True, debug=False)


if __name__ == '__main__':
    main()
