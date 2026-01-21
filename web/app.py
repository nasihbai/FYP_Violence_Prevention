"""
Flask Web Dashboard for Violence Detection System
=================================================
Real-time web interface for monitoring and managing violence detection.

Features:
- Live video streaming
- Detection statistics
- Alert history
- System configuration
- API endpoints for integration
"""

import os
import sys
import cv2
import json
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WebConfig, VideoConfig, AlertConfig
from core.detection_engine import ThreadSafeDetector, FrameResult

logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.config['SECRET_KEY'] = WebConfig.SECRET_KEY
CORS(app)

# Global state
detector: ThreadSafeDetector = None
video_source = None
is_running = False
current_frame = None
frame_lock = threading.Lock()
stats = {
    'total_frames': 0,
    'violence_detections': 0,
    'alerts_triggered': 0,
    'start_time': None,
    'current_fps': 0
}
alert_history = []


def initialize_detector(model_path: str = None, source=0, use_yolo: bool = True):
    """Initialize the detection system."""
    global detector, video_source

    video_source = source

    # Use default model path if not provided
    if model_path is None:
        model_path = str(Path(__file__).parent.parent / 'models' / 'violence_lstm_enhanced.h5')
        # Fall back to original model
        if not Path(model_path).exists():
            model_path = str(Path(__file__).parent.parent / 'lstm-model.h5')

    detector = ThreadSafeDetector(
        lstm_model_path=model_path if Path(model_path).exists() else None,
        use_yolo=use_yolo
    )
    detector.start()

    logger.info(f"Detector initialized with source: {source}")


def generate_frames():
    """Generator for video streaming."""
    global current_frame, is_running, stats

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        return

    is_running = True
    stats['start_time'] = datetime.now()

    try:
        while is_running:
            ret, frame = cap.read()

            if not ret:
                # Try to reconnect or loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Process frame
            result = detector.process_frame(frame)
            stats['total_frames'] += 1
            stats['current_fps'] = result.fps

            # Check for violence
            if result.has_violence:
                stats['violence_detections'] += 1

                # Add to alert history
                for det in result.detections:
                    if det.is_violent:
                        alert_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'person_id': det.person_id,
                            'confidence': det.confidence,
                            'bbox': det.bbox
                        })
                        # Keep only last 100 alerts
                        if len(alert_history) > 100:
                            alert_history.pop(0)

            # Draw results
            annotated = detector.draw_results(frame, result)

            # Store current frame
            with frame_lock:
                current_frame = annotated.copy()

            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, WebConfig.STREAM_QUALITY])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Limit frame rate
            time.sleep(1.0 / WebConfig.STREAM_FPS)

    finally:
        cap.release()
        is_running = False


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stats')
def get_stats():
    """Get current statistics."""
    uptime = None
    if stats['start_time']:
        uptime = str(datetime.now() - stats['start_time']).split('.')[0]

    return jsonify({
        'total_frames': stats['total_frames'],
        'violence_detections': stats['violence_detections'],
        'alerts_triggered': len(alert_history),
        'current_fps': round(stats['current_fps'], 1),
        'uptime': uptime,
        'is_running': is_running
    })


@app.route('/api/alerts')
def get_alerts():
    """Get alert history."""
    limit = request.args.get('limit', 50, type=int)
    return jsonify(alert_history[-limit:])


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration."""
    if request.method == 'GET':
        return jsonify({
            'video_source': video_source,
            'violence_threshold': detector.violence_threshold if detector else 0.6,
            'use_yolo': detector.use_yolo if detector else True,
            'warmup_frames': detector.warmup_frames if detector else 30
        })
    else:
        # Update configuration
        data = request.json
        if detector:
            if 'violence_threshold' in data:
                detector.violence_threshold = data['violence_threshold']
        return jsonify({'status': 'updated'})


@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start detection."""
    global is_running
    if not is_running:
        threading.Thread(target=lambda: list(generate_frames()), daemon=True).start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop detection."""
    global is_running
    is_running = False
    return jsonify({'status': 'stopped'})


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """Reset statistics."""
    global stats, alert_history
    stats = {
        'total_frames': 0,
        'violence_detections': 0,
        'alerts_triggered': 0,
        'start_time': datetime.now() if is_running else None,
        'current_fps': 0
    }
    alert_history = []
    if detector:
        detector.reset()
    return jsonify({'status': 'reset'})


@app.route('/api/snapshot')
def snapshot():
    """Get current frame as image."""
    with frame_lock:
        if current_frame is not None:
            _, buffer = cv2.imencode('.jpg', current_frame)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return jsonify({'error': 'No frame available'}), 404


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'detector_loaded': detector is not None,
        'is_running': is_running
    })


# ==================== MAIN ====================

def create_app(model_path: str = None, source=0, use_yolo: bool = True):
    """Create and configure Flask app."""
    initialize_detector(model_path, source, use_yolo)
    return app


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Run the Flask server."""
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Violence Detection Web Dashboard')
    parser.add_argument('--model', type=str, help='Path to LSTM model')
    parser.add_argument('--source', default=0, help='Video source (camera index or file path)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    # Convert source to int if it's a camera index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    initialize_detector(args.model, source, not args.no_yolo)
    run_server(args.host, args.port, args.debug)
