"""
Optimized Real-Time Violence Detector
=====================================
High-performance detection with all optimizations:
- TFLite inference for 3-5x speedup
- Reduced sequence length (15 frames)
- Advanced feature engineering
- Optimized temporal smoothing
- Early detection for high-confidence predictions
- Efficient pose extraction
"""

import os
import time
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from threading import Lock, Thread
from queue import Queue, Empty
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Configuration for optimized detector."""
    # Model settings
    model_path: str = ""
    use_tflite: bool = True  # Use TFLite for faster inference

    # Sequence settings
    sequence_length: int = 15  # Reduced from 20
    prediction_stride: int = 5  # Predict every N frames (not every sequence_length)

    # Thresholds
    violence_threshold: float = 0.6
    early_detection_threshold: float = 0.9  # Immediate alert for very high confidence

    # Smoothing
    smoothing_window: int = 3  # Reduced from 5
    use_exponential_smoothing: bool = True
    smoothing_alpha: float = 0.3  # For exponential smoothing

    # Performance
    warmup_frames: int = 20  # Reduced from 60
    skip_frames: int = 0  # Process every Nth frame (0 = process all)
    max_persons: int = 5

    # Pose extraction
    pose_model_complexity: int = 0  # 0=Lite, 1=Full, 2=Heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Feature engineering
    use_features: bool = True


@dataclass
class PersonState:
    """State tracking for a single person."""
    person_id: int
    landmark_buffer: deque = field(default_factory=lambda: deque(maxlen=15))
    feature_buffer: deque = field(default_factory=lambda: deque(maxlen=15))
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=3))
    smoothed_prediction: float = 0.0
    last_prediction_time: float = 0.0
    frames_since_prediction: int = 0
    is_violent: bool = False
    confidence: float = 0.0


@dataclass
class DetectionResult:
    """Result for a single detected person."""
    person_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    is_violent: bool
    confidence: float
    landmarks: Optional[np.ndarray] = None


@dataclass
class FrameResult:
    """Result for entire frame."""
    frame: np.ndarray
    detections: List[DetectionResult]
    fps: float
    inference_time_ms: float
    has_violence: bool
    timestamp: float


class OptimizedDetector:
    """
    High-performance violence detector with all optimizations.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._setup_model()
        self._setup_pose()
        self._setup_state()

    def _setup_model(self):
        """Initialize the inference model."""
        model_path = self.config.model_path

        # Find model if not specified
        if not model_path:
            base_dir = Path(__file__).parent.parent
            tflite_path = base_dir / "models" / "optimized_model.tflite"
            h5_path = base_dir / "models" / "optimized_model.h5"
            fallback_path = base_dir / "lstm-model.h5"
            fallback_rwf_path = base_dir / "models" / "violence_lstm_rwf2000.h5"

            # Prefer Keras model over TFLite for reliability
            # TFLite conversion may have compatibility issues
            if h5_path.exists():
                model_path = str(h5_path)
                self.config.use_tflite = False
            elif self.config.use_tflite and tflite_path.exists():
                model_path = str(tflite_path)
            elif fallback_rwf_path.exists():
                model_path = str(fallback_rwf_path)
                self.config.use_tflite = False
                self.config.sequence_length = 20  # Original model uses 20 frames
            elif fallback_path.exists():
                model_path = str(fallback_path)
                self.config.use_tflite = False
                self.config.sequence_length = 20  # Original model uses 20 frames
            else:
                raise FileNotFoundError("No model found. Train a model first.")

        self.model_path = model_path
        logger.info(f"Loading model from {model_path}")

        if self.config.use_tflite and model_path.endswith('.tflite'):
            self._setup_tflite(model_path)
        else:
            self._setup_keras(model_path)

    def _setup_tflite(self, model_path: str):
        """Setup TFLite interpreter for fast inference."""
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']

        # Auto-detect if model needs features based on input shape
        # Raw landmarks = 132 features, with feature engineering > 132
        expected_features = self.input_shape[-1]
        if expected_features == 132:
            self.config.use_features = False
            logger.info("Model expects raw landmarks (132 features)")
        else:
            self.config.use_features = True
            logger.info(f"Model expects engineered features ({expected_features} features)")

        self.use_tflite = True
        logger.info(f"TFLite model loaded. Input shape: {self.input_shape}")

    def _setup_keras(self, model_path: str):
        """Setup Keras model (fallback)."""
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

        # Auto-detect if model needs features based on input shape
        input_shape = self.model.input_shape
        expected_features = input_shape[-1] if input_shape[-1] is not None else 132
        if expected_features == 132:
            self.config.use_features = False
            logger.info("Model expects raw landmarks (132 features)")
        else:
            self.config.use_features = True
            logger.info(f"Model expects engineered features ({expected_features} features)")

        self.use_tflite = False
        logger.info(f"Keras model loaded. Input shape: {self.model.input_shape}")

    def _setup_pose(self):
        """Initialize MediaPipe pose detector."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.pose_model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def _setup_state(self):
        """Initialize tracking state."""
        self.person_states: Dict[int, PersonState] = {}
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0
        self.lock = Lock()

        # Feature extractor - use DEFAULT_CONFIG to match training
        if self.config.use_features:
            from core.feature_engineering import RealTimeFeatureExtractor, DEFAULT_CONFIG
            self.feature_extractor = RealTimeFeatureExtractor(DEFAULT_CONFIG)
        else:
            self.feature_extractor = None

    def _predict(self, sequence: np.ndarray) -> float:
        """Run inference on a sequence."""
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        if self.use_tflite:
            sequence = sequence.astype(self.input_dtype)
            self.interpreter.set_tensor(self.input_details[0]['index'], sequence)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            return float(output[0][0])
        else:
            output = self.model.predict(sequence, verbose=0)
            return float(output[0][0])

    def _smooth_prediction(self, person_state: PersonState, new_prediction: float) -> float:
        """Apply temporal smoothing to predictions."""
        if self.config.use_exponential_smoothing:
            # Exponential moving average
            alpha = self.config.smoothing_alpha
            smoothed = alpha * new_prediction + (1 - alpha) * person_state.smoothed_prediction
        else:
            # Simple moving average
            person_state.prediction_history.append(new_prediction)
            smoothed = np.mean(list(person_state.prediction_history))

        person_state.smoothed_prediction = smoothed
        return smoothed

    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose landmarks from frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks is None:
            return None

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

        return np.array(landmarks, dtype=np.float32)

    def _get_bounding_box(self, landmarks: np.ndarray, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Get bounding box from landmarks."""
        h, w = frame_shape[:2]

        # Reshape to (33, 4)
        lm_array = landmarks.reshape(33, 4)

        # Get x, y coordinates
        x_coords = lm_array[:, 0] * w
        y_coords = lm_array[:, 1] * h

        # Filter visible landmarks
        visibility = lm_array[:, 3]
        visible = visibility > 0.5

        if not np.any(visible):
            return (0, 0, w, h)

        x_coords = x_coords[visible]
        y_coords = y_coords[visible]

        padding = 20
        x1 = max(0, int(np.min(x_coords)) - padding)
        y1 = max(0, int(np.min(y_coords)) - padding)
        x2 = min(w, int(np.max(x_coords)) + padding)
        y2 = min(h, int(np.max(y_coords)) + padding)

        return (x1, y1, x2, y2)

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single frame for violence detection.

        Args:
            frame: BGR image from OpenCV

        Returns:
            FrameResult with detections and metadata
        """
        start_time = time.perf_counter()
        self.frame_count += 1

        # Skip frames if configured
        if self.config.skip_frames > 0 and self.frame_count % (self.config.skip_frames + 1) != 0:
            # Return previous result without processing
            return FrameResult(
                frame=frame,
                detections=[],
                fps=self.current_fps,
                inference_time_ms=0,
                has_violence=False,
                timestamp=time.time()
            )

        # Update FPS
        self._update_fps()

        detections = []
        has_violence = False

        # Warmup period
        if self.frame_count <= self.config.warmup_frames:
            inference_time = (time.perf_counter() - start_time) * 1000
            return FrameResult(
                frame=frame,
                detections=[],
                fps=self.current_fps,
                inference_time_ms=inference_time,
                has_violence=False,
                timestamp=time.time()
            )

        # Extract landmarks
        landmarks = self._extract_landmarks(frame)

        if landmarks is not None:
            # For now, single person tracking (ID = 0)
            person_id = 0

            with self.lock:
                # Get or create person state
                if person_id not in self.person_states:
                    self.person_states[person_id] = PersonState(
                        person_id=person_id,
                        landmark_buffer=deque(maxlen=self.config.sequence_length),
                        prediction_history=deque(maxlen=self.config.smoothing_window)
                    )

                person_state = self.person_states[person_id]

                # Add landmarks to buffer
                person_state.landmark_buffer.append(landmarks)
                person_state.frames_since_prediction += 1

                # Check if we should make a prediction
                should_predict = (
                    len(person_state.landmark_buffer) >= self.config.sequence_length and
                    person_state.frames_since_prediction >= self.config.prediction_stride
                )

                if should_predict:
                    # Get sequence
                    sequence = np.array(list(person_state.landmark_buffer))

                    # Apply feature engineering if enabled
                    if self.config.use_features and self.feature_extractor:
                        try:
                            from core.feature_engineering import extract_features_from_sequence, LIGHTWEIGHT_CONFIG
                            sequence = extract_features_from_sequence(sequence, LIGHTWEIGHT_CONFIG)
                        except Exception as e:
                            logger.warning(f"Feature extraction failed: {e}")

                    # Run prediction
                    raw_prediction = self._predict(sequence)
                    smoothed_prediction = self._smooth_prediction(person_state, raw_prediction)

                    # Update state
                    person_state.confidence = smoothed_prediction
                    person_state.is_violent = smoothed_prediction > self.config.violence_threshold
                    person_state.frames_since_prediction = 0

                    # Early detection for very high confidence
                    if raw_prediction > self.config.early_detection_threshold:
                        person_state.is_violent = True
                        person_state.confidence = raw_prediction

                # Get bounding box
                bbox = self._get_bounding_box(landmarks, frame.shape)

                # Create detection result
                detection = DetectionResult(
                    person_id=person_id,
                    bbox=bbox,
                    is_violent=person_state.is_violent,
                    confidence=person_state.confidence,
                    landmarks=landmarks
                )
                detections.append(detection)

                if person_state.is_violent:
                    has_violence = True

        inference_time = (time.perf_counter() - start_time) * 1000

        return FrameResult(
            frame=frame,
            detections=detections,
            fps=self.current_fps,
            inference_time_ms=inference_time,
            has_violence=has_violence,
            timestamp=time.time()
        )

    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_fps_time = current_time

    def draw_results(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw detection results on frame."""
        output = frame.copy()

        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox

            # Color based on violence status
            if detection.is_violent:
                color = (0, 0, 255)  # Red
                label = f"VIOLENT: {detection.confidence:.2f}"
            else:
                color = (0, 255, 0)  # Green
                label = f"Neutral: {detection.confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw skeleton if landmarks available
            if detection.landmarks is not None:
                self._draw_skeleton(output, detection.landmarks)

        # Draw FPS and inference time
        info_text = f"FPS: {result.fps:.1f} | Inference: {result.inference_time_ms:.1f}ms"
        cv2.putText(output, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Warning if warmup
        if self.frame_count <= self.config.warmup_frames:
            cv2.putText(output, f"Warming up... {self.frame_count}/{self.config.warmup_frames}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return output

    def _draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray):
        """Draw pose skeleton on frame."""
        h, w = frame.shape[:2]
        lm_array = landmarks.reshape(33, 4)

        # Draw key points
        for i, (x, y, z, vis) in enumerate(lm_array):
            if vis > 0.5:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Draw connections
        connections = self.mp_pose.POSE_CONNECTIONS
        for conn in connections:
            start_idx, end_idx = conn
            start_vis = lm_array[start_idx, 3]
            end_vis = lm_array[end_idx, 3]

            if start_vis > 0.5 and end_vis > 0.5:
                start_pt = (int(lm_array[start_idx, 0] * w), int(lm_array[start_idx, 1] * h))
                end_pt = (int(lm_array[end_idx, 0] * w), int(lm_array[end_idx, 1] * h))
                cv2.line(frame, start_pt, end_pt, (0, 255, 0), 1)

    def reset(self):
        """Reset detector state."""
        with self.lock:
            self.person_states.clear()
            self.frame_count = 0

        if self.feature_extractor:
            self.feature_extractor.reset()

    def close(self):
        """Release resources."""
        self.pose.close()


def run_detection(
    source: Any = 0,
    config: Optional[DetectorConfig] = None,
    show_window: bool = True
):
    """
    Run real-time violence detection.

    Args:
        source: Video source (0 for webcam, path for file, URL for stream)
        config: Detector configuration
        show_window: Whether to show OpenCV window
    """
    detector = OptimizedDetector(config)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    print("Violence Detection Started (Optimized)")
    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Handle video file looping
                if isinstance(source, str) and not source.startswith(('rtsp://', 'http://')):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    detector.reset()
                    continue
                break

            # Process frame
            result = detector.process_frame(frame)

            # Draw results
            output = detector.draw_results(frame, result)

            if show_window:
                cv2.imshow("Violence Detection (Optimized)", output)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        detector.close()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # Quick test
    config = DetectorConfig(
        use_tflite=True,
        sequence_length=15,
        prediction_stride=5,
        warmup_frames=20,
        smoothing_window=3,
        use_features=False,  # Disable for faster inference without trained feature model
    )

    run_detection(source=0, config=config)
