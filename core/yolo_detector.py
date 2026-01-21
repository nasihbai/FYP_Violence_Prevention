"""
YOLO-based Multi-Person Detection Module
=========================================
Uses YOLOv8 for detecting multiple people in frames.
Provides bounding boxes and tracking IDs for each person.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """Data class for storing person detection results."""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]
    crop: Optional[np.ndarray] = None


class YOLODetector:
    """
    YOLO-based person detector with optional tracking.

    Features:
    - Multi-person detection using YOLOv8
    - Optional person tracking with unique IDs
    - Configurable confidence thresholds
    - Efficient batch processing
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "auto",
        enable_tracking: bool = True
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model or model name (yolov8n.pt, yolov8s.pt, etc.)
            confidence: Minimum confidence threshold for detections
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            enable_tracking: Enable person tracking with unique IDs
        """
        self.confidence = confidence
        self.enable_tracking = enable_tracking
        self.model = None
        self.tracker_history: Dict[int, List[Tuple[int, int]]] = {}
        self.next_id = 0

        self._load_model(model_path, device)

    def _load_model(self, model_path: str, device: str):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

            # Set device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.device = device
            logger.info(f"YOLO model loaded successfully on {device}")

        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray, extract_crops: bool = True) -> List[PersonDetection]:
        """
        Detect people in a frame.

        Args:
            frame: Input frame (BGR format)
            extract_crops: Whether to extract cropped images of each person

        Returns:
            List of PersonDetection objects
        """
        if self.model is None:
            logger.error("YOLO model not loaded")
            return []

        detections = []

        try:
            # Run YOLO inference
            # Class 0 is 'person' in COCO dataset
            if self.enable_tracking:
                results = self.model.track(
                    frame,
                    classes=[0],
                    conf=self.confidence,
                    persist=True,
                    verbose=False
                )
            else:
                results = self.model(
                    frame,
                    classes=[0],
                    conf=self.confidence,
                    verbose=False
                )

            if results and len(results) > 0:
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())

                        # Get tracking ID if available
                        if self.enable_tracking and box.id is not None:
                            track_id = int(box.id[0].cpu().numpy())
                        else:
                            track_id = i

                        # Calculate center point
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Extract crop if requested
                        crop = None
                        if extract_crops:
                            # Add padding to crop
                            h, w = frame.shape[:2]
                            pad = 10
                            crop_x1 = max(0, x1 - pad)
                            crop_y1 = max(0, y1 - pad)
                            crop_x2 = min(w, x2 + pad)
                            crop_y2 = min(h, y2 + pad)
                            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                        detection = PersonDetection(
                            id=track_id,
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            center=(center_x, center_y),
                            crop=crop
                        )
                        detections.append(detection)

                        # Update tracker history
                        if track_id not in self.tracker_history:
                            self.tracker_history[track_id] = []
                        self.tracker_history[track_id].append((center_x, center_y))

                        # Keep only last 30 positions
                        if len(self.tracker_history[track_id]) > 30:
                            self.tracker_history[track_id].pop(0)

        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")

        return detections

    def detect_interactions(
        self,
        detections: List[PersonDetection],
        distance_threshold: int = 100
    ) -> List[Tuple[int, int, float]]:
        """
        Detect potential interactions between people based on proximity.

        Args:
            detections: List of person detections
            distance_threshold: Maximum distance (pixels) to consider as interaction

        Returns:
            List of tuples (person1_id, person2_id, distance)
        """
        interactions = []

        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i + 1:], start=i + 1):
                # Calculate distance between centers
                dist = np.sqrt(
                    (det1.center[0] - det2.center[0]) ** 2 +
                    (det1.center[1] - det2.center[1]) ** 2
                )

                if dist < distance_threshold:
                    interactions.append((det1.id, det2.id, dist))

        return interactions

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[PersonDetection],
        labels: Optional[Dict[int, str]] = None,
        colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        show_tracking_trail: bool = False
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on frame.

        Args:
            frame: Input frame
            detections: List of person detections
            labels: Optional dict mapping person ID to label string
            colors: Optional dict mapping person ID to BGR color
            show_tracking_trail: Whether to show tracking trail

        Returns:
            Frame with annotations
        """
        annotated_frame = frame.copy()

        default_color = (0, 255, 0)  # Green

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Get color
            color = colors.get(det.id, default_color) if colors else default_color

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label_text = f"ID:{det.id}"
            if labels and det.id in labels:
                label_text += f" {labels[det.id]}"
            label_text += f" {det.confidence:.2f}"

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated_frame,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Draw tracking trail if enabled
            if show_tracking_trail and det.id in self.tracker_history:
                trail = self.tracker_history[det.id]
                for k in range(1, len(trail)):
                    thickness = int(np.sqrt(30 / float(len(trail) - k + 1)) * 2)
                    cv2.line(
                        annotated_frame,
                        trail[k - 1],
                        trail[k],
                        color,
                        thickness
                    )

        return annotated_frame

    def reset_tracking(self):
        """Reset tracking history."""
        self.tracker_history.clear()
        self.next_id = 0


class SimplePersonDetector:
    """
    Fallback person detector using OpenCV's HOG descriptor.
    Use this when YOLO is not available.
    """

    def __init__(self):
        """Initialize HOG person detector."""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.next_id = 0

    def detect(self, frame: np.ndarray, extract_crops: bool = True) -> List[PersonDetection]:
        """
        Detect people using HOG descriptor.

        Args:
            frame: Input frame
            extract_crops: Whether to extract cropped images

        Returns:
            List of PersonDetection objects
        """
        detections = []

        # Resize for faster processing
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )

        for i, ((x, y, w, h), weight) in enumerate(zip(boxes, weights)):
            # Scale back to original size
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + w) / scale)
            y2 = int((y + h) / scale)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            crop = None
            if extract_crops:
                h_frame, w_frame = frame.shape[:2]
                crop_x1 = max(0, x1)
                crop_y1 = max(0, y1)
                crop_x2 = min(w_frame, x2)
                crop_y2 = min(h_frame, y2)
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            detection = PersonDetection(
                id=self.next_id,
                bbox=(x1, y1, x2, y2),
                confidence=float(weight),
                center=(center_x, center_y),
                crop=crop
            )
            detections.append(detection)
            self.next_id += 1

        return detections


def create_detector(use_yolo: bool = True, **kwargs) -> YOLODetector:
    """
    Factory function to create appropriate detector.

    Args:
        use_yolo: Whether to use YOLO (falls back to HOG if unavailable)
        **kwargs: Additional arguments for detector

    Returns:
        Detector instance
    """
    if use_yolo:
        try:
            return YOLODetector(**kwargs)
        except ImportError:
            logger.warning("YOLO not available, falling back to HOG detector")
            return SimplePersonDetector()
    else:
        return SimplePersonDetector()
