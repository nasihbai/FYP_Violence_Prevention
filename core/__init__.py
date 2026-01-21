"""
Core Module Package
===================
Contains core detection and processing components.
"""

from .yolo_detector import YOLODetector, PersonDetection, SimplePersonDetector, create_detector
from .lstm_model import (
    AttentionLayer,
    TemporalBlock,
    create_enhanced_lstm_model,
    create_simple_lstm_model,
    ViolenceClassifier,
    get_training_callbacks
)
from .pose_extractor import PoseExtractor, PoseLandmarks, LandmarkBuffer, MultiPersonPoseExtractor
from .detection_engine import ThreadSafeDetector, VideoProcessor, DetectionResult, FrameResult

__all__ = [
    'YOLODetector',
    'PersonDetection',
    'SimplePersonDetector',
    'create_detector',
    'AttentionLayer',
    'TemporalBlock',
    'create_enhanced_lstm_model',
    'create_simple_lstm_model',
    'ViolenceClassifier',
    'get_training_callbacks',
    'PoseExtractor',
    'PoseLandmarks',
    'LandmarkBuffer',
    'MultiPersonPoseExtractor',
    'ThreadSafeDetector',
    'VideoProcessor',
    'DetectionResult',
    'FrameResult'
]
