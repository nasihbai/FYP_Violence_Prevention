"""
Enhanced Data Collection Script
===============================
Collect pose landmark data for training violence detection models.

Features:
- Multi-camera support
- YOLO-based person detection
- Progress tracking
- Data validation

Usage:
    python collect_data.py --label neutral --frames 1000
    python collect_data.py --label violent --frames 500 --camera 1
"""

import os
import sys
import cv2
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mediapipe as mp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Enhanced pose data collector."""

    def __init__(
        self,
        camera_index: int = 0,
        output_dir: str = '.',
        use_yolo: bool = False
    ):
        """
        Initialize data collector.

        Args:
            camera_index: Camera device index
            output_dir: Directory to save collected data
            use_yolo: Use YOLO for person detection (multi-person)
        """
        self.camera_index = camera_index
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # YOLO detector
        self.use_yolo = use_yolo
        self.yolo_detector = None

        if use_yolo:
            try:
                from core.yolo_detector import YOLODetector
                self.yolo_detector = YOLODetector()
                logger.info("YOLO detector initialized")
            except Exception as e:
                logger.warning(f"YOLO not available: {e}")
                self.use_yolo = False

        # Data storage
        self.landmarks_list = []

    def extract_landmarks(self, frame: np.ndarray) -> list:
        """
        Extract pose landmarks from frame.

        Args:
            frame: BGR frame

        Returns:
            List of landmark values [x, y, z, visibility] * 33
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks is None:
            return None

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

        return landmarks

    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw pose landmarks on frame."""
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame

    def collect(
        self,
        label: str,
        num_frames: int = 1000,
        show_preview: bool = True,
        countdown: int = 3
    ) -> str:
        """
        Collect pose data.

        Args:
            label: Label for the action (e.g., 'neutral', 'violent')
            num_frames: Number of frames to collect
            show_preview: Show video preview
            countdown: Countdown seconds before starting

        Returns:
            Path to saved data file
        """
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return None

        logger.info(f"Collecting '{label}' data: {num_frames} frames")
        logger.info("Press 'q' to quit, 's' to skip frame, 'r' to restart")

        self.landmarks_list = []
        frame_count = 0
        start_time = None

        # Countdown
        if countdown > 0:
            for i in range(countdown, 0, -1):
                ret, frame = cap.read()
                if ret and show_preview:
                    cv2.putText(
                        frame, f"Starting in {i}...",
                        (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                    )
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(1000)

        logger.info("Recording started!")
        start_time = time.time()

        while frame_count < num_frames:
            ret, frame = cap.read()

            if not ret:
                logger.warning("Failed to read frame")
                continue

            # Extract landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract and store landmarks
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

                self.landmarks_list.append(landmarks)
                frame_count += 1

                # Draw landmarks
                frame = self.draw_landmarks(frame, results)

            # Show preview
            if show_preview:
                # Add info overlay
                progress = frame_count / num_frames * 100
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                info_text = [
                    f"Label: {label}",
                    f"Progress: {frame_count}/{num_frames} ({progress:.1f}%)",
                    f"FPS: {fps:.1f}",
                    "",
                    "Press 'q' to quit",
                    "Press 'r' to restart"
                ]

                y_offset = 30
                for text in info_text:
                    cv2.putText(
                        frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    y_offset += 25

                # Progress bar
                bar_width = int((frame.shape[1] - 40) * progress / 100)
                cv2.rectangle(frame, (20, frame.shape[0] - 30),
                              (frame.shape[1] - 20, frame.shape[0] - 10),
                              (50, 50, 50), -1)
                cv2.rectangle(frame, (20, frame.shape[0] - 30),
                              (20 + bar_width, frame.shape[0] - 10),
                              (0, 255, 0), -1)

                cv2.imshow('Data Collection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Collection cancelled by user")
                break
            elif key == ord('r'):
                logger.info("Restarting collection...")
                self.landmarks_list = []
                frame_count = 0
                start_time = time.time()

        cap.release()
        cv2.destroyAllWindows()

        # Save data
        if self.landmarks_list:
            output_path = self._save_data(label)
            logger.info(f"Collected {len(self.landmarks_list)} frames")
            logger.info(f"Data saved to: {output_path}")
            return output_path
        else:
            logger.warning("No data collected")
            return None

    def _save_data(self, label: str) -> str:
        """Save collected data to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{label}_{timestamp}.txt"
        filepath = self.output_dir / filename

        df = pd.DataFrame(self.landmarks_list)
        df.to_csv(filepath, index=True)

        # Also save/append to main label file
        main_filepath = self.output_dir / f"{label}.txt"
        if main_filepath.exists():
            existing_df = pd.read_csv(main_filepath, index_col=0)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(main_filepath)
            logger.info(f"Appended to existing file: {main_filepath}")
        else:
            df.to_csv(main_filepath)
            logger.info(f"Created new file: {main_filepath}")

        return str(filepath)

    def validate_data(self, filepath: str) -> dict:
        """
        Validate collected data.

        Args:
            filepath: Path to data file

        Returns:
            Validation report
        """
        df = pd.read_csv(filepath, index_col=0)

        report = {
            'filepath': filepath,
            'num_samples': len(df),
            'num_features': len(df.columns),
            'expected_features': 132,  # 33 landmarks * 4 values
            'has_nan': df.isna().any().any(),
            'nan_count': df.isna().sum().sum(),
            'feature_ranges': {}
        }

        # Check feature ranges (should be roughly 0-1 for x, y)
        for i in range(0, min(4, len(df.columns))):
            col = df.columns[i]
            report['feature_ranges'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean()
            }

        report['valid'] = (
            report['num_features'] == report['expected_features'] and
            not report['has_nan']
        )

        return report


def main():
    parser = argparse.ArgumentParser(description='Collect Pose Data for Training')
    parser.add_argument('--label', '-l', required=True, help='Action label (e.g., neutral, violent)')
    parser.add_argument('--frames', '-f', type=int, default=1000, help='Number of frames to collect')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory')
    parser.add_argument('--no-preview', action='store_true', help='Disable video preview')
    parser.add_argument('--countdown', type=int, default=3, help='Countdown seconds')
    parser.add_argument('--validate', type=str, help='Validate existing data file')

    args = parser.parse_args()

    collector = DataCollector(
        camera_index=args.camera,
        output_dir=args.output_dir
    )

    if args.validate:
        report = collector.validate_data(args.validate)
        print("\n=== Data Validation Report ===")
        for key, value in report.items():
            print(f"  {key}: {value}")
        return

    filepath = collector.collect(
        label=args.label,
        num_frames=args.frames,
        show_preview=not args.no_preview,
        countdown=args.countdown
    )

    if filepath:
        report = collector.validate_data(filepath)
        print("\n=== Data Validation Report ===")
        print(f"  Valid: {report['valid']}")
        print(f"  Samples: {report['num_samples']}")
        print(f"  Features: {report['num_features']}")


if __name__ == '__main__':
    main()
