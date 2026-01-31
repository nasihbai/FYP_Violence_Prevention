"""
Advanced Feature Engineering for Violence Detection
====================================================
Transforms raw pose landmarks into discriminative features:
- Velocity (frame-to-frame motion)
- Acceleration (velocity changes)
- Joint angles between connected landmarks
- Normalized/scale-invariant poses
- Relative distances between key body parts
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# MediaPipe pose landmark indices
LANDMARK_INDICES = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32,
}

# Key joints for violence detection (hands, arms, head)
VIOLENCE_RELEVANT_JOINTS = [
    'nose', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
]

# Joint connections for angle computation
JOINT_ANGLES = [
    ('left_shoulder', 'left_elbow', 'left_wrist'),    # Left arm angle
    ('right_shoulder', 'right_elbow', 'right_wrist'), # Right arm angle
    ('left_hip', 'left_shoulder', 'left_elbow'),      # Left shoulder angle
    ('right_hip', 'right_shoulder', 'right_elbow'),   # Right shoulder angle
    ('left_shoulder', 'left_hip', 'left_knee'),       # Left body angle
    ('right_shoulder', 'right_hip', 'right_knee'),    # Right body angle
]

# Important distances for violence (fist-to-head, arm extension, etc.)
VIOLENCE_DISTANCES = [
    ('left_wrist', 'nose'),        # Left fist to head
    ('right_wrist', 'nose'),       # Right fist to head
    ('left_wrist', 'right_wrist'), # Hands together (grappling)
    ('left_elbow', 'left_hip'),    # Arm retraction (punch windup)
    ('right_elbow', 'right_hip'),  # Arm retraction
    ('left_shoulder', 'right_shoulder'),  # Shoulder width (normalization ref)
]


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    use_velocity: bool = True
    use_acceleration: bool = True
    use_angles: bool = True
    use_distances: bool = True
    use_normalized_coords: bool = True
    use_raw_coords: bool = False  # Can disable to reduce features
    num_landmarks: int = 33
    features_per_landmark: int = 4  # x, y, z, visibility


class FeatureExtractor:
    """
    Advanced feature extractor for pose sequences.
    Transforms raw landmarks into discriminative features for violence detection.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._compute_feature_dim()

    def _compute_feature_dim(self):
        """Compute total feature dimension based on config."""
        dim = 0

        if self.config.use_raw_coords:
            dim += self.config.num_landmarks * self.config.features_per_landmark  # 132

        if self.config.use_normalized_coords:
            dim += self.config.num_landmarks * 3  # 99 (x, y, z normalized)

        if self.config.use_velocity:
            dim += self.config.num_landmarks * 3  # 99

        if self.config.use_acceleration:
            dim += self.config.num_landmarks * 3  # 99

        if self.config.use_angles:
            dim += len(JOINT_ANGLES)  # 6

        if self.config.use_distances:
            dim += len(VIOLENCE_DISTANCES)  # 6

        self.feature_dim = dim

    def get_feature_dim(self) -> int:
        """Return the total feature dimension."""
        return self.feature_dim

    def extract_features(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """
        Extract advanced features from a sequence of landmarks.

        Args:
            landmarks_sequence: Array of shape (sequence_length, 132)
                               Each frame has 33 landmarks x 4 values (x, y, z, vis)

        Returns:
            Feature array of shape (sequence_length, feature_dim)
        """
        seq_len = len(landmarks_sequence)
        features_list = []

        # Reshape to (seq_len, 33, 4)
        landmarks_3d = landmarks_sequence.reshape(seq_len, self.config.num_landmarks, 4)

        # Extract x, y, z coordinates
        coords = landmarks_3d[:, :, :3]  # (seq_len, 33, 3)
        visibility = landmarks_3d[:, :, 3]  # (seq_len, 33)

        # 1. Raw coordinates (optional)
        if self.config.use_raw_coords:
            features_list.append(landmarks_sequence)

        # 2. Normalized coordinates (scale-invariant)
        if self.config.use_normalized_coords:
            normalized = self._normalize_poses(coords)
            features_list.append(normalized.reshape(seq_len, -1))

        # 3. Velocity (frame-to-frame motion)
        if self.config.use_velocity:
            velocity = self._compute_velocity(coords)
            features_list.append(velocity.reshape(seq_len, -1))

        # 4. Acceleration (velocity changes)
        if self.config.use_acceleration:
            acceleration = self._compute_acceleration(coords)
            features_list.append(acceleration.reshape(seq_len, -1))

        # 5. Joint angles
        if self.config.use_angles:
            angles = self._compute_angles(coords)
            features_list.append(angles)

        # 6. Important distances
        if self.config.use_distances:
            distances = self._compute_distances(coords)
            features_list.append(distances)

        # Concatenate all features
        features = np.concatenate(features_list, axis=1)

        return features.astype(np.float32)

    def _normalize_poses(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize poses to be scale and translation invariant.

        Centers at hip midpoint, scales by shoulder width.
        """
        seq_len = coords.shape[0]
        normalized = np.zeros_like(coords)

        left_hip_idx = LANDMARK_INDICES['left_hip']
        right_hip_idx = LANDMARK_INDICES['right_hip']
        left_shoulder_idx = LANDMARK_INDICES['left_shoulder']
        right_shoulder_idx = LANDMARK_INDICES['right_shoulder']

        for i in range(seq_len):
            # Center at hip midpoint
            hip_center = (coords[i, left_hip_idx] + coords[i, right_hip_idx]) / 2
            centered = coords[i] - hip_center

            # Scale by shoulder width
            shoulder_width = np.linalg.norm(
                coords[i, left_shoulder_idx] - coords[i, right_shoulder_idx]
            )
            if shoulder_width > 0.01:  # Avoid division by zero
                normalized[i] = centered / shoulder_width
            else:
                normalized[i] = centered

        return normalized

    def _compute_velocity(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute frame-to-frame velocity (motion).

        Higher velocity in hands/arms indicates aggressive motion.
        """
        seq_len = coords.shape[0]
        velocity = np.zeros_like(coords)

        # First frame has zero velocity
        velocity[1:] = coords[1:] - coords[:-1]

        # Scale velocity to reasonable range
        velocity = velocity * 10.0  # Amplify small movements

        return velocity

    def _compute_acceleration(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute acceleration (change in velocity).

        Sudden accelerations are indicative of strikes/punches.
        """
        seq_len = coords.shape[0]
        velocity = np.zeros_like(coords)
        velocity[1:] = coords[1:] - coords[:-1]

        acceleration = np.zeros_like(coords)
        acceleration[2:] = velocity[2:] - velocity[1:-1]

        # Scale acceleration
        acceleration = acceleration * 10.0

        return acceleration

    def _compute_angles(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute joint angles between connected body parts.

        Extended arms (180 deg) vs bent arms indicate different poses.
        """
        seq_len = coords.shape[0]
        angles = np.zeros((seq_len, len(JOINT_ANGLES)))

        for frame_idx in range(seq_len):
            for angle_idx, (j1_name, j2_name, j3_name) in enumerate(JOINT_ANGLES):
                j1 = coords[frame_idx, LANDMARK_INDICES[j1_name]]
                j2 = coords[frame_idx, LANDMARK_INDICES[j2_name]]
                j3 = coords[frame_idx, LANDMARK_INDICES[j3_name]]

                angle = self._angle_between_points(j1, j2, j3)
                angles[frame_idx, angle_idx] = angle / 180.0  # Normalize to [0, 1]

        return angles

    def _compute_distances(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute distances between key body parts.

        Fist-to-head distance is critical for punch detection.
        """
        seq_len = coords.shape[0]
        distances = np.zeros((seq_len, len(VIOLENCE_DISTANCES)))

        for frame_idx in range(seq_len):
            for dist_idx, (j1_name, j2_name) in enumerate(VIOLENCE_DISTANCES):
                j1 = coords[frame_idx, LANDMARK_INDICES[j1_name]]
                j2 = coords[frame_idx, LANDMARK_INDICES[j2_name]]

                dist = np.linalg.norm(j1 - j2)
                distances[frame_idx, dist_idx] = dist

        return distances

    @staticmethod
    def _angle_between_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute angle at p2 formed by p1-p2-p3.

        Returns angle in degrees [0, 180].
        """
        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)
        return np.degrees(angle)


def extract_features_from_sequence(
    landmarks_sequence: np.ndarray,
    config: Optional[FeatureConfig] = None
) -> np.ndarray:
    """
    Convenience function to extract features from a single sequence.

    Args:
        landmarks_sequence: Array of shape (sequence_length, 132)
        config: Feature extraction configuration

    Returns:
        Feature array
    """
    extractor = FeatureExtractor(config)
    return extractor.extract_features(landmarks_sequence)


def extract_features_from_dataset(
    X: np.ndarray,
    config: Optional[FeatureConfig] = None
) -> np.ndarray:
    """
    Extract features from entire dataset.

    Args:
        X: Array of shape (num_samples, sequence_length, 132)
        config: Feature extraction configuration

    Returns:
        Feature array of shape (num_samples, sequence_length, feature_dim)
    """
    extractor = FeatureExtractor(config)
    num_samples = len(X)

    # Process first sample to get output shape
    sample_features = extractor.extract_features(X[0])
    feature_dim = sample_features.shape[1]

    # Allocate output array
    X_features = np.zeros((num_samples, X.shape[1], feature_dim), dtype=np.float32)
    X_features[0] = sample_features

    # Process remaining samples
    for i in range(1, num_samples):
        X_features[i] = extractor.extract_features(X[i])

    return X_features


# Quick feature extraction for real-time use
class RealTimeFeatureExtractor:
    """
    Optimized feature extractor for real-time inference.
    Maintains state for velocity/acceleration computation.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.extractor = FeatureExtractor(self.config)
        self.prev_coords = None
        self.prev_velocity = None

    def reset(self):
        """Reset state for new person/sequence."""
        self.prev_coords = None
        self.prev_velocity = None

    def extract_frame_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract features for a single frame, using previous frame for velocity.

        Args:
            landmarks: Array of shape (132,) - single frame landmarks

        Returns:
            Feature array for single frame
        """
        landmarks_3d = landmarks.reshape(33, 4)
        coords = landmarks_3d[:, :3]

        features_list = []

        # Raw coordinates
        if self.config.use_raw_coords:
            features_list.append(landmarks)

        # Normalized coordinates
        if self.config.use_normalized_coords:
            normalized = self._normalize_single_pose(coords)
            features_list.append(normalized.flatten())

        # Velocity
        if self.config.use_velocity:
            if self.prev_coords is not None:
                velocity = (coords - self.prev_coords) * 10.0
            else:
                velocity = np.zeros_like(coords)
            features_list.append(velocity.flatten())

        # Acceleration
        if self.config.use_acceleration:
            if self.prev_coords is not None:
                curr_velocity = coords - self.prev_coords
                if self.prev_velocity is not None:
                    acceleration = (curr_velocity - self.prev_velocity) * 10.0
                else:
                    acceleration = np.zeros_like(coords)
                self.prev_velocity = curr_velocity
            else:
                acceleration = np.zeros_like(coords)
            features_list.append(acceleration.flatten())

        # Angles
        if self.config.use_angles:
            angles = self._compute_single_frame_angles(coords)
            features_list.append(angles)

        # Distances
        if self.config.use_distances:
            distances = self._compute_single_frame_distances(coords)
            features_list.append(distances)

        # Update state
        self.prev_coords = coords.copy()

        return np.concatenate(features_list).astype(np.float32)

    def _normalize_single_pose(self, coords: np.ndarray) -> np.ndarray:
        """Normalize a single pose."""
        left_hip = coords[LANDMARK_INDICES['left_hip']]
        right_hip = coords[LANDMARK_INDICES['right_hip']]
        hip_center = (left_hip + right_hip) / 2
        centered = coords - hip_center

        left_shoulder = coords[LANDMARK_INDICES['left_shoulder']]
        right_shoulder = coords[LANDMARK_INDICES['right_shoulder']]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_width > 0.01:
            return centered / shoulder_width
        return centered

    def _compute_single_frame_angles(self, coords: np.ndarray) -> np.ndarray:
        """Compute angles for single frame."""
        angles = np.zeros(len(JOINT_ANGLES))
        for i, (j1_name, j2_name, j3_name) in enumerate(JOINT_ANGLES):
            j1 = coords[LANDMARK_INDICES[j1_name]]
            j2 = coords[LANDMARK_INDICES[j2_name]]
            j3 = coords[LANDMARK_INDICES[j3_name]]
            angles[i] = FeatureExtractor._angle_between_points(j1, j2, j3) / 180.0
        return angles

    def _compute_single_frame_distances(self, coords: np.ndarray) -> np.ndarray:
        """Compute distances for single frame."""
        distances = np.zeros(len(VIOLENCE_DISTANCES))
        for i, (j1_name, j2_name) in enumerate(VIOLENCE_DISTANCES):
            j1 = coords[LANDMARK_INDICES[j1_name]]
            j2 = coords[LANDMARK_INDICES[j2_name]]
            distances[i] = np.linalg.norm(j1 - j2)
        return distances


# Default configuration for best performance
DEFAULT_CONFIG = FeatureConfig(
    use_velocity=True,
    use_acceleration=True,
    use_angles=True,
    use_distances=True,
    use_normalized_coords=True,
    use_raw_coords=False,  # Disable raw to reduce feature size
)

# Lightweight config for faster inference
LIGHTWEIGHT_CONFIG = FeatureConfig(
    use_velocity=True,
    use_acceleration=False,
    use_angles=True,
    use_distances=True,
    use_normalized_coords=True,
    use_raw_coords=False,
)
