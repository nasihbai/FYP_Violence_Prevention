"""
Data Augmentation Utilities
===========================
Provides data augmentation techniques for pose landmark data
to improve model generalization and handle class imbalance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path
import logging
from collections import Counter
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class PoseDataAugmenter:
    """
    Augmentation techniques for pose landmark sequences.

    Supports:
    - Scaling (zoom in/out)
    - Translation (shift position)
    - Rotation (2D rotation around center)
    - Temporal jittering (speed variation)
    - Noise injection
    - Horizontal flipping
    - Interpolation for smooth transitions
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize augmenter.

        Args:
            random_seed: Seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

    def scale(self, landmarks: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Scale landmarks (zoom effect).

        Args:
            landmarks: Shape (timesteps, features) or (features,)
            scale_range: Min and max scale factors

        Returns:
            Scaled landmarks
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        augmented = landmarks.copy()

        # Scale x, y, z coordinates (every 4th feature starting from 0, 1, 2)
        if landmarks.ndim == 2:
            for i in range(0, augmented.shape[1], 4):
                augmented[:, i:i+3] *= scale
        else:
            for i in range(0, len(augmented), 4):
                augmented[i:i+3] *= scale

        return augmented

    def translate(self, landmarks: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
        """
        Translate landmarks (shift position).

        Args:
            landmarks: Landmark array
            max_shift: Maximum shift as fraction of coordinate range

        Returns:
            Translated landmarks
        """
        shift_x = np.random.uniform(-max_shift, max_shift)
        shift_y = np.random.uniform(-max_shift, max_shift)

        augmented = landmarks.copy()

        if landmarks.ndim == 2:
            # Shift x coordinates (every 4th starting from 0)
            for i in range(0, augmented.shape[1], 4):
                augmented[:, i] += shift_x
            # Shift y coordinates (every 4th starting from 1)
            for i in range(1, augmented.shape[1], 4):
                augmented[:, i] += shift_y
        else:
            for i in range(0, len(augmented), 4):
                augmented[i] += shift_x
            for i in range(1, len(augmented), 4):
                augmented[i] += shift_y

        return augmented

    def rotate_2d(self, landmarks: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """
        Rotate landmarks in 2D plane.

        Args:
            landmarks: Landmark array
            max_angle: Maximum rotation angle in degrees

        Returns:
            Rotated landmarks
        """
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = np.deg2rad(angle)

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        augmented = landmarks.copy()

        # Find center for rotation
        if landmarks.ndim == 2:
            x_coords = landmarks[:, 0::4]
            y_coords = landmarks[:, 1::4]
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)

            for i in range(0, augmented.shape[1], 4):
                x = augmented[:, i] - center_x
                y = augmented[:, i+1] - center_y

                augmented[:, i] = x * cos_a - y * sin_a + center_x
                augmented[:, i+1] = x * sin_a + y * cos_a + center_y
        else:
            x_coords = landmarks[0::4]
            y_coords = landmarks[1::4]
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)

            for i in range(0, len(augmented), 4):
                x = augmented[i] - center_x
                y = augmented[i+1] - center_y

                augmented[i] = x * cos_a - y * sin_a + center_x
                augmented[i+1] = x * sin_a + y * cos_a + center_y

        return augmented

    def add_noise(self, landmarks: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to landmarks.

        Args:
            landmarks: Landmark array
            noise_level: Standard deviation of noise

        Returns:
            Noisy landmarks
        """
        noise = np.random.normal(0, noise_level, landmarks.shape)
        augmented = landmarks + noise

        # Don't add noise to visibility values (every 4th starting from 3)
        if landmarks.ndim == 2:
            for i in range(3, augmented.shape[1], 4):
                augmented[:, i] = landmarks[:, i]
        else:
            for i in range(3, len(augmented), 4):
                augmented[i] = landmarks[i]

        return augmented

    def horizontal_flip(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Flip landmarks horizontally (mirror).

        Args:
            landmarks: Landmark array

        Returns:
            Flipped landmarks
        """
        augmented = landmarks.copy()

        # Flip x coordinates (subtract from 1 since normalized to [0,1])
        if landmarks.ndim == 2:
            for i in range(0, augmented.shape[1], 4):
                augmented[:, i] = 1.0 - augmented[:, i]
        else:
            for i in range(0, len(augmented), 4):
                augmented[i] = 1.0 - augmented[i]

        # Swap left/right landmark pairs (MediaPipe pose)
        # Left/Right pairs: (11,12), (13,14), (15,16), (17,18), (19,20), (21,22), (23,24), (25,26), (27,28), (29,30), (31,32)
        swap_pairs = [
            (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
            (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
        ]

        for left, right in swap_pairs:
            left_idx = left * 4
            right_idx = right * 4

            if landmarks.ndim == 2:
                if left_idx + 4 <= augmented.shape[1] and right_idx + 4 <= augmented.shape[1]:
                    temp = augmented[:, left_idx:left_idx+4].copy()
                    augmented[:, left_idx:left_idx+4] = augmented[:, right_idx:right_idx+4]
                    augmented[:, right_idx:right_idx+4] = temp
            else:
                if left_idx + 4 <= len(augmented) and right_idx + 4 <= len(augmented):
                    temp = augmented[left_idx:left_idx+4].copy()
                    augmented[left_idx:left_idx+4] = augmented[right_idx:right_idx+4]
                    augmented[right_idx:right_idx+4] = temp

        return augmented

    def temporal_jitter(self, sequence: np.ndarray, speed_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Apply temporal jittering (speed up/slow down).

        Args:
            sequence: Shape (timesteps, features)
            speed_range: Range of speed factors

        Returns:
            Time-warped sequence with same length
        """
        if sequence.ndim != 2:
            return sequence

        speed = np.random.uniform(speed_range[0], speed_range[1])
        n_timesteps = sequence.shape[0]

        # Create warped time indices
        original_indices = np.arange(n_timesteps)
        warped_length = int(n_timesteps / speed)
        warped_indices = np.linspace(0, n_timesteps - 1, warped_length)

        # Interpolate each feature
        augmented = np.zeros_like(sequence)

        for feat_idx in range(sequence.shape[1]):
            interp_func = interp1d(original_indices, sequence[:, feat_idx], kind='linear', fill_value='extrapolate')
            warped_values = interp_func(warped_indices)

            # Resample back to original length
            resample_func = interp1d(np.arange(len(warped_values)), warped_values, kind='linear', fill_value='extrapolate')
            augmented[:, feat_idx] = resample_func(np.linspace(0, len(warped_values) - 1, n_timesteps))

        return augmented

    def augment_sequence(
        self,
        sequence: np.ndarray,
        augmentation_types: Optional[List[str]] = None,
        probability: float = 0.5
    ) -> np.ndarray:
        """
        Apply random augmentations to a sequence.

        Args:
            sequence: Input sequence
            augmentation_types: List of augmentations to apply (None for all)
            probability: Probability of applying each augmentation

        Returns:
            Augmented sequence
        """
        if augmentation_types is None:
            augmentation_types = ['scale', 'translate', 'rotate', 'noise', 'flip', 'temporal']

        augmented = sequence.copy()

        for aug_type in augmentation_types:
            if np.random.random() < probability:
                if aug_type == 'scale':
                    augmented = self.scale(augmented)
                elif aug_type == 'translate':
                    augmented = self.translate(augmented)
                elif aug_type == 'rotate':
                    augmented = self.rotate_2d(augmented)
                elif aug_type == 'noise':
                    augmented = self.add_noise(augmented)
                elif aug_type == 'flip':
                    augmented = self.horizontal_flip(augmented)
                elif aug_type == 'temporal' and sequence.ndim == 2:
                    augmented = self.temporal_jitter(augmented)

        return augmented


def load_and_prepare_dataset(
    file_paths: List[str],
    labels: List[int],
    sequence_length: int = 20,
    augment: bool = True,
    augmentation_factor: int = 3,
    balance_classes: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load datasets from files and prepare for training.

    Args:
        file_paths: List of CSV/TXT file paths
        labels: List of labels corresponding to each file
        sequence_length: Number of timesteps per sequence
        augment: Whether to apply data augmentation
        balance_classes: Whether to balance class distribution

    Returns:
        Tuple of (X, y) arrays
    """
    X = []
    y = []

    augmenter = PoseDataAugmenter()

    for file_path, label in zip(file_paths, labels):
        logger.info(f"Loading {file_path} with label {label}")

        try:
            df = pd.read_csv(file_path)
            data = df.iloc[:, 1:].values  # Skip index column

            # Create sequences
            n_samples = len(data)
            for i in range(sequence_length, n_samples):
                sequence = data[i - sequence_length:i, :]
                X.append(sequence)
                y.append(label)

                # Apply augmentation
                if augment:
                    for _ in range(augmentation_factor - 1):
                        aug_sequence = augmenter.augment_sequence(sequence)
                        X.append(aug_sequence)
                        y.append(label)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)

    # Balance classes if requested
    if balance_classes:
        X, y = balance_dataset(X, y)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    logger.info(f"Dataset prepared: X shape = {X.shape}, y shape = {y.shape}")
    logger.info(f"Class distribution: {dict(Counter(y))}")

    return X, y


def balance_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset using SMOTE-like oversampling.

    Args:
        X: Feature array
        y: Label array

    Returns:
        Balanced (X, y)
    """
    class_counts = Counter(y)
    max_count = max(class_counts.values())

    X_balanced = list(X)
    y_balanced = list(y)

    augmenter = PoseDataAugmenter()

    for class_label, count in class_counts.items():
        if count < max_count:
            # Get samples of this class
            class_indices = np.where(y == class_label)[0]
            class_samples = X[class_indices]

            # Oversample with augmentation
            n_needed = max_count - count
            for _ in range(n_needed):
                # Pick random sample and augment
                idx = np.random.randint(len(class_samples))
                augmented = augmenter.augment_sequence(class_samples[idx])
                X_balanced.append(augmented)
                y_balanced.append(class_label)

    return np.array(X_balanced), np.array(y_balanced)


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced datasets.

    Args:
        y: Label array

    Returns:
        Dictionary mapping class index to weight
    """
    class_counts = Counter(y)
    total = len(y)
    n_classes = len(class_counts)

    weights = {}
    for class_label, count in class_counts.items():
        weights[class_label] = total / (n_classes * count)

    logger.info(f"Computed class weights: {weights}")
    return weights


def save_augmented_dataset(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    prefix: str = "augmented"
):
    """
    Save augmented dataset to files.

    Args:
        X: Feature array
        y: Label array
        output_dir: Output directory
        prefix: File prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays
    np.save(output_dir / f"{prefix}_X.npy", X)
    np.save(output_dir / f"{prefix}_y.npy", y)

    logger.info(f"Saved augmented dataset to {output_dir}")


def load_augmented_dataset(input_dir: str, prefix: str = "augmented") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load augmented dataset from files.

    Args:
        input_dir: Input directory
        prefix: File prefix

    Returns:
        Tuple of (X, y)
    """
    input_dir = Path(input_dir)

    X = np.load(input_dir / f"{prefix}_X.npy")
    y = np.load(input_dir / f"{prefix}_y.npy")

    return X, y
