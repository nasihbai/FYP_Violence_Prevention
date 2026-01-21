"""
Enhanced Model Training Script
==============================
Trains an improved LSTM model with:
- Data augmentation
- Class balancing
- Attention mechanism
- Comprehensive evaluation

Usage:
    python train_model_enhanced.py --data-dir ./data --output-dir ./models
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.lstm_model import (
    create_enhanced_lstm_model,
    create_simple_lstm_model,
    get_training_callbacks
)
from utils.data_augmentation import (
    load_and_prepare_dataset,
    compute_class_weights,
    save_augmented_dataset
)
from utils.evaluation import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Violence Detection Model')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=20, help='Sequence length')
    parser.add_argument('--augment', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--augmentation-factor', type=int, default=3, help='Augmentation multiplier')
    parser.add_argument('--simple-model', action='store_true', help='Use simple LSTM instead of enhanced')
    parser.add_argument('--no-attention', action='store_true', help='Disable attention mechanism')
    parser.add_argument('--no-yolo', action='store_true', help='Train for single-person detection')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define dataset files and labels
    # Adjust these paths based on your actual data files
    dataset_files = []
    labels = []

    # Check for violence detection datasets
    if (data_dir / 'neutral.txt').exists():
        dataset_files.append(str(data_dir / 'neutral.txt'))
        labels.append(0)
        logger.info("Found neutral.txt")

    if (data_dir / 'violent.txt').exists():
        dataset_files.append(str(data_dir / 'violent.txt'))
        labels.append(1)
        logger.info("Found violent.txt")

    # Alternative: hand gesture datasets
    if not dataset_files:
        gesture_files = ['resting.txt', 'holding.txt', 'gripping.txt', 'grasping.txt']
        for i, filename in enumerate(gesture_files):
            if (data_dir / filename).exists():
                dataset_files.append(str(data_dir / filename))
                labels.append(i)
                logger.info(f"Found {filename}")

    if not dataset_files:
        logger.error("No dataset files found! Please ensure data files exist in the data directory.")
        logger.info("Expected files: neutral.txt, violent.txt (for violence detection)")
        logger.info("Or: resting.txt, holding.txt, gripping.txt, grasping.txt (for hand gestures)")
        sys.exit(1)

    logger.info(f"Loading {len(dataset_files)} dataset files...")

    # Load and prepare dataset
    X, y = load_and_prepare_dataset(
        file_paths=dataset_files,
        labels=labels,
        sequence_length=args.sequence_length,
        augment=args.augment,
        augmentation_factor=args.augmentation_factor,
        balance_classes=True
    )

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Compute class weights
    class_weights = compute_class_weights(y_train)

    # Determine number of classes
    num_classes = len(np.unique(y))
    num_features = X.shape[2]

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Number of features: {num_features}")

    # Create model
    if args.simple_model:
        logger.info("Creating simple LSTM model...")
        model = create_simple_lstm_model(
            sequence_length=args.sequence_length,
            num_features=num_features,
            num_classes=num_classes
        )
    else:
        logger.info("Creating enhanced LSTM model with attention...")
        model = create_enhanced_lstm_model(
            sequence_length=args.sequence_length,
            num_features=num_features,
            num_classes=num_classes,
            lstm_units=64,
            dropout_rate=0.3,
            use_attention=not args.no_attention,
            use_bidirectional=True,
            use_tcn=True
        )

    model.summary()

    # Setup callbacks
    callbacks = get_training_callbacks(
        checkpoint_dir=str(output_dir / 'checkpoints'),
        early_stopping_patience=15,
        reduce_lr_patience=5
    )

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model_filename = 'violence_lstm_enhanced.h5' if not args.simple_model else 'violence_lstm_simple.h5'
    model_path = output_dir / model_filename
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Evaluate model
    logger.info("Evaluating model...")
    class_names = ['neutral', 'violent'] if num_classes == 2 else [f'class_{i}' for i in range(num_classes)]
    evaluate_model(
        model,
        X_test,
        y_test,
        class_names=class_names,
        output_dir=str(output_dir / 'evaluation')
    )

    # Save training history
    history_path = output_dir / 'training_history.npy'
    np.save(str(history_path), history.history)
    logger.info(f"Training history saved to {history_path}")

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
