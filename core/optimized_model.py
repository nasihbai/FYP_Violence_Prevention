"""
Optimized Violence Detection Models
====================================
High-performance model architectures for real-time violence detection:
- Temporal Convolutional Networks (TCN) for parallel processing
- Lightweight GRU/LSTM variants
- Hybrid CNN-RNN architectures
- TFLite-optimized models
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CausalConv1D(layers.Layer):
    """Causal 1D convolution for temporal sequences."""

    def __init__(self, filters: int, kernel_size: int, dilation_rate: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = (kernel_size - 1) * dilation_rate

    def build(self, input_shape):
        self.conv = layers.Conv1D(
            self.filters,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal',
            kernel_initializer='he_normal'
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.conv(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })
        return config


class TCNBlock(layers.Layer):
    """
    Temporal Convolutional Network block with residual connections.
    Much faster than LSTM due to parallel processing.
    """

    def __init__(self, filters: int, kernel_size: int = 3, dilation_rate: int = 1,
                 dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(
            self.filters, self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal',
            activation='relu',
            kernel_initializer='he_normal'
        )
        self.conv2 = layers.Conv1D(
            self.filters, self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal',
            activation='relu',
            kernel_initializer='he_normal'
        )
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        # Residual projection if dimensions don't match
        if input_shape[-1] != self.filters:
            self.residual_proj = layers.Conv1D(self.filters, 1, padding='same')
        else:
            self.residual_proj = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout2(x, training=training)

        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(inputs)
        else:
            residual = inputs

        return layers.ReLU()(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config


def create_tcn_model(
    sequence_length: int = 15,
    num_features: int = 132,
    num_classes: int = 1,
    filters: int = 64,
    kernel_size: int = 3,
    num_blocks: int = 4,
    dropout_rate: float = 0.2
) -> Model:
    """
    Create a pure TCN model - fastest option for real-time inference.

    Advantages:
    - Parallel processing (no sequential bottleneck)
    - Dilated convolutions capture long-range dependencies
    - Much faster inference than LSTM

    Args:
        sequence_length: Number of frames (reduced from 20 to 15)
        num_features: Features per frame
        num_classes: Output classes (1 for binary sigmoid)
        filters: Number of filters per TCN block
        kernel_size: Convolution kernel size
        num_blocks: Number of TCN blocks
        dropout_rate: Dropout rate
    """
    inputs = layers.Input(shape=(sequence_length, num_features), name='input')

    x = inputs

    # Stack TCN blocks with increasing dilation
    for i in range(num_blocks):
        dilation = 2 ** i  # 1, 2, 4, 8
        x = TCNBlock(filters, kernel_size, dilation, dropout_rate, name=f'tcn_block_{i}')(x)

    # Global pooling to aggregate temporal information
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='tcn_violence_detector')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_hybrid_cnn_gru_model(
    sequence_length: int = 15,
    num_features: int = 132,
    num_classes: int = 1,
    cnn_filters: int = 64,
    gru_units: int = 32,
    dropout_rate: float = 0.2
) -> Model:
    """
    Hybrid CNN-GRU model - good balance of speed and accuracy.

    1D CNN extracts local motion patterns
    GRU captures temporal dependencies (faster than LSTM)
    """
    inputs = layers.Input(shape=(sequence_length, num_features), name='input')

    # CNN feature extraction (parallel processing)
    x = layers.Conv1D(cnn_filters, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(cnn_filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # GRU for temporal modeling (faster than LSTM)
    x = layers.GRU(gru_units, return_sequences=False)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Classification
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='hybrid_cnn_gru')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_lightweight_lstm_model(
    sequence_length: int = 15,
    num_features: int = 132,
    num_classes: int = 1,
    lstm_units: int = 32,
    dropout_rate: float = 0.2
) -> Model:
    """
    Lightweight LSTM - minimal architecture for fastest LSTM inference.
    Reduced from 3 layers to 2, smaller hidden size.
    """
    inputs = layers.Input(shape=(sequence_length, num_features), name='input')

    # Single LSTM layer with fewer units
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.Dropout(dropout_rate)(x)

    # Final LSTM
    x = layers.LSTM(lstm_units // 2, return_sequences=False)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Minimal classification head
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='lightweight_lstm')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_ultra_fast_model(
    sequence_length: int = 10,
    num_features: int = 132,
    num_classes: int = 1
) -> Model:
    """
    Ultra-fast model for edge deployment.
    Pure 1D CNN - no recurrent layers at all.
    Designed for <10ms inference.
    """
    inputs = layers.Input(shape=(sequence_length, num_features), name='input')

    # Depthwise separable convolutions for efficiency
    x = layers.SeparableConv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.SeparableConv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='ultra_fast')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def convert_to_tflite(
    model: Model,
    output_path: str,
    quantize: bool = True,
    representative_data: Optional[np.ndarray] = None
) -> str:
    """
    Convert Keras model to TensorFlow Lite for faster inference.

    Args:
        model: Trained Keras model
        output_path: Path to save .tflite file
        quantize: Whether to apply INT8 quantization
        representative_data: Sample data for calibration (required for quantization)

    Returns:
        Path to saved TFLite model
    """
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Allow custom ops and select TF ops for compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False

        if quantize:
            # Dynamic range quantization (works without calibration data)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        logger.info(f"TFLite model saved to {output_path}")
        logger.info(f"Model size: {len(tflite_model) / 1024:.2f} KB")

        return output_path

    except Exception as e:
        logger.warning(f"TFLite conversion failed: {e}")
        logger.warning("Skipping TFLite - use Keras model for inference")
        return None


class TFLiteInference:
    """
    Fast TFLite inference wrapper.
    Significantly faster than Keras model.predict()
    """

    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Run inference on a single sequence.

        Args:
            sequence: Input array of shape (sequence_length, num_features)
                      or (batch, sequence_length, num_features)

        Returns:
            Prediction array
        """
        # Ensure correct shape
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        # Convert to correct dtype
        sequence = sequence.astype(self.input_dtype)

        self.interpreter.set_tensor(self.input_details[0]['index'], sequence)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Predict on batch (processes sequentially for TFLite)."""
        results = []
        for seq in sequences:
            results.append(self.predict(seq))
        return np.array(results).squeeze()


# Model registry for easy access
MODEL_REGISTRY = {
    'tcn': create_tcn_model,
    'hybrid_cnn_gru': create_hybrid_cnn_gru_model,
    'lightweight_lstm': create_lightweight_lstm_model,
    'ultra_fast': create_ultra_fast_model,
}


def create_model(model_type: str = 'tcn', **kwargs) -> Model:
    """
    Factory function to create models by name.

    Args:
        model_type: One of 'tcn', 'hybrid_cnn_gru', 'lightweight_lstm', 'ultra_fast'
        **kwargs: Model-specific parameters

    Returns:
        Compiled Keras model
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_type](**kwargs)


def benchmark_models(X_sample: np.ndarray, iterations: int = 100):
    """
    Benchmark all model architectures for inference speed.

    Args:
        X_sample: Sample input data of shape (n, sequence_length, features)
        iterations: Number of inference iterations for timing

    Returns:
        Dict of model timings
    """
    import time

    results = {}
    sequence_length = X_sample.shape[1]
    num_features = X_sample.shape[2]

    for name, create_fn in MODEL_REGISTRY.items():
        try:
            model = create_fn(sequence_length=sequence_length, num_features=num_features)

            # Warmup
            _ = model.predict(X_sample[:1], verbose=0)

            # Time single inference
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = model.predict(X_sample[:1], verbose=0)
                times.append((time.perf_counter() - start) * 1000)

            results[name] = {
                'avg_ms': np.mean(times),
                'std_ms': np.std(times),
                'params': model.count_params(),
                'fps': 1000 / np.mean(times)
            }

            # Cleanup
            del model
            tf.keras.backend.clear_session()

        except Exception as e:
            results[name] = {'error': str(e)}

    return results
