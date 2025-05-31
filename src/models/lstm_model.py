"""
MIT License

Copyright (c) 2024 kunalsingh2514@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Input, Attention, MultiHeadAttention, LayerNormalization,
    Conv1D, GlobalAveragePooling1D, Concatenate, Add
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
from pathlib import Path
import time
import threading
from dataclasses import dataclass
from collections import deque
import json

from ..utils.config_manager import get_config
from ..utils.logger import get_logger
from ..utils.error_handler import ModelLoadingError, error_handler

@dataclass
class VehicleDetectionData:
    """Data structure for vehicle detection results from YOLOv8."""
    timestamp: float
    vehicle_count: int
    vehicle_types: Dict[str, int]  # e.g., {'car': 5, 'truck': 2}
    avg_confidence: float
    traffic_density: float
    frame_id: int
    intersection_id: str

@dataclass
class TrafficPredictionInput:
    """Enhanced input structure for traffic prediction."""
    detection_data: List[VehicleDetectionData]
    weather_data: Optional[Dict[str, float]] = None
    time_features: Optional[Dict[str, float]] = None
    historical_patterns: Optional[np.ndarray] = None

@dataclass
class PredictionOutput:
    """Enhanced prediction output with uncertainty quantification."""
    predicted_values: np.ndarray
    confidence_intervals: Tuple[np.ndarray, np.ndarray]  # (lower, upper)
    uncertainty_score: float
    prediction_horizon: int
    model_confidence: float
    feature_importance: Dict[str, float]

class LSTMModel:
    """
    Advanced LSTM model for traffic prediction with multiple architectures.

    Features:
    - Standard LSTM
    - Bidirectional LSTM
    - LSTM with Attention
    - Multi-step prediction
    - Transfer learning capabilities
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LSTM model.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = get_config()
        self.logger = get_logger("LSTMModel")

        # Model configuration
        self.sequence_length = self.config.get('models.lstm.sequence_length', 10)
        self.features = self.config.get('models.lstm.features',
                                      ['traffic_density', 'avg_speed', 'time_of_day'])
        self.epochs = self.config.get('models.lstm.epochs', 50)
        self.batch_size = self.config.get('models.lstm.batch_size', 32)
        self.learning_rate = self.config.get('models.lstm.learning_rate', 0.001)

        # Model components
        self.model: Optional[tf.keras.Model] = None
        self.scaler = None
        self.feature_scaler = None

        # Training history
        self.training_history = {}
        self.model_metrics = {}

    def build_standard_lstm(self, input_shape: Tuple[int, int],
                          output_size: int = 1) -> tf.keras.Model:
        """
        Build standard LSTM model.

        Args:
            input_shape: (sequence_length, n_features)
            output_size: Number of output neurons

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(32, return_sequences=False),
            Dropout(0.2),

            Dense(16, activation='relu'),
            Dense(output_size, activation='linear')
        ])

        # Use string names for metrics to avoid serialization issues
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
        )

        return model

    def build_bidirectional_lstm(self, input_shape: Tuple[int, int],
                                output_size: int = 1) -> tf.keras.Model:
        """
        Build bidirectional LSTM model.

        Args:
            input_shape: (sequence_length, n_features)
            output_size: Number of output neurons

        Returns:
            Compiled Keras model
        """
        from tensorflow.keras.layers import Bidirectional

        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),

            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.2),
            BatchNormalization(),

            Dense(16, activation='relu'),
            Dense(output_size, activation='linear')
        ])

        # Use string names for metrics to avoid serialization issues
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
        )

        return model

    def build_attention_lstm(self, input_shape: Tuple[int, int],
                           output_size: int = 1) -> tf.keras.Model:
        """
        Build LSTM model with attention mechanism.

        Args:
            input_shape: (sequence_length, n_features)
            output_size: Number of output neurons

        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)

        lstm1 = LSTM(128, return_sequences=True)(inputs)
        lstm1 = Dropout(0.2)(lstm1)
        lstm1 = BatchNormalization()(lstm1)

        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        lstm2 = BatchNormalization()(lstm2)

        attention = MultiHeadAttention(num_heads=4, key_dim=64)(lstm2, lstm2)
        attention = LayerNormalization()(attention + lstm2)

        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)

        dense1 = Dense(32, activation='relu')(pooled)
        dense1 = Dropout(0.2)(dense1)

        outputs = Dense(output_size, activation='linear')(dense1)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )

        return model

    def build_multi_step_lstm(self, input_shape: Tuple[int, int],
                            output_steps: int = 5) -> tf.keras.Model:
        """
        Build LSTM model for multi-step prediction.

        Args:
            input_shape: (sequence_length, n_features)
            output_steps: Number of future steps to predict

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(32, return_sequences=False),
            Dropout(0.2),

            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_steps, activation='linear')
        ])

        # Use string names for metrics to avoid serialization issues
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        return model

    def build_transformer_lstm(self, input_shape: Tuple[int, int],
                              output_size: int = 1) -> tf.keras.Model:
        """
        Build advanced Transformer-LSTM hybrid model for enhanced prediction.

        Args:
            input_shape: (sequence_length, n_features)
            output_size: Number of output neurons

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=input_shape)

        # Convolutional feature extraction
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.1)(conv1)

        conv2 = Conv1D(32, 3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(0.1)(conv2)

        # LSTM layers
        lstm1 = LSTM(128, return_sequences=True)(conv2)
        lstm1 = Dropout(0.2)(lstm1)
        lstm1 = BatchNormalization()(lstm1)

        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        lstm2 = BatchNormalization()(lstm2)

        # Multi-head attention
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(lstm2, lstm2)
        attention = LayerNormalization()(attention + lstm2)  # Residual connection

        # Second attention layer
        attention2 = MultiHeadAttention(num_heads=4, key_dim=32)(attention, attention)
        attention2 = LayerNormalization()(attention2 + attention)

        # Global pooling and dense layers
        pooled = GlobalAveragePooling1D()(attention2)

        # Dense layers with residual connections
        dense1 = Dense(128, activation='relu')(pooled)
        dense1 = Dropout(0.3)(dense1)
        dense1 = BatchNormalization()(dense1)

        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)

        # Residual connection
        if pooled.shape[-1] == 64:
            dense2 = Add()([pooled, dense2])

        outputs = Dense(output_size, activation='linear')(dense2)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=AdamW(learning_rate=self.learning_rate, weight_decay=1e-4),
            loss='mse',
            metrics=['mae', 'mape']
        )

        return model

    def build_uncertainty_lstm(self, input_shape: Tuple[int, int],
                              output_size: int = 1) -> tf.keras.Model:
        """
        Build LSTM model with uncertainty quantification using Monte Carlo Dropout.

        Args:
            input_shape: (sequence_length, n_features)
            output_size: Number of output neurons

        Returns:
            Compiled Keras model with uncertainty estimation
        """
        # Input layer
        inputs = Input(shape=input_shape)

        # LSTM layers with higher dropout for uncertainty
        lstm1 = LSTM(128, return_sequences=True)(inputs)
        lstm1 = Dropout(0.3, training=True)(lstm1)  # Keep dropout during inference
        lstm1 = BatchNormalization()(lstm1)

        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.3, training=True)(lstm2)
        lstm2 = BatchNormalization()(lstm2)

        lstm3 = LSTM(32, return_sequences=False)(lstm2)
        lstm3 = Dropout(0.3, training=True)(lstm3)

        # Dense layers for mean prediction
        mean_dense = Dense(32, activation='relu')(lstm3)
        mean_dense = Dropout(0.2, training=True)(mean_dense)
        mean_output = Dense(output_size, activation='linear', name='mean')(mean_dense)

        # Dense layers for variance prediction
        var_dense = Dense(32, activation='relu')(lstm3)
        var_dense = Dropout(0.2, training=True)(var_dense)
        var_output = Dense(output_size, activation='softplus', name='variance')(var_dense)

        model = Model(inputs=inputs, outputs=[mean_output, var_output])

        # Custom loss function for uncertainty
        def uncertainty_loss(y_true, y_pred):
            mean_pred, var_pred = y_pred[0], y_pred[1]
            return tf.reduce_mean(0.5 * tf.log(var_pred) + 0.5 * tf.square(y_true - mean_pred) / var_pred)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={'mean': 'mse', 'variance': 'mse'},
            loss_weights={'mean': 1.0, 'variance': 0.1},
            metrics={'mean': ['mae'], 'variance': ['mae']}
        )

        return model

    def build_multi_intersection_lstm(self, input_shape: Tuple[int, int],
                                     num_intersections: int = 4,
                                     output_size: int = 1) -> tf.keras.Model:
        """
        Build LSTM model for coordinated multi-intersection prediction.

        Args:
            input_shape: (sequence_length, n_features)
            num_intersections: Number of intersections to coordinate
            output_size: Number of output neurons per intersection

        Returns:
            Compiled Keras model for multi-intersection coordination
        """
        # Input layer
        inputs = Input(shape=input_shape)

        # Shared feature extraction
        shared_lstm1 = LSTM(128, return_sequences=True)(inputs)
        shared_lstm1 = Dropout(0.2)(shared_lstm1)
        shared_lstm1 = BatchNormalization()(shared_lstm1)

        shared_lstm2 = LSTM(64, return_sequences=True)(shared_lstm1)
        shared_lstm2 = Dropout(0.2)(shared_lstm2)
        shared_lstm2 = BatchNormalization()(shared_lstm2)

        # Intersection-specific branches
        intersection_outputs = []

        for i in range(num_intersections):
            # Intersection-specific LSTM
            int_lstm = LSTM(32, return_sequences=False, name=f'intersection_{i}_lstm')(shared_lstm2)
            int_lstm = Dropout(0.2)(int_lstm)

            # Intersection-specific dense layers
            int_dense = Dense(16, activation='relu', name=f'intersection_{i}_dense')(int_lstm)
            int_output = Dense(output_size, activation='linear', name=f'intersection_{i}_output')(int_dense)

            intersection_outputs.append(int_output)

        # Coordination layer - learns inter-intersection dependencies
        if len(intersection_outputs) > 1:
            coord_input = Concatenate()(intersection_outputs)
            coord_dense = Dense(64, activation='relu', name='coordination_layer')(coord_input)
            coord_dense = Dropout(0.2)(coord_dense)

            # Final coordinated outputs
            final_outputs = []
            for i in range(num_intersections):
                coord_output = Dense(output_size, activation='linear',
                                   name=f'coordinated_intersection_{i}')(coord_dense)
                final_outputs.append(coord_output)

            outputs = final_outputs
        else:
            outputs = intersection_outputs

        model = Model(inputs=inputs, outputs=outputs)

        # Multi-output loss
        loss_dict = {f'coordinated_intersection_{i}': 'mse' for i in range(num_intersections)}
        metrics_dict = {f'coordinated_intersection_{i}': ['mae'] for i in range(num_intersections)}

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss_dict,
            metrics=metrics_dict
        )

        return model

    @error_handler(reraise=True)
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              model_type: str = 'standard') -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_type: Type of model ('standard', 'bidirectional', 'attention', 'multi_step')

        Returns:
            Training history dictionary
        """
        try:
            self.logger.info(f"Training {model_type} LSTM model...")

            # Determine input shape and output size
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1

            # Build model based on type
            if model_type == 'standard':
                self.model = self.build_standard_lstm(input_shape, output_size)
            elif model_type == 'bidirectional':
                self.model = self.build_bidirectional_lstm(input_shape, output_size)
            elif model_type == 'attention':
                self.model = self.build_attention_lstm(input_shape, output_size)
            elif model_type == 'multi_step':
                self.model = self.build_multi_step_lstm(input_shape, output_size)
            elif model_type == 'transformer':
                self.model = self.build_transformer_lstm(input_shape, output_size)
            elif model_type == 'uncertainty':
                self.model = self.build_uncertainty_lstm(input_shape, output_size)
            elif model_type == 'multi_intersection':
                num_intersections = self.config.get('models.lstm.num_intersections', 4)
                self.model = self.build_multi_intersection_lstm(input_shape, num_intersections, output_size)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)

            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                )
            ]

            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )

            # Store training history
            self.training_history = history.history

            # Calculate final metrics
            train_loss = self.model.evaluate(X_train, y_train, verbose=0)
            self.model_metrics['train_loss'] = train_loss[0]
            self.model_metrics['train_mae'] = train_loss[1]

            if validation_data:
                val_loss = self.model.evaluate(X_val, y_val, verbose=0)
                self.model_metrics['val_loss'] = val_loss[0]
                self.model_metrics['val_mae'] = val_loss[1]

            self.logger.info(f"Model training completed. Final train loss: {train_loss[0]:.4f}")

            return self.training_history

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise ModelLoadingError(f"Training failed: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Input features

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ModelLoadingError("Model not trained or loaded")

        try:
            predictions = self.model.predict(X, verbose=0)
            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise ModelLoadingError(f"Prediction failed: {e}")

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and associated data.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ModelLoadingError("No model to save")

        try:
            # Save model
            self.model.save(filepath)

            # Save additional data
            model_data = {
                'training_history': self.training_history,
                'model_metrics': self.model_metrics,
                'config': {
                    'sequence_length': self.sequence_length,
                    'features': self.features,
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate
                }
            }

            # Save metadata
            metadata_path = filepath.replace('.h5', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"Model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise ModelLoadingError(f"Model saving failed: {e}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.

        Args:
            filepath: Path to the saved model
        """
        try:
            # Custom objects to handle keras serialization issues
            custom_objects = {
                'mse': 'mean_squared_error',
                'mae': 'mean_absolute_error',
                'mape': 'mean_absolute_percentage_error'
            }

            # Load model with custom objects
            self.model = tf.keras.models.load_model(filepath, custom_objects=custom_objects, compile=False)

            # Recompile the model with proper metrics
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mean_squared_error',
                metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
            )

            # Load metadata if available
            metadata_path = filepath.replace('.h5', '_metadata.pkl')
            if Path(metadata_path).exists():
                with open(metadata_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.training_history = model_data.get('training_history', {})
                self.model_metrics = model_data.get('model_metrics', {})

                # Update config if available
                config = model_data.get('config', {})
                for key, value in config.items():
                    setattr(self, key, value)

            self.logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Don't raise error, just log it and continue in simulation mode
            self.model = None

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "No model available"

        import io
        import sys

        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            self.model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout

        return summary

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics and history."""
        return {
            'training_history': self.training_history,
            'model_metrics': self.model_metrics,
            'model_summary': self.get_model_summary()
        }


class RealTimeLSTMPredictor:
    """
    Enhanced LSTM predictor for real-time traffic prediction with YOLOv8 integration.

    Features:
    - Real-time data processing from YOLOv8 vehicle detection
    - Online learning and model adaptation
    - Multi-step prediction with uncertainty quantification
    - Performance optimization for real-time inference
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the real-time LSTM predictor."""
        self.config = get_config()
        self.logger = get_logger("RealTimeLSTMPredictor")

        # Configuration
        self.sequence_length = self.config.get('models.lstm.sequence_length', 10)
        self.prediction_horizon = self.config.get('models.lstm.prediction_horizon', 5)
        self.update_frequency = self.config.get('models.lstm.update_frequency', 100)  # frames
        self.buffer_size = self.config.get('models.lstm.buffer_size', 1000)

        # Model components
        self.lstm_model = LSTMModel(config_path)
        self.model_type = self.config.get('models.lstm.real_time_type', 'transformer')

        # Data buffers for real-time processing
        self.detection_buffer = deque(maxlen=self.buffer_size)
        self.prediction_buffer = deque(maxlen=self.buffer_size)

        # Feature processing
        self.feature_scaler = None
        self.target_scaler = None

        # Performance tracking
        self.frame_count = 0
        self.prediction_count = 0
        self.total_inference_time = 0.0

        # Threading for real-time processing
        self._lock = threading.Lock()
        self._is_training = False

        self.logger.info("RealTimeLSTMPredictor initialized")

    def process_detection_data(self, detection_data: VehicleDetectionData) -> np.ndarray:
        """
        Process YOLOv8 detection data into features for LSTM prediction.

        Args:
            detection_data: Vehicle detection results from YOLOv8

        Returns:
            Feature vector for LSTM input
        """
        # Extract time-based features
        timestamp = detection_data.timestamp
        hour = time.localtime(timestamp).tm_hour
        day_of_week = time.localtime(timestamp).tm_wday

        # Basic traffic features
        features = [
            detection_data.vehicle_count,
            detection_data.traffic_density,
            detection_data.avg_confidence,
            hour / 24.0,  # Normalized hour
            day_of_week / 7.0,  # Normalized day of week
        ]

        # Vehicle type distribution
        total_vehicles = max(detection_data.vehicle_count, 1)
        for vehicle_type in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
            count = detection_data.vehicle_types.get(vehicle_type, 0)
            features.append(count / total_vehicles)

        # Additional derived features
        features.extend([
            np.sin(2 * np.pi * hour / 24),  # Cyclical hour encoding
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),  # Cyclical day encoding
            np.cos(2 * np.pi * day_of_week / 7),
        ])

        return np.array(features, dtype=np.float32)

    def add_detection_data(self, detection_data: VehicleDetectionData) -> None:
        """
        Add new detection data to the processing buffer.

        Args:
            detection_data: New vehicle detection data
        """
        with self._lock:
            # Process detection data into features
            features = self.process_detection_data(detection_data)

            # Add to buffer with timestamp
            self.detection_buffer.append({
                'timestamp': detection_data.timestamp,
                'features': features,
                'vehicle_count': detection_data.vehicle_count,
                'intersection_id': detection_data.intersection_id
            })

            self.frame_count += 1

            # Trigger prediction if we have enough data
            if len(self.detection_buffer) >= self.sequence_length:
                self._trigger_prediction()

    def _trigger_prediction(self) -> None:
        """Trigger prediction based on current buffer state."""
        if self.lstm_model.model is None:
            return

        try:
            # Prepare sequence data
            sequence_data = list(self.detection_buffer)[-self.sequence_length:]
            X = np.array([item['features'] for item in sequence_data])
            X = X.reshape(1, self.sequence_length, -1)

            # Scale features if scaler is available
            if self.feature_scaler is not None:
                X_scaled = self.feature_scaler.transform(X.reshape(-1, X.shape[-1]))
                X = X_scaled.reshape(X.shape)

            # Make prediction
            start_time = time.time()

            if self.model_type == 'uncertainty':
                # Handle uncertainty model output
                predictions = self.lstm_model.predict(X)
                if isinstance(predictions, list) and len(predictions) == 2:
                    mean_pred, var_pred = predictions
                    prediction_output = PredictionOutput(
                        predicted_values=mean_pred[0],
                        confidence_intervals=(
                            mean_pred[0] - 1.96 * np.sqrt(var_pred[0]),
                            mean_pred[0] + 1.96 * np.sqrt(var_pred[0])
                        ),
                        uncertainty_score=float(np.mean(var_pred[0])),
                        prediction_horizon=self.prediction_horizon,
                        model_confidence=1.0 / (1.0 + float(np.mean(var_pred[0]))),
                        feature_importance={}
                    )
                else:
                    # Fallback for standard prediction
                    prediction_output = self._create_standard_prediction_output(predictions[0])
            else:
                # Standard prediction
                predictions = self.lstm_model.predict(X)
                prediction_output = self._create_standard_prediction_output(predictions[0])

            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.prediction_count += 1

            # Store prediction
            self.prediction_buffer.append({
                'timestamp': time.time(),
                'prediction': prediction_output,
                'inference_time': inference_time,
                'input_sequence': sequence_data[-1]['intersection_id']
            })

            # Log prediction
            self.logger.debug(f"Prediction made: {prediction_output.predicted_values} "
                            f"(confidence: {prediction_output.model_confidence:.3f}, "
                            f"inference: {inference_time:.3f}s)")

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")

    def _create_standard_prediction_output(self, predictions: np.ndarray) -> PredictionOutput:
        """Create standard prediction output for non-uncertainty models."""
        return PredictionOutput(
            predicted_values=predictions,
            confidence_intervals=(predictions * 0.9, predictions * 1.1),  # Simple ±10%
            uncertainty_score=0.1,  # Default uncertainty
            prediction_horizon=self.prediction_horizon,
            model_confidence=0.8,  # Default confidence
            feature_importance={}
        )

    def get_latest_prediction(self) -> Optional[PredictionOutput]:
        """Get the most recent prediction."""
        with self._lock:
            if self.prediction_buffer:
                return self.prediction_buffer[-1]['prediction']
            return None

    def get_prediction_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent prediction history."""
        with self._lock:
            return list(self.prediction_buffer)[-n:]

    def start_online_training(self, initial_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        """
        Start online training mode for continuous model improvement.

        Args:
            initial_data: Optional initial training data (X, y)
        """
        if initial_data is not None:
            X_init, y_init = initial_data
            self.logger.info("Training initial model...")

            # Train initial model
            self.lstm_model.train(X_init, y_init, model_type=self.model_type)

            # Fit scalers
            from sklearn.preprocessing import StandardScaler
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()

            self.feature_scaler.fit(X_init.reshape(-1, X_init.shape[-1]))
            self.target_scaler.fit(y_init.reshape(-1, 1))

            self.logger.info("Initial model training completed")

        self._is_training = True
        self.logger.info("Online training mode activated")

    def update_model(self, force_update: bool = False) -> bool:
        """
        Update model with recent data if conditions are met.

        Args:
            force_update: Force model update regardless of conditions

        Returns:
            True if model was updated, False otherwise
        """
        if not self._is_training:
            return False

        if not force_update and self.frame_count % self.update_frequency != 0:
            return False

        if len(self.detection_buffer) < self.sequence_length * 2:
            return False

        try:
            self.logger.info("Updating model with recent data...")

            # Prepare training data from buffer
            buffer_data = list(self.detection_buffer)

            X_sequences = []
            y_sequences = []

            for i in range(len(buffer_data) - self.sequence_length):
                # Input sequence
                X_seq = [item['features'] for item in buffer_data[i:i+self.sequence_length]]
                # Target (next vehicle count)
                y_target = buffer_data[i+self.sequence_length]['vehicle_count']

                X_sequences.append(X_seq)
                y_sequences.append(y_target)

            if len(X_sequences) < 10:  # Need minimum data for training
                return False

            X_update = np.array(X_sequences)
            y_update = np.array(y_sequences)

            # Scale data
            if self.feature_scaler is not None:
                X_scaled = self.feature_scaler.transform(X_update.reshape(-1, X_update.shape[-1]))
                X_update = X_scaled.reshape(X_update.shape)

            if self.target_scaler is not None:
                y_update = self.target_scaler.transform(y_update.reshape(-1, 1)).flatten()

            # Perform incremental training (few epochs)
            if self.lstm_model.model is not None:
                self.lstm_model.model.fit(
                    X_update, y_update,
                    epochs=3,  # Few epochs for online learning
                    batch_size=min(16, len(X_update)),
                    verbose=0
                )

                self.logger.info(f"Model updated with {len(X_update)} samples")
                return True

        except Exception as e:
            self.logger.error(f"Model update failed: {e}")

        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_inference_time = (self.total_inference_time / self.prediction_count
                            if self.prediction_count > 0 else 0.0)

        return {
            'frames_processed': self.frame_count,
            'predictions_made': self.prediction_count,
            'average_inference_time': avg_inference_time,
            'buffer_size': len(self.detection_buffer),
            'prediction_buffer_size': len(self.prediction_buffer),
            'is_training': self._is_training,
            'model_type': self.model_type
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        with self._lock:
            self.detection_buffer.clear()
            self.prediction_buffer.clear()
            self.lstm_model = None

        self.logger.info("RealTimeLSTMPredictor cleaned up")
