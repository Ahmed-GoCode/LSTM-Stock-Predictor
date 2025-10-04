"""
LSTM Model implementation for stock price prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.models import Sequential, load_model
from typing import Tuple, List, Dict, Optional, Any
import logging
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import joblib

from ..config.config import config
from ..utils.exceptions import ModelError, TrainingError, PredictionError

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Data structure for model performance metrics"""
    train_loss: float
    val_loss: float
    train_mae: float
    val_mae: float
    train_rmse: float
    val_rmse: float
    train_mape: float
    val_mape: float
    directional_accuracy: float
    training_time: float
    epochs_trained: int

@dataclass
class TrainingHistory:
    """Training history and metadata"""
    history: Dict[str, List[float]]
    metrics: ModelMetrics
    config: Dict[str, Any]
    timestamp: str
    model_version: str

class LSTMModel:
    """
    Advanced LSTM model for stock price prediction
    """
    
    def __init__(self, 
                 lstm_units: List[int] = None,
                 dropout_rate: float = None,
                 learning_rate: float = None,
                 sequence_length: int = None,
                 n_features: int = None):
        """
        Initialize LSTM model
        
        Args:
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            sequence_length: Input sequence length
            n_features: Number of input features
        """
        self.lstm_units = lstm_units or config.model.lstm_units
        self.dropout_rate = dropout_rate or config.model.dropout_rate
        self.learning_rate = learning_rate or config.model.learning_rate
        self.sequence_length = sequence_length or config.model.sequence_length
        self.n_features = n_features
        
        self.model = None
        self.is_trained = False
        self.training_history = None
        self.scaler = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def build_model(self, n_features: int) -> None:
        """
        Build the LSTM model architecture
        
        Args:
            n_features: Number of input features
        """
        try:
            logger.info(f"Building LSTM model with {self.lstm_units} units")
            
            self.n_features = n_features
            self.model = Sequential()
            
            # Input layer
            self.model.add(layers.InputLayer(
                input_shape=(self.sequence_length, n_features)
            ))
            
            # LSTM layers
            for i, units in enumerate(self.lstm_units):
                return_sequences = i < len(self.lstm_units) - 1  # Return sequences for all but last layer
                
                self.model.add(layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate,
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    name=f'lstm_{i+1}'
                ))
                
                # Add BatchNormalization after each LSTM layer
                if i < len(self.lstm_units) - 1:
                    self.model.add(layers.BatchNormalization())
            
            # Dense layers
            self.model.add(layers.Dense(50, activation='relu'))
            self.model.add(layers.Dropout(self.dropout_rate))
            self.model.add(layers.Dense(25, activation='relu'))
            self.model.add(layers.Dropout(self.dropout_rate / 2))
            
            # Output layer
            self.model.add(layers.Dense(1, activation='linear'))
            
            # Compile the model
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
            
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', self._rmse_metric, self._mape_metric]
            )
            
            logger.info(f"Model built successfully. Total parameters: {self.model.count_params():,}")
            self._print_model_summary()
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise ModelError(f"Failed to build LSTM model: {e}")
    
    def _rmse_metric(self, y_true, y_pred):
        """Custom RMSE metric"""
        return tf.keras.metrics.RootMeanSquaredError()(y_true, y_pred)
    
    def _mape_metric(self, y_true, y_pred):
        """Custom MAPE metric"""
        return tf.keras.metrics.MeanAbsolutePercentageError()(y_true, y_pred)
    
    def _print_model_summary(self):
        """Print model architecture summary"""
        if self.model:
            logger.info("Model Architecture:")
            self.model.summary(print_fn=logger.info)
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              epochs: int = None,
              batch_size: int = None,
              validation_split: float = None,
              callbacks_list: List = None,
              verbose: int = 1) -> TrainingHistory:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation split ratio
            callbacks_list: List of Keras callbacks
            verbose: Verbosity level
            
        Returns:
            TrainingHistory object
        """
        try:
            if self.model is None:
                self.build_model(X_train.shape[2])
            
            # Set training parameters
            epochs = epochs or config.model.epochs
            batch_size = batch_size or config.model.batch_size
            validation_split = validation_split or config.model.validation_split
            
            logger.info(f"Starting training: epochs={epochs}, batch_size={batch_size}")
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
                validation_split = None  # Don't use validation_split if validation data is provided
            
            # Setup callbacks
            if callbacks_list is None:
                callbacks_list = self._get_default_callbacks()
            
            # Train the model
            start_time = datetime.now()
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                validation_split=validation_split,
                callbacks=callbacks_list,
                verbose=verbose,
                shuffle=False  # Don't shuffle time series data
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            metrics = self._calculate_metrics(history, training_time)
            
            # Create training history object
            self.training_history = TrainingHistory(
                history=history.history,
                metrics=metrics,
                config=self._get_model_config(),
                timestamp=datetime.now().isoformat(),
                model_version="1.0"
            )
            
            self.is_trained = True
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise TrainingError(f"Model training failed: {e}")
    
    def _get_default_callbacks(self) -> List[callbacks.Callback]:
        """Get default training callbacks"""
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callback_list
    
    def _calculate_metrics(self, history, training_time: float) -> ModelMetrics:
        """Calculate comprehensive training metrics"""
        try:
            # Get final metrics from history
            train_loss = history.history['loss'][-1]
            val_loss = history.history.get('val_loss', [0])[-1]
            train_mae = history.history['mae'][-1]
            val_mae = history.history.get('val_mae', [0])[-1]
            
            # RMSE and MAPE
            train_rmse = history.history.get('_rmse_metric', [0])[-1]
            val_rmse = history.history.get('val__rmse_metric', [0])[-1]
            train_mape = history.history.get('_mape_metric', [0])[-1]
            val_mape = history.history.get('val__mape_metric', [0])[-1]
            
            # Placeholder for directional accuracy (would need predictions to calculate)
            directional_accuracy = 0.0
            
            return ModelMetrics(
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                train_mae=float(train_mae),
                val_mae=float(val_mae),
                train_rmse=float(train_rmse),
                val_rmse=float(val_rmse),
                train_mape=float(train_mape),
                val_mape=float(val_mape),
                directional_accuracy=directional_accuracy,
                training_time=training_time,
                epochs_trained=len(history.history['loss'])
            )
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return ModelMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, training_time, 0)
    
    def predict(self, X: np.ndarray, batch_size: int = None) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features
            batch_size: Prediction batch size
            
        Returns:
            Predictions array
        """
        try:
            if not self.is_trained:
                raise PredictionError("Model must be trained before making predictions")
            
            batch_size = batch_size or config.model.batch_size
            
            logger.info(f"Making predictions for {len(X)} samples")
            
            predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise PredictionError(f"Prediction failed: {e}")
    
    def predict_future(self, 
                      last_sequence: np.ndarray,
                      n_steps: int,
                      confidence_intervals: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict future values using recursive prediction
        
        Args:
            last_sequence: Last sequence of historical data
            n_steps: Number of future steps to predict
            confidence_intervals: Whether to calculate confidence intervals
            
        Returns:
            Dictionary with predictions and optionally confidence intervals
        """
        try:
            if not self.is_trained:
                raise PredictionError("Model must be trained before making predictions")
            
            logger.info(f"Predicting {n_steps} future steps")
            
            # Ensure input has correct shape
            if len(last_sequence.shape) == 2:
                last_sequence = last_sequence.reshape(1, *last_sequence.shape)
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            # Generate predictions recursively
            for step in range(n_steps):
                # Predict next value
                next_pred = self.model.predict(current_sequence, verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Update sequence for next prediction
                # Remove first timestep and add prediction as last timestep
                next_features = np.zeros((1, 1, current_sequence.shape[2]))
                next_features[0, 0, 0] = next_pred[0, 0]  # Use prediction as first feature
                
                # For other features, use the last values (simplified approach)
                if current_sequence.shape[2] > 1:
                    next_features[0, 0, 1:] = current_sequence[0, -1, 1:]
                
                # Update current sequence
                current_sequence = np.concatenate([
                    current_sequence[:, 1:, :],
                    next_features
                ], axis=1)
            
            result = {"predictions": np.array(predictions)}
            
            # Calculate confidence intervals using Monte Carlo dropout
            if confidence_intervals:
                result.update(self._calculate_confidence_intervals(last_sequence, n_steps))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in future prediction: {e}")
            raise PredictionError(f"Future prediction failed: {e}")
    
    def _calculate_confidence_intervals(self, 
                                      last_sequence: np.ndarray,
                                      n_steps: int,
                                      n_simulations: int = 100) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals using Monte Carlo dropout"""
        try:
            # Enable dropout during inference
            predictions_list = []
            
            for _ in range(n_simulations):
                # Create a model with dropout enabled during inference
                mc_model = self._create_mc_model()
                
                current_sequence = last_sequence.copy()
                simulation_predictions = []
                
                for step in range(n_steps):
                    next_pred = mc_model(current_sequence, training=True)
                    simulation_predictions.append(next_pred[0, 0].numpy())
                    
                    # Update sequence (simplified)
                    next_features = np.zeros((1, 1, current_sequence.shape[2]))
                    next_features[0, 0, 0] = next_pred[0, 0]
                    
                    if current_sequence.shape[2] > 1:
                        next_features[0, 0, 1:] = current_sequence[0, -1, 1:]
                    
                    current_sequence = np.concatenate([
                        current_sequence[:, 1:, :],
                        next_features
                    ], axis=1)
                
                predictions_list.append(simulation_predictions)
            
            # Calculate confidence intervals
            predictions_array = np.array(predictions_list)
            lower_bound = np.percentile(predictions_array, 2.5, axis=0)
            upper_bound = np.percentile(predictions_array, 97.5, axis=0)
            
            return {
                "lower_confidence": lower_bound,
                "upper_confidence": upper_bound,
                "prediction_std": np.std(predictions_array, axis=0)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating confidence intervals: {e}")
            return {}
    
    def _create_mc_model(self):
        """Create a model with Monte Carlo dropout enabled"""
        # This would create a copy of the model with dropout always enabled
        # For simplicity, returning the original model
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ModelError("Model must be trained before evaluation")
            
            logger.info("Evaluating model performance")
            
            # Get predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - predictions))
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            # Directional accuracy
            actual_direction = np.sign(np.diff(y_test))
            predicted_direction = np.sign(np.diff(predictions))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            # R-squared
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            
            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "directional_accuracy": float(directional_accuracy),
                "r2_score": float(r2_score)
            }
            
            logger.info(f"Evaluation completed: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise ModelError(f"Model evaluation failed: {e}")
    
    def save_model(self, filepath: str, save_history: bool = True) -> None:
        """
        Save the trained model and metadata
        
        Args:
            filepath: Path to save the model
            save_history: Whether to save training history
        """
        try:
            if self.model is None:
                raise ModelError("No model to save")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the Keras model
            model_path = filepath.replace('.pkl', '.h5')
            self.model.save(model_path)
            
            # Save additional metadata
            metadata = {
                "lstm_units": self.lstm_units,
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "sequence_length": self.sequence_length,
                "n_features": self.n_features,
                "is_trained": self.is_trained,
                "model_config": self._get_model_config()
            }
            
            # Save training history if available
            if save_history and self.training_history:
                metadata["training_history"] = asdict(self.training_history)
            
            # Save metadata as JSON
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise ModelError(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str, load_history: bool = True) -> None:
        """
        Load a saved model and metadata
        
        Args:
            filepath: Path to the saved model
            load_history: Whether to load training history
        """
        try:
            # Load the Keras model
            model_path = filepath.replace('.pkl', '.h5')
            self.model = load_model(model_path)
            
            # Load metadata
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Restore model parameters
            self.lstm_units = metadata["lstm_units"]
            self.dropout_rate = metadata["dropout_rate"]
            self.learning_rate = metadata["learning_rate"]
            self.sequence_length = metadata["sequence_length"]
            self.n_features = metadata["n_features"]
            self.is_trained = metadata["is_trained"]
            
            # Load training history if available
            if load_history and "training_history" in metadata:
                history_data = metadata["training_history"]
                self.training_history = TrainingHistory(**history_data)
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelError(f"Failed to load model: {e}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "model_type": "LSTM",
            "framework": "TensorFlow/Keras"
        }
    
    def get_model_summary(self) -> str:
        """Get model architecture summary as string"""
        if self.model is None:
            return "No model built"
        
        # Capture model summary
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
    
    def plot_training_history(self, save_path: str = None) -> None:
        """
        Plot training history (requires matplotlib)
        
        Args:
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_history:
                logger.warning("No training history available")
                return
            
            history = self.training_history.history
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss
            axes[0, 0].plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            
            # MAE
            axes[0, 1].plot(history['mae'], label='Training MAE')
            if 'val_mae' in history:
                axes[0, 1].plot(history['val_mae'], label='Validation MAE')
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            
            # Learning rate (if available)
            if 'lr' in history:
                axes[1, 0].plot(history['lr'])
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
            
            # Additional metrics
            if '_rmse_metric' in history:
                axes[1, 1].plot(history['_rmse_metric'], label='Training RMSE')
                if 'val__rmse_metric' in history:
                    axes[1, 1].plot(history['val__rmse_metric'], label='Validation RMSE')
                axes[1, 1].set_title('Root Mean Square Error')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('RMSE')
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")

# Additional utility functions for ensemble models
class LSTMEnsemble:
    """
    Ensemble of LSTM models for improved predictions
    """
    
    def __init__(self, n_models: int = 5):
        """
        Initialize LSTM ensemble
        
        Args:
            n_models: Number of models in ensemble
        """
        self.n_models = n_models
        self.models = []
        self.is_trained = False
        
    def train_ensemble(self, 
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray = None,
                      y_val: np.ndarray = None,
                      **kwargs) -> List[TrainingHistory]:
        """Train ensemble of models"""
        histories = []
        
        for i in range(self.n_models):
            logger.info(f"Training ensemble model {i+1}/{self.n_models}")
            
            # Create model with slight variations
            model = LSTMModel(
                lstm_units=config.model.lstm_units,
                dropout_rate=config.model.dropout_rate + np.random.uniform(-0.05, 0.05),
                learning_rate=config.model.learning_rate * np.random.uniform(0.8, 1.2)
            )
            
            # Train model
            history = model.train(X_train, y_train, X_val, y_val, **kwargs)
            
            self.models.append(model)
            histories.append(history)
        
        self.is_trained = True
        return histories
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise PredictionError("Ensemble must be trained before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        return {
            "mean_prediction": np.mean(predictions, axis=0),
            "median_prediction": np.median(predictions, axis=0),
            "std_prediction": np.std(predictions, axis=0),
            "individual_predictions": predictions
        }