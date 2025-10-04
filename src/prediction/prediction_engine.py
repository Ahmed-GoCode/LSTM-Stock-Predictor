"""
Main prediction engine that coordinates all components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import joblib
import os

from ..data_fetcher.unified_fetcher import UnifiedDataFetcher, UnifiedStockData
from ..preprocessing.data_processor import DataProcessor, PreprocessingResult
from ..models.lstm_model import LSTMModel, TrainingHistory
from ..models.evaluation import ModelEvaluator, EvaluationMetrics
from ..utils.exceptions import PredictionError, ModelError, DataFetchError
from ..config.config import config

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Result object for predictions"""
    symbol: str
    predictions: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]
    dates: pd.DatetimeIndex
    metrics: Dict[str, float]
    model_info: Dict[str, Any]
    prediction_type: str  # 'historical', 'future', 'backtest'
    timestamp: str

@dataclass
class TrainingResult:
    """Result object for model training"""
    symbol: str
    model_metrics: EvaluationMetrics
    training_history: TrainingHistory
    preprocessing_info: Dict[str, Any]
    model_config: Dict[str, Any]
    training_time: float
    timestamp: str

class PredictionEngine:
    """
    Main prediction engine that orchestrates the entire prediction pipeline
    """
    
    def __init__(self, 
                 alpha_vantage_key: Optional[str] = None,
                 model_config: Optional[Dict] = None):
        """
        Initialize the prediction engine
        
        Args:
            alpha_vantage_key: Alpha Vantage API key (optional)
            model_config: Model configuration overrides
        """
        self.data_fetcher = UnifiedDataFetcher(alpha_vantage_key)
        self.data_processor = DataProcessor()
        self.model = LSTMModel()
        self.evaluator = ModelEvaluator()
        
        # Override model config if provided
        if model_config:
            for key, value in model_config.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        self.is_trained = False
        self.current_symbol = None
        self.preprocessing_result = None
        self.training_result = None
        
    def train_model(self, 
                   symbol: str,
                   period: str = "2y",
                   validation_split: float = None,
                   save_model: bool = True,
                   model_name: str = None) -> TrainingResult:
        """
        Complete training pipeline from data fetching to model training
        
        Args:
            symbol: Stock ticker symbol
            period: Data period for training
            validation_split: Validation split ratio
            save_model: Whether to save the trained model
            model_name: Custom model name for saving
            
        Returns:
            TrainingResult object
        """
        try:
            start_time = datetime.now()
            logger.info(f"Starting training pipeline for {symbol}")
            
            # Step 1: Fetch data
            logger.info("Step 1: Fetching stock data")
            stock_data = self.data_fetcher.fetch_stock_data(
                symbol=symbol,
                period=period,
                include_technical_indicators=True,
                include_fundamentals=False
            )
            
            # Validate data
            self.data_fetcher.validate_data(stock_data)
            
            # Step 2: Preprocess data
            logger.info("Step 2: Preprocessing data")
            self.preprocessing_result = self.data_processor.prepare_data(
                data=stock_data.primary_data,
                validation_split=validation_split
            )
            
            # Step 3: Build and train model
            logger.info("Step 3: Training LSTM model")
            self.model = LSTMModel(
                sequence_length=self.preprocessing_result.sequence_length
            )
            
            training_history = self.model.train(
                X_train=self.preprocessing_result.X_train,
                y_train=self.preprocessing_result.y_train,
                X_val=self.preprocessing_result.X_test,
                y_val=self.preprocessing_result.y_test
            )
            
            # Step 4: Evaluate model
            logger.info("Step 4: Evaluating model performance")
            test_predictions = self.model.predict(self.preprocessing_result.X_test)
            
            evaluation_metrics = self.evaluator.evaluate_predictions(
                y_true=self.preprocessing_result.y_test,
                y_pred=test_predictions
            )
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Create training result
            self.training_result = TrainingResult(
                symbol=symbol,
                model_metrics=evaluation_metrics,
                training_history=training_history,
                preprocessing_info=self.preprocessing_result.metadata,
                model_config=self.model._get_model_config(),
                training_time=training_time,
                timestamp=datetime.now().isoformat()
            )
            
            # Step 5: Save model if requested
            if save_model:
                model_name = model_name or f"{symbol}_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_path = os.path.join(config.models_dir, f"{model_name}.pkl")
                self.save_trained_model(model_path)
            
            self.is_trained = True
            self.current_symbol = symbol
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Model performance: RMSE={evaluation_metrics.rmse:.4f}, "
                       f"MAE={evaluation_metrics.mae:.4f}, "
                       f"Directional Accuracy={evaluation_metrics.directional_accuracy:.2f}%")
            
            return self.training_result
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise ModelError(f"Training pipeline failed: {e}")
    
    def predict(self, 
               symbol: str = None,
               n_days: int = 30,
               confidence_intervals: bool = True,
               return_probabilities: bool = False) -> PredictionResult:
        """
        Make future predictions using the trained model
        
        Args:
            symbol: Stock symbol (uses current symbol if None)
            n_days: Number of days to predict
            confidence_intervals: Whether to include confidence intervals
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            PredictionResult object
        """
        try:
            if not self.is_trained:
                raise PredictionError("Model must be trained before making predictions")
            
            symbol = symbol or self.current_symbol
            if not symbol:
                raise PredictionError("No symbol specified and no model trained")
            
            logger.info(f"Making {n_days}-day predictions for {symbol}")
            
            # Get latest data for the symbol
            stock_data = self.data_fetcher.fetch_stock_data(
                symbol=symbol,
                period="1y",  # Get recent data for prediction
                include_technical_indicators=True
            )
            
            # Preprocess the latest data using the saved processor
            if not hasattr(self.data_processor, 'scaler') or self.data_processor.scaler is None:
                raise PredictionError("Data processor not properly initialized. Please train model first.")
            
            # Use only the features that were used in training
            feature_data = stock_data.primary_data[self.preprocessing_result.feature_names + [config.data.target]]
            
            # Scale the data using the fitted scaler
            scaled_data = pd.DataFrame(
                self.data_processor.scaler.transform(feature_data),
                columns=feature_data.columns,
                index=feature_data.index
            )
            
            # Get the last sequence for prediction
            last_sequence = scaled_data.iloc[-self.preprocessing_result.sequence_length:].values
            
            # Make future predictions
            future_results = self.model.predict_future(
                last_sequence=last_sequence,
                n_steps=n_days,
                confidence_intervals=confidence_intervals
            )
            
            # Inverse transform predictions to original scale
            predictions = self.data_processor.inverse_transform_predictions(
                future_results["predictions"],
                feature_data,
                config.data.target
            )
            
            # Generate future dates
            last_date = stock_data.primary_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=n_days,
                freq='B'  # Business days
            )
            
            # Process confidence intervals if available
            confidence_data = {}
            if confidence_intervals and "lower_confidence" in future_results:
                confidence_data["lower"] = self.data_processor.inverse_transform_predictions(
                    future_results["lower_confidence"],
                    feature_data,
                    config.data.target
                )
                confidence_data["upper"] = self.data_processor.inverse_transform_predictions(
                    future_results["upper_confidence"],
                    feature_data,
                    config.data.target
                )
                confidence_data["std"] = self.data_processor.inverse_transform_predictions(
                    future_results["prediction_std"],
                    feature_data,
                    config.data.target
                )
            
            # Calculate prediction metrics
            metrics = {
                "prediction_horizon": n_days,
                "last_price": float(stock_data.primary_data[config.data.target].iloc[-1]),
                "predicted_return": float((predictions[-1] - predictions[0]) / predictions[0] * 100),
                "volatility": float(np.std(predictions)),
                "trend": "upward" if predictions[-1] > predictions[0] else "downward"
            }
            
            # Create result object
            result = PredictionResult(
                symbol=symbol,
                predictions=predictions,
                confidence_intervals=confidence_data,
                dates=future_dates,
                metrics=metrics,
                model_info=self.model._get_model_config(),
                prediction_type="future",
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Predictions completed: {metrics['trend']} trend, "
                       f"{metrics['predicted_return']:.2f}% total return")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise PredictionError(f"Prediction failed: {e}")
    
    def backtest_predictions(self, 
                           symbol: str,
                           start_date: str,
                           end_date: str,
                           prediction_horizon: int = 5) -> Dict[str, Any]:
        """
        Backtest the model on historical data
        
        Args:
            symbol: Stock symbol
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            prediction_horizon: Days ahead to predict
            
        Returns:
            Backtest results dictionary
        """
        try:
            if not self.is_trained:
                raise PredictionError("Model must be trained before backtesting")
            
            logger.info(f"Backtesting {symbol} from {start_date} to {end_date}")
            
            # Fetch historical data for backtesting
            stock_data = self.data_fetcher.fetch_stock_data(
                symbol=symbol,
                start=start_date,
                end=end_date,
                include_technical_indicators=True
            )
            
            # Prepare data for backtesting
            feature_data = stock_data.primary_data[self.preprocessing_result.feature_names + [config.data.target]]
            scaled_data = pd.DataFrame(
                self.data_processor.scaler.transform(feature_data),
                columns=feature_data.columns,
                index=feature_data.index
            )
            
            # Perform rolling window backtesting
            backtest_results = self._rolling_window_backtest(
                scaled_data=scaled_data,
                feature_data=feature_data,
                prediction_horizon=prediction_horizon
            )
            
            # Evaluate backtest performance
            evaluation_report = self.evaluator.generate_evaluation_report(
                y_true=backtest_results["actual_values"],
                y_pred=backtest_results["predicted_values"],
                model_name=f"LSTM_Backtest_{symbol}",
                prices=backtest_results.get("prices")
            )
            
            # Combine results
            final_results = {
                "symbol": symbol,
                "backtest_period": f"{start_date} to {end_date}",
                "prediction_horizon": prediction_horizon,
                "backtest_data": backtest_results,
                "evaluation": evaluation_report,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Backtesting completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            raise PredictionError(f"Backtesting failed: {e}")
    
    def _rolling_window_backtest(self, 
                               scaled_data: pd.DataFrame,
                               feature_data: pd.DataFrame,
                               prediction_horizon: int) -> Dict[str, List]:
        """Perform rolling window backtesting"""
        
        predictions = []
        actual_values = []
        prediction_dates = []
        prices = []
        
        # Rolling window parameters
        min_data_points = self.preprocessing_result.sequence_length + 50
        
        for i in range(min_data_points, len(scaled_data) - prediction_horizon):
            try:
                # Get sequence for prediction
                sequence = scaled_data.iloc[i-self.preprocessing_result.sequence_length:i].values
                
                # Make prediction
                prediction = self.model.predict(sequence.reshape(1, *sequence.shape))
                
                # Inverse transform prediction
                pred_original = self.data_processor.inverse_transform_predictions(
                    prediction,
                    feature_data.iloc[:i+1],
                    config.data.target
                )[0]
                
                # Get actual value
                actual_idx = i + prediction_horizon - 1
                if actual_idx < len(feature_data):
                    actual_value = feature_data.iloc[actual_idx][config.data.target]
                    
                    predictions.append(pred_original)
                    actual_values.append(actual_value)
                    prediction_dates.append(feature_data.index[actual_idx])
                    prices.append(feature_data.iloc[i-1][config.data.target])
                
            except Exception as e:
                logger.warning(f"Error in backtest step {i}: {e}")
                continue
        
        return {
            "predicted_values": np.array(predictions),
            "actual_values": np.array(actual_values),
            "prediction_dates": prediction_dates,
            "prices": np.array(prices)
        }
    
    def save_trained_model(self, filepath: str) -> None:
        """
        Save the complete trained model and preprocessing pipeline
        
        Args:
            filepath: Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save LSTM model
            self.model.save_model(filepath)
            
            # Save data processor
            processor_path = filepath.replace('.pkl', '_processor.pkl')
            self.data_processor.save_processor(processor_path)
            
            # Save preprocessing result
            preprocessing_path = filepath.replace('.pkl', '_preprocessing.pkl')
            joblib.dump(self.preprocessing_result, preprocessing_path)
            
            # Save training result
            if self.training_result:
                training_path = filepath.replace('.pkl', '_training.pkl')
                joblib.dump(self.training_result, training_path)
            
            logger.info(f"Complete model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise ModelError(f"Failed to save model: {e}")
    
    def load_trained_model(self, filepath: str) -> None:
        """
        Load a complete trained model and preprocessing pipeline
        
        Args:
            filepath: Path to the saved model
        """
        try:
            # Load LSTM model
            self.model = LSTMModel()
            self.model.load_model(filepath)
            
            # Load data processor
            processor_path = filepath.replace('.pkl', '_processor.pkl')
            self.data_processor = DataProcessor()
            self.data_processor.load_processor(processor_path)
            
            # Load preprocessing result
            preprocessing_path = filepath.replace('.pkl', '_preprocessing.pkl')
            if os.path.exists(preprocessing_path):
                self.preprocessing_result = joblib.load(preprocessing_path)
            
            # Load training result
            training_path = filepath.replace('.pkl', '_training.pkl')
            if os.path.exists(training_path):
                self.training_result = joblib.load(training_path)
            
            self.is_trained = True
            logger.info(f"Complete model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelError(f"Failed to load model: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        summary = {
            "status": "trained",
            "current_symbol": self.current_symbol,
            "model_architecture": self.model.get_model_summary(),
            "training_metrics": asdict(self.training_result.model_metrics) if self.training_result else {},
            "preprocessing_info": self.preprocessing_result.metadata if self.preprocessing_result else {},
            "model_config": self.model._get_model_config(),
            "feature_count": len(self.preprocessing_result.feature_names) if self.preprocessing_result else 0,
            "sequence_length": self.preprocessing_result.sequence_length if self.preprocessing_result else 0,
            "training_time": self.training_result.training_time if self.training_result else 0
        }
        
        return summary
    
    def export_predictions(self, 
                         prediction_result: PredictionResult,
                         filepath: str,
                         format: str = "csv") -> None:
        """
        Export predictions to file
        
        Args:
            prediction_result: PredictionResult object
            filepath: Output file path
            format: Export format ('csv', 'json', 'excel')
        """
        try:
            # Prepare data for export
            export_data = {
                "Date": prediction_result.dates,
                "Predicted_Price": prediction_result.predictions,
                "Symbol": prediction_result.symbol
            }
            
            # Add confidence intervals if available
            if prediction_result.confidence_intervals:
                if "lower" in prediction_result.confidence_intervals:
                    export_data["Lower_Confidence"] = prediction_result.confidence_intervals["lower"]
                if "upper" in prediction_result.confidence_intervals:
                    export_data["Upper_Confidence"] = prediction_result.confidence_intervals["upper"]
                if "std" in prediction_result.confidence_intervals:
                    export_data["Prediction_Std"] = prediction_result.confidence_intervals["std"]
            
            df = pd.DataFrame(export_data)
            
            # Export based on format
            if format.lower() == "csv":
                df.to_csv(filepath, index=False)
            elif format.lower() == "json":
                # Include metadata in JSON export
                export_dict = {
                    "metadata": {
                        "symbol": prediction_result.symbol,
                        "prediction_type": prediction_result.prediction_type,
                        "timestamp": prediction_result.timestamp,
                        "metrics": prediction_result.metrics,
                        "model_info": prediction_result.model_info
                    },
                    "predictions": df.to_dict(orient="records")
                }
                
                import json
                with open(filepath, 'w') as f:
                    json.dump(export_dict, f, indent=2, default=str)
            
            elif format.lower() == "excel":
                with pd.ExcelWriter(filepath) as writer:
                    df.to_excel(writer, sheet_name="Predictions", index=False)
                    
                    # Add metadata sheet
                    metadata_df = pd.DataFrame([
                        {"Metric": k, "Value": v} 
                        for k, v in prediction_result.metrics.items()
                    ])
                    metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Predictions exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting predictions: {e}")
            raise PredictionError(f"Export failed: {e}")