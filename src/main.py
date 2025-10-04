"""
Main entry point for the Stock Price Predictor
"""

import logging
import sys
import os
from typing import Optional, Dict, Any
import argparse
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.prediction.prediction_engine import PredictionEngine, PredictionResult, TrainingResult
from src.config.config import config
from src.utils.exceptions import StockPredictorError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('stock_predictor.log')
    ]
)

logger = logging.getLogger(__name__)

class StockPredictor:
    """
    Main class for the Stock Price Predictor application
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize the Stock Predictor
        
        Args:
            alpha_vantage_key: Alpha Vantage API key (optional)
        """
        self.engine = PredictionEngine(alpha_vantage_key=alpha_vantage_key)
        logger.info("Stock Predictor initialized successfully")
    
    def train(self, 
              symbol: str,
              period: str = "2y",
              save_model: bool = True,
              model_name: Optional[str] = None) -> TrainingResult:
        """
        Train a model for a specific stock
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Training data period ('1y', '2y', '5y', etc.)
            save_model: Whether to save the trained model
            model_name: Custom name for the model
            
        Returns:
            TrainingResult object
        """
        try:
            logger.info(f"Starting training for {symbol}")
            
            result = self.engine.train_model(
                symbol=symbol,
                period=period,
                save_model=save_model,
                model_name=model_name
            )
            
            logger.info(f"Training completed successfully for {symbol}")
            self._print_training_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            raise StockPredictorError(f"Training failed: {e}")
    
    def predict(self, 
                symbol: Optional[str] = None,
                days: int = 30,
                confidence_intervals: bool = True) -> PredictionResult:
        """
        Make future predictions
        
        Args:
            symbol: Stock symbol (uses trained symbol if None)
            days: Number of days to predict
            confidence_intervals: Include confidence intervals
            
        Returns:
            PredictionResult object
        """
        try:
            result = self.engine.predict(
                symbol=symbol,
                n_days=days,
                confidence_intervals=confidence_intervals
            )
            
            self._print_prediction_summary(result)
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise StockPredictorError(f"Prediction failed: {e}")
    
    def backtest(self, 
                 symbol: str,
                 start_date: str,
                 end_date: str,
                 prediction_horizon: int = 5) -> Dict[str, Any]:
        """
        Backtest the model on historical data
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            prediction_horizon: Days ahead to predict
            
        Returns:
            Backtest results
        """
        try:
            result = self.engine.backtest_predictions(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                prediction_horizon=prediction_horizon
            )
            
            self._print_backtest_summary(result)
            return result
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            raise StockPredictorError(f"Backtesting failed: {e}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a previously trained model
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.engine.load_trained_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise StockPredictorError(f"Model loading failed: {e}")
    
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
            self.engine.export_predictions(prediction_result, filepath, format)
            logger.info(f"Predictions exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise StockPredictorError(f"Export failed: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the current model"""
        return self.engine.get_model_summary()
    
    def _print_training_summary(self, result: TrainingResult) -> None:
        """Print training summary"""
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY FOR {result.symbol}")
        print(f"{'='*60}")
        print(f"Training Time: {result.training_time:.2f} seconds")
        print(f"Model Architecture: LSTM")
        print(f"Sequence Length: {result.preprocessing_info.get('sequence_length', 'N/A')}")
        print(f"Features Used: {len(result.preprocessing_info.get('feature_columns', []))}")
        print(f"\nPerformance Metrics:")
        print(f"  RMSE: {result.model_metrics.rmse:.6f}")
        print(f"  MAE: {result.model_metrics.mae:.6f}")
        print(f"  MAPE: {result.model_metrics.mape:.2f}%")
        print(f"  R²: {result.model_metrics.r2_score:.4f}")
        print(f"  Directional Accuracy: {result.model_metrics.directional_accuracy:.2f}%")
        print(f"  Correlation: {result.model_metrics.correlation:.4f}")
        print(f"{'='*60}\n")
    
    def _print_prediction_summary(self, result: PredictionResult) -> None:
        """Print prediction summary"""
        print(f"\n{'='*60}")
        print(f"PREDICTION SUMMARY FOR {result.symbol}")
        print(f"{'='*60}")
        print(f"Prediction Horizon: {len(result.predictions)} days")
        print(f"Current Price: ${result.metrics['last_price']:.2f}")
        print(f"Predicted Final Price: ${result.predictions[-1]:.2f}")
        print(f"Expected Return: {result.metrics['predicted_return']:.2f}%")
        print(f"Trend Direction: {result.metrics['trend'].title()}")
        print(f"Volatility: {result.metrics['volatility']:.4f}")
        
        if result.confidence_intervals:
            print(f"\nConfidence Intervals (95%):")
            if 'lower' in result.confidence_intervals:
                print(f"  Lower Bound: ${result.confidence_intervals['lower'][-1]:.2f}")
                print(f"  Upper Bound: ${result.confidence_intervals['upper'][-1]:.2f}")
        
        print(f"{'='*60}\n")
    
    def _print_backtest_summary(self, result: Dict[str, Any]) -> None:
        """Print backtest summary"""
        eval_metrics = result.get('evaluation', {}).get('basic_metrics', {})
        trading_metrics = result.get('evaluation', {}).get('trading_metrics', {})
        
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY FOR {result['symbol']}")
        print(f"{'='*60}")
        print(f"Period: {result['backtest_period']}")
        print(f"Prediction Horizon: {result['prediction_horizon']} days")
        
        if eval_metrics:
            print(f"\nPrediction Accuracy:")
            print(f"  RMSE: {eval_metrics.get('rmse', 0):.6f}")
            print(f"  MAE: {eval_metrics.get('mae', 0):.6f}")
            print(f"  MAPE: {eval_metrics.get('mape', 0):.2f}%")
            print(f"  Directional Accuracy: {eval_metrics.get('directional_accuracy', 0):.2f}%")
        
        if trading_metrics:
            print(f"\nTrading Performance:")
            print(f"  Total Return: {trading_metrics.get('total_return', 0):.2f}%")
            print(f"  Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  Max Drawdown: {trading_metrics.get('max_drawdown', 0):.2f}%")
            print(f"  Win Rate: {trading_metrics.get('win_rate', 0):.2f}%")
        
        print(f"{'='*60}\n")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Stock Price Predictor using LSTM")
    
    parser.add_argument("command", choices=["train", "predict", "backtest"], 
                       help="Command to execute")
    parser.add_argument("--symbol", "-s", required=True, 
                       help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--period", "-p", default="2y",
                       help="Training data period (default: 2y)")
    parser.add_argument("--days", "-d", type=int, default=30,
                       help="Number of days to predict (default: 30)")
    parser.add_argument("--start-date", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--model-name", help="Custom model name")
    parser.add_argument("--load-model", help="Path to load existing model")
    parser.add_argument("--export", help="Export predictions to file")
    parser.add_argument("--format", choices=["csv", "json", "excel"], default="csv",
                       help="Export format (default: csv)")
    parser.add_argument("--alpha-vantage-key", help="Alpha Vantage API key")
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = StockPredictor(alpha_vantage_key=args.alpha_vantage_key)
        
        # Load existing model if specified
        if args.load_model:
            predictor.load_model(args.load_model)
        
        # Execute command
        if args.command == "train":
            result = predictor.train(
                symbol=args.symbol,
                period=args.period,
                model_name=args.model_name
            )
            
        elif args.command == "predict":
            result = predictor.predict(
                symbol=args.symbol,
                days=args.days
            )
            
            # Export if requested
            if args.export:
                predictor.export_predictions(result, args.export, args.format)
                
        elif args.command == "backtest":
            if not args.start_date or not args.end_date:
                print("Error: --start-date and --end-date required for backtesting")
                sys.exit(1)
                
            result = predictor.backtest(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date
            )
        
        print("Operation completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"CLI error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()