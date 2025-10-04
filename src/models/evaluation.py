"""
Model evaluation and performance metrics for stock prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    # Error metrics
    mse: float
    rmse: float
    mae: float
    mape: float
    
    # Statistical metrics
    r2_score: float
    correlation: float
    
    # Trading metrics
    directional_accuracy: float
    hit_rate: float
    
    # Risk metrics
    max_error: float
    error_std: float
    
    # Additional metrics
    theil_u: float
    mean_bias: float

class ModelEvaluator:
    """
    Comprehensive model evaluation for stock price prediction
    """
    
    def __init__(self):
        """Initialize the model evaluator"""
        self.evaluation_results = {}
        
    def evaluate_predictions(self, 
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           prices: np.ndarray = None,
                           dates: pd.DatetimeIndex = None) -> EvaluationMetrics:
        """
        Comprehensive evaluation of model predictions
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            prices: Original price series (for percentage calculations)
            dates: Date index for time-series specific metrics
            
        Returns:
            EvaluationMetrics object
        """
        try:
            logger.info("Evaluating model predictions")
            
            # Basic error metrics
            mse = self._calculate_mse(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = self._calculate_mae(y_true, y_pred)
            mape = self._calculate_mape(y_true, y_pred)
            
            # Statistical metrics
            r2_score = self._calculate_r2(y_true, y_pred)
            correlation = self._calculate_correlation(y_true, y_pred)
            
            # Trading metrics
            directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)
            hit_rate = self._calculate_hit_rate(y_true, y_pred)
            
            # Risk metrics
            errors = y_true - y_pred
            max_error = np.max(np.abs(errors))
            error_std = np.std(errors)
            
            # Advanced metrics
            theil_u = self._calculate_theil_u(y_true, y_pred)
            mean_bias = np.mean(errors)
            
            metrics = EvaluationMetrics(
                mse=float(mse),
                rmse=float(rmse),
                mae=float(mae),
                mape=float(mape),
                r2_score=float(r2_score),
                correlation=float(correlation),
                directional_accuracy=float(directional_accuracy),
                hit_rate=float(hit_rate),
                max_error=float(max_error),
                error_std=float(error_std),
                theil_u=float(theil_u),
                mean_bias=float(mean_bias)
            )
            
            logger.info(f"Evaluation completed: RMSE={rmse:.4f}, MAE={mae:.4f}, DA={directional_accuracy:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise ValueError(f"Evaluation failed: {e}")
    
    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _calculate_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient"""
        try:
            correlation, _ = stats.pearsonr(y_true, y_pred)
            return correlation if not np.isnan(correlation) else 0
        except:
            return 0
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions)"""
        if len(y_true) < 2:
            return 0
        
        actual_direction = np.sign(np.diff(y_true))
        predicted_direction = np.sign(np.diff(y_pred))
        
        # Remove cases where actual direction is 0 (no change)
        non_zero_indices = actual_direction != 0
        if np.sum(non_zero_indices) == 0:
            return 0
        
        accuracy = np.mean(actual_direction[non_zero_indices] == predicted_direction[non_zero_indices])
        return accuracy * 100
    
    def _calculate_hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.02) -> float:
        """Calculate hit rate (percentage of predictions within tolerance)"""
        relative_errors = np.abs((y_true - y_pred) / y_true)
        hits = np.sum(relative_errors <= tolerance)
        return (hits / len(y_true)) * 100
    
    def _calculate_theil_u(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Theil's U statistic"""
        try:
            if len(y_true) < 2:
                return float('inf')
            
            # Theil's U statistic
            numerator = np.sqrt(np.mean((y_pred[1:] - y_true[1:]) ** 2))
            denominator = np.sqrt(np.mean((y_true[1:] - y_true[:-1]) ** 2))
            
            return numerator / denominator if denominator != 0 else float('inf')
        except:
            return float('inf')
    
    def calculate_confidence_intervals(self, 
                                     errors: np.ndarray,
                                     confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence intervals for predictions
        
        Args:
            errors: Prediction errors
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Dictionary with confidence interval bounds
        """
        try:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(errors, lower_percentile)
            upper_bound = np.percentile(errors, upper_percentile)
            
            return {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "confidence_level": confidence_level
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {"lower_bound": 0, "upper_bound": 0, "confidence_level": confidence_level}
    
    def analyze_prediction_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Analyze prediction errors in detail
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with error analysis
        """
        try:
            errors = y_true - y_pred
            relative_errors = errors / y_true
            
            analysis = {
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "std_error": float(np.std(errors)),
                "min_error": float(np.min(errors)),
                "max_error": float(np.max(errors)),
                "skewness": float(stats.skew(errors)),
                "kurtosis": float(stats.kurtosis(errors)),
                "mean_relative_error": float(np.mean(relative_errors)),
                "median_relative_error": float(np.median(relative_errors)),
                "std_relative_error": float(np.std(relative_errors))
            }
            
            # Test for normality of errors
            _, p_value = stats.normaltest(errors)
            analysis["normality_p_value"] = float(p_value)
            analysis["errors_normal"] = p_value > 0.05
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in error analysis: {e}")
            return {}
    
    def calculate_trading_metrics(self, 
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                prices: np.ndarray,
                                transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate trading-specific metrics
        
        Args:
            y_true: Actual price changes
            y_pred: Predicted price changes
            prices: Original price series
            transaction_cost: Transaction cost as percentage
            
        Returns:
            Dictionary with trading metrics
        """
        try:
            # Generate trading signals
            predicted_direction = np.sign(y_pred)
            actual_direction = np.sign(y_true)
            
            # Calculate returns based on predictions
            predicted_returns = predicted_direction * (y_true / prices[:-1])
            
            # Apply transaction costs
            # Assume transaction when prediction changes direction
            position_changes = np.diff(predicted_direction) != 0
            transaction_costs = np.sum(position_changes) * transaction_cost
            
            # Calculate trading metrics
            total_return = np.sum(predicted_returns) - transaction_costs
            win_rate = np.mean(predicted_returns > 0) * 100
            
            # Sharpe ratio (annualized)
            if np.std(predicted_returns) > 0:
                sharpe_ratio = (np.mean(predicted_returns) * 252) / (np.std(predicted_returns) * np.sqrt(252))
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + predicted_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100
            
            return {
                "total_return": float(total_return * 100),
                "annualized_return": float(total_return * 252 * 100),
                "win_rate": float(win_rate),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "num_trades": int(np.sum(position_changes)),
                "avg_return_per_trade": float(np.mean(predicted_returns) * 100)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {}
    
    def generate_evaluation_report(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str = "LSTM",
                                 prices: np.ndarray = None) -> Dict[str, any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            prices: Original price series
            
        Returns:
            Comprehensive evaluation report
        """
        try:
            logger.info(f"Generating evaluation report for {model_name}")
            
            # Basic metrics
            basic_metrics = self.evaluate_predictions(y_true, y_pred, prices)
            
            # Error analysis
            error_analysis = self.analyze_prediction_errors(y_true, y_pred)
            
            # Confidence intervals
            errors = y_true - y_pred
            confidence_intervals = self.calculate_confidence_intervals(errors)
            
            # Trading metrics (if prices available)
            trading_metrics = {}
            if prices is not None and len(prices) == len(y_true) + 1:
                trading_metrics = self.calculate_trading_metrics(y_true, y_pred, prices)
            
            report = {
                "model_name": model_name,
                "evaluation_timestamp": pd.Timestamp.now().isoformat(),
                "data_points": len(y_true),
                "basic_metrics": basic_metrics.__dict__,
                "error_analysis": error_analysis,
                "confidence_intervals": confidence_intervals,
                "trading_metrics": trading_metrics,
                "summary": self._generate_summary(basic_metrics, trading_metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            return {"error": str(e)}
    
    def _generate_summary(self, 
                         basic_metrics: EvaluationMetrics,
                         trading_metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate a summary interpretation of the metrics"""
        summary = {}
        
        # RMSE interpretation
        if basic_metrics.rmse < 0.01:
            summary["rmse_quality"] = "Excellent"
        elif basic_metrics.rmse < 0.05:
            summary["rmse_quality"] = "Good"
        elif basic_metrics.rmse < 0.1:
            summary["rmse_quality"] = "Fair"
        else:
            summary["rmse_quality"] = "Poor"
        
        # Directional accuracy interpretation
        if basic_metrics.directional_accuracy > 60:
            summary["direction_quality"] = "Good"
        elif basic_metrics.directional_accuracy > 50:
            summary["direction_quality"] = "Fair"
        else:
            summary["direction_quality"] = "Poor"
        
        # R² interpretation
        if basic_metrics.r2_score > 0.8:
            summary["r2_quality"] = "Excellent"
        elif basic_metrics.r2_score > 0.6:
            summary["r2_quality"] = "Good"
        elif basic_metrics.r2_score > 0.4:
            summary["r2_quality"] = "Fair"
        else:
            summary["r2_quality"] = "Poor"
        
        # Trading performance (if available)
        if trading_metrics and "sharpe_ratio" in trading_metrics:
            if trading_metrics["sharpe_ratio"] > 1.5:
                summary["trading_quality"] = "Excellent"
            elif trading_metrics["sharpe_ratio"] > 1.0:
                summary["trading_quality"] = "Good"
            elif trading_metrics["sharpe_ratio"] > 0.5:
                summary["trading_quality"] = "Fair"
            else:
                summary["trading_quality"] = "Poor"
        
        return summary
    
    def compare_models(self, 
                      evaluations: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple model evaluations
        
        Args:
            evaluations: Dictionary of model evaluations
            
        Returns:
            DataFrame comparing models
        """
        try:
            comparison_data = []
            
            for model_name, evaluation in evaluations.items():
                if "basic_metrics" in evaluation:
                    metrics = evaluation["basic_metrics"]
                    trading = evaluation.get("trading_metrics", {})
                    
                    row = {
                        "Model": model_name,
                        "RMSE": metrics.get("rmse", 0),
                        "MAE": metrics.get("mae", 0),
                        "MAPE": metrics.get("mape", 0),
                        "R²": metrics.get("r2_score", 0),
                        "Directional Accuracy": metrics.get("directional_accuracy", 0),
                        "Correlation": metrics.get("correlation", 0)
                    }
                    
                    # Add trading metrics if available
                    if trading:
                        row.update({
                            "Sharpe Ratio": trading.get("sharpe_ratio", 0),
                            "Max Drawdown": trading.get("max_drawdown", 0),
                            "Win Rate": trading.get("win_rate", 0)
                        })
                    
                    comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Rank models
            if not comparison_df.empty:
                # Lower is better for error metrics
                for col in ["RMSE", "MAE", "MAPE", "Max Drawdown"]:
                    if col in comparison_df.columns:
                        comparison_df[f"{col}_Rank"] = comparison_df[col].rank()
                
                # Higher is better for other metrics
                for col in ["R²", "Directional Accuracy", "Correlation", "Sharpe Ratio", "Win Rate"]:
                    if col in comparison_df.columns:
                        comparison_df[f"{col}_Rank"] = comparison_df[col].rank(ascending=False)
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()
    
    def save_evaluation_report(self, report: Dict, filepath: str) -> None:
        """Save evaluation report to file"""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")