"""
Comprehensive backtesting engine for stock price prediction models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import warnings

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Result object for backtesting"""
    symbol: str
    start_date: str
    end_date: str
    prediction_horizon: int
    total_predictions: int
    
    # Performance metrics
    accuracy_metrics: Dict[str, float]
    trading_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    
    # Detailed results
    predictions: List[float]
    actual_values: List[float]
    prediction_dates: List[str]
    returns: List[float]
    
    # Metadata
    model_info: Dict[str, Any]
    backtest_config: Dict[str, Any]
    timestamp: str

class BacktestEngine:
    """
    Comprehensive backtesting engine for evaluating model performance
    """
    
    def __init__(self):
        """Initialize the backtest engine"""
        self.results_history = []
        
    def run_backtest(self,
                    model,
                    data_fetcher,
                    data_processor,
                    symbol: str,
                    start_date: str,
                    end_date: str,
                    prediction_horizon: int = 5,
                    rebalance_frequency: int = 1,
                    transaction_cost: float = 0.001,
                    initial_capital: float = 10000.0,
                    confidence_level: float = 0.95) -> BacktestResult:
        """
        Run comprehensive backtesting
        
        Args:
            model: Trained prediction model
            data_fetcher: Data fetching service
            data_processor: Data preprocessing service
            symbol: Stock symbol
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            prediction_horizon: Days ahead to predict
            rebalance_frequency: How often to rebalance (days)
            transaction_cost: Transaction cost as decimal
            initial_capital: Starting capital
            confidence_level: Confidence level for risk metrics
            
        Returns:
            BacktestResult object
        """
        try:
            logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
            
            # Fetch historical data
            stock_data = data_fetcher.fetch_stock_data(
                symbol=symbol,
                start=start_date,
                end=end_date,
                include_technical_indicators=True
            )
            
            # Prepare data
            feature_data = stock_data.primary_data[data_processor.feature_names + ['Close']]
            
            # Run walk-forward analysis
            backtest_data = self._walk_forward_analysis(
                model=model,
                data_processor=data_processor,
                feature_data=feature_data,
                prediction_horizon=prediction_horizon,
                rebalance_frequency=rebalance_frequency
            )
            
            # Calculate performance metrics
            accuracy_metrics = self._calculate_accuracy_metrics(
                predictions=backtest_data['predictions'],
                actual_values=backtest_data['actual_values']
            )
            
            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(
                predictions=backtest_data['predictions'],
                actual_values=backtest_data['actual_values'],
                prices=backtest_data['prices'],
                transaction_cost=transaction_cost,
                initial_capital=initial_capital
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                returns=backtest_data['returns'],
                confidence_level=confidence_level
            )
            
            # Create result object
            result = BacktestResult(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                prediction_horizon=prediction_horizon,
                total_predictions=len(backtest_data['predictions']),
                accuracy_metrics=accuracy_metrics,
                trading_metrics=trading_metrics,
                risk_metrics=risk_metrics,
                predictions=backtest_data['predictions'],
                actual_values=backtest_data['actual_values'],
                prediction_dates=[d.strftime('%Y-%m-%d') for d in backtest_data['dates']],
                returns=backtest_data['returns'],
                model_info=self._get_model_info(model),
                backtest_config={
                    'prediction_horizon': prediction_horizon,
                    'rebalance_frequency': rebalance_frequency,
                    'transaction_cost': transaction_cost,
                    'initial_capital': initial_capital,
                    'confidence_level': confidence_level
                },
                timestamp=datetime.now().isoformat()
            )
            
            self.results_history.append(result)
            
            logger.info(f"Backtest completed: {len(backtest_data['predictions'])} predictions")
            self._log_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise ValueError(f"Backtesting failed: {e}")
    
    def _walk_forward_analysis(self,
                              model,
                              data_processor,
                              feature_data: pd.DataFrame,
                              prediction_horizon: int,
                              rebalance_frequency: int) -> Dict[str, List]:
        """
        Perform walk-forward analysis
        """
        predictions = []
        actual_values = []
        dates = []
        prices = []
        returns = []
        
        # Parameters
        min_train_size = data_processor.sequence_length + 50
        
        # Walk forward through the data
        for i in range(min_train_size, len(feature_data) - prediction_horizon, rebalance_frequency):
            try:
                # Get training data up to current point
                train_data = feature_data.iloc[:i]
                
                # Scale the training data
                scaled_train_data = pd.DataFrame(
                    data_processor.scaler.transform(train_data),
                    columns=train_data.columns,
                    index=train_data.index
                )
                
                # Get sequence for prediction
                sequence = scaled_train_data.iloc[-data_processor.sequence_length:].values
                
                # Make prediction
                if len(sequence) == data_processor.sequence_length:
                    prediction = model.predict(sequence.reshape(1, *sequence.shape))
                    
                    # Inverse transform prediction
                    pred_original = data_processor.inverse_transform_predictions(
                        prediction,
                        train_data,
                        'Close'
                    )[0]
                    
                    # Get actual value
                    actual_idx = i + prediction_horizon - 1
                    if actual_idx < len(feature_data):
                        actual_value = feature_data.iloc[actual_idx]['Close']
                        current_price = feature_data.iloc[i-1]['Close']
                        
                        # Calculate return
                        return_value = (actual_value - current_price) / current_price
                        
                        predictions.append(pred_original)
                        actual_values.append(actual_value)
                        dates.append(feature_data.index[actual_idx])
                        prices.append(current_price)
                        returns.append(return_value)
                
            except Exception as e:
                logger.warning(f"Error in walk-forward step {i}: {e}")
                continue
        
        return {
            'predictions': predictions,
            'actual_values': actual_values,
            'dates': dates,
            'prices': prices,
            'returns': returns
        }
    
    def _calculate_accuracy_metrics(self,
                                   predictions: List[float],
                                   actual_values: List[float]) -> Dict[str, float]:
        """Calculate prediction accuracy metrics"""
        
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        # Basic error metrics
        mse = np.mean((actual_values - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_values - predictions))
        
        # MAPE with handling for zero values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
            if np.isnan(mape) or np.isinf(mape):
                mape = 0
        
        # R-squared
        ss_res = np.sum((actual_values - predictions) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(actual_values))
        predicted_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        # Correlation
        correlation = np.corrcoef(actual_values, predictions)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        # Theil's U statistic
        theil_u = self._calculate_theil_u(actual_values, predictions)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2_score': float(r2),
            'directional_accuracy': float(directional_accuracy),
            'correlation': float(correlation),
            'theil_u': float(theil_u)
        }
    
    def _calculate_trading_metrics(self,
                                  predictions: List[float],
                                  actual_values: List[float],
                                  prices: List[float],
                                  transaction_cost: float,
                                  initial_capital: float) -> Dict[str, float]:
        """Calculate trading performance metrics"""
        
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        prices = np.array(prices)
        
        # Generate trading signals (buy if prediction > current price)
        signals = np.where(predictions > prices, 1, -1)  # 1 = buy, -1 = sell
        
        # Calculate position changes for transaction costs
        position_changes = np.diff(np.concatenate([[0], signals])) != 0
        total_transactions = np.sum(position_changes)
        
        # Calculate returns based on signals
        actual_returns = (actual_values - prices) / prices
        strategy_returns = signals * actual_returns
        
        # Apply transaction costs
        transaction_costs = total_transactions * transaction_cost
        net_strategy_returns = strategy_returns - (transaction_costs / len(strategy_returns))
        
        # Performance metrics
        total_return = np.sum(net_strategy_returns)
        annualized_return = total_return * 252 / len(strategy_returns)  # Assuming daily data
        
        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        win_rate = winning_trades / len(strategy_returns) * 100
        
        # Sharpe ratio
        if np.std(net_strategy_returns) > 0:
            sharpe_ratio = np.mean(net_strategy_returns) / np.std(net_strategy_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + net_strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Average trade return
        avg_trade_return = np.mean(strategy_returns)
        
        # Hit ratio (percentage of correct directional predictions)
        correct_directions = np.sum((signals > 0) == (actual_returns > 0))
        hit_ratio = correct_directions / len(signals) * 100
        
        return {
            'total_return': float(total_return * 100),
            'annualized_return': float(annualized_return * 100),
            'win_rate': float(win_rate),
            'hit_ratio': float(hit_ratio),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown * 100),
            'calmar_ratio': float(calmar_ratio),
            'total_transactions': int(total_transactions),
            'avg_trade_return': float(avg_trade_return * 100),
            'transaction_cost_impact': float(transaction_costs * 100)
        }
    
    def _calculate_risk_metrics(self,
                               returns: List[float],
                               confidence_level: float) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        returns = np.array(returns)
        
        # Basic risk metrics
        volatility = np.std(returns)
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        
        # Value at Risk (VaR)
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        # Conditional Value at Risk (CVaR)
        cvar = np.mean(returns[returns <= var])
        
        # Maximum consecutive losses
        max_consecutive_losses = self._calculate_max_consecutive_losses(returns)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        # Sortino ratio (assuming risk-free rate = 0)
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum single loss
        max_single_loss = np.min(returns)
        
        # Recovery time (simplified)
        recovery_time = self._calculate_recovery_time(returns)
        
        return {
            'volatility': float(volatility * 100),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'var_95': float(var * 100),
            'cvar_95': float(cvar * 100),
            'max_consecutive_losses': int(max_consecutive_losses),
            'downside_deviation': float(downside_deviation * 100),
            'sortino_ratio': float(sortino_ratio),
            'max_single_loss': float(max_single_loss * 100),
            'recovery_time': int(recovery_time)
        }
    
    def _calculate_theil_u(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Theil's U statistic"""
        try:
            if len(actual) < 2:
                return float('inf')
            
            numerator = np.sqrt(np.mean((predicted[1:] - actual[1:]) ** 2))
            denominator = np.sqrt(np.mean((actual[1:] - actual[:-1]) ** 2))
            
            return numerator / denominator if denominator != 0 else float('inf')
        except:
            return float('inf')
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        except:
            return 0
    
    def _calculate_max_consecutive_losses(self, returns: np.ndarray) -> int:
        """Calculate maximum consecutive losses"""
        try:
            losses = returns < 0
            consecutive_losses = []
            current_streak = 0
            
            for loss in losses:
                if loss:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        consecutive_losses.append(current_streak)
                    current_streak = 0
            
            if current_streak > 0:
                consecutive_losses.append(current_streak)
            
            return max(consecutive_losses) if consecutive_losses else 0
        except:
            return 0
    
    def _calculate_recovery_time(self, returns: np.ndarray) -> int:
        """Calculate average recovery time from drawdowns"""
        try:
            cumulative = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            
            # Find drawdown periods
            in_drawdown = drawdown < -0.01  # 1% threshold
            recovery_times = []
            
            i = 0
            while i < len(in_drawdown):
                if in_drawdown[i]:
                    start = i
                    while i < len(in_drawdown) and in_drawdown[i]:
                        i += 1
                    recovery_times.append(i - start)
                else:
                    i += 1
            
            return int(np.mean(recovery_times)) if recovery_times else 0
        except:
            return 0
    
    def _get_model_info(self, model) -> Dict[str, Any]:
        """Get model information"""
        try:
            if hasattr(model, '_get_model_config'):
                return model._get_model_config()
            else:
                return {
                    'model_type': str(type(model).__name__),
                    'info': 'Model info not available'
                }
        except:
            return {'model_type': 'Unknown'}
    
    def _log_summary(self, result: BacktestResult) -> None:
        """Log backtest summary"""
        logger.info(f"Backtest Summary for {result.symbol}:")
        logger.info(f"  Period: {result.start_date} to {result.end_date}")
        logger.info(f"  Total Predictions: {result.total_predictions}")
        logger.info(f"  RMSE: {result.accuracy_metrics['rmse']:.6f}")
        logger.info(f"  Directional Accuracy: {result.accuracy_metrics['directional_accuracy']:.2f}%")
        logger.info(f"  Total Return: {result.trading_metrics['total_return']:.2f}%")
        logger.info(f"  Sharpe Ratio: {result.trading_metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown: {result.trading_metrics['max_drawdown']:.2f}%")
    
    def compare_strategies(self, results: List[BacktestResult]) -> pd.DataFrame:
        """
        Compare multiple backtest results
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            DataFrame comparing the strategies
        """
        comparison_data = []
        
        for result in results:
            row = {
                'Symbol': result.symbol,
                'Period': f"{result.start_date} to {result.end_date}",
                'Predictions': result.total_predictions,
                'RMSE': result.accuracy_metrics['rmse'],
                'MAE': result.accuracy_metrics['mae'],
                'Directional_Accuracy': result.accuracy_metrics['directional_accuracy'],
                'Total_Return': result.trading_metrics['total_return'],
                'Sharpe_Ratio': result.trading_metrics['sharpe_ratio'],
                'Max_Drawdown': result.trading_metrics['max_drawdown'],
                'Win_Rate': result.trading_metrics['win_rate'],
                'Volatility': result.risk_metrics['volatility']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add rankings
        for metric in ['RMSE', 'MAE', 'Max_Drawdown', 'Volatility']:
            if metric in comparison_df.columns:
                comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank()
        
        for metric in ['Directional_Accuracy', 'Total_Return', 'Sharpe_Ratio', 'Win_Rate']:
            if metric in comparison_df.columns:
                comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df
    
    def generate_backtest_report(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report
        
        Args:
            result: BacktestResult object
            
        Returns:
            Comprehensive report dictionary
        """
        
        # Performance summary
        performance_grade = self._grade_performance(result)
        
        # Risk assessment
        risk_grade = self._grade_risk(result)
        
        # Recommendations
        recommendations = self._generate_recommendations(result)
        
        report = {
            'summary': {
                'symbol': result.symbol,
                'backtest_period': f"{result.start_date} to {result.end_date}",
                'total_predictions': result.total_predictions,
                'performance_grade': performance_grade,
                'risk_grade': risk_grade
            },
            'accuracy_metrics': result.accuracy_metrics,
            'trading_metrics': result.trading_metrics,
            'risk_metrics': result.risk_metrics,
            'model_info': result.model_info,
            'recommendations': recommendations,
            'detailed_results': {
                'predictions': result.predictions[:10],  # First 10 predictions
                'actual_values': result.actual_values[:10],
                'prediction_dates': result.prediction_dates[:10]
            },
            'backtest_config': result.backtest_config,
            'timestamp': result.timestamp
        }
        
        return report
    
    def _grade_performance(self, result: BacktestResult) -> str:
        """Grade overall performance"""
        directional_accuracy = result.accuracy_metrics['directional_accuracy']
        sharpe_ratio = result.trading_metrics['sharpe_ratio']
        
        if directional_accuracy >= 60 and sharpe_ratio >= 1.5:
            return 'Excellent'
        elif directional_accuracy >= 55 and sharpe_ratio >= 1.0:
            return 'Good'
        elif directional_accuracy >= 50 and sharpe_ratio >= 0.5:
            return 'Fair'
        else:
            return 'Poor'
    
    def _grade_risk(self, result: BacktestResult) -> str:
        """Grade risk profile"""
        max_drawdown = abs(result.trading_metrics['max_drawdown'])
        volatility = result.risk_metrics['volatility']
        
        if max_drawdown <= 10 and volatility <= 20:
            return 'Low Risk'
        elif max_drawdown <= 20 and volatility <= 30:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    def _generate_recommendations(self, result: BacktestResult) -> List[str]:
        """Generate recommendations based on backtest results"""
        recommendations = []
        
        # Accuracy recommendations
        if result.accuracy_metrics['directional_accuracy'] < 55:
            recommendations.append("Consider improving model features or architecture to increase directional accuracy")
        
        # Risk recommendations
        if abs(result.trading_metrics['max_drawdown']) > 15:
            recommendations.append("Implement position sizing or stop-loss mechanisms to reduce maximum drawdown")
        
        # Return recommendations
        if result.trading_metrics['sharpe_ratio'] < 1.0:
            recommendations.append("Review trading strategy to improve risk-adjusted returns")
        
        # Transaction cost recommendations
        if result.trading_metrics['total_transactions'] > len(result.predictions) * 0.5:
            recommendations.append("Consider reducing trading frequency to minimize transaction costs")
        
        # Model stability
        if result.accuracy_metrics['theil_u'] > 1.5:
            recommendations.append("Model predictions may be less accurate than naive forecast - consider model improvements")
        
        if not recommendations:
            recommendations.append("Overall performance is satisfactory - continue monitoring and periodic retraining")
        
        return recommendations
    
    def save_backtest_results(self, result: BacktestResult, filepath: str) -> None:
        """Save backtest results to file"""
        try:
            import json
            
            # Convert result to dictionary
            result_dict = asdict(result)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            raise ValueError(f"Failed to save results: {e}")
    
    def load_backtest_results(self, filepath: str) -> BacktestResult:
        """Load backtest results from file"""
        try:
            import json
            
            with open(filepath, 'r') as f:
                result_dict = json.load(f)
            
            # Convert back to BacktestResult
            result = BacktestResult(**result_dict)
            
            logger.info(f"Backtest results loaded from {filepath}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading backtest results: {e}")
            raise ValueError(f"Failed to load results: {e}")