"""
Comprehensive risk assessment module for stock price predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics data structure"""
    # Volatility measures
    historical_volatility: float
    realized_volatility: float
    garch_volatility: float
    
    # Value at Risk
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Drawdown measures
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: float
    
    # Distribution measures
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_p_value: float
    
    # Tail risk
    tail_ratio: float
    expected_shortfall: float
    
    # Market risk
    beta: float
    correlation_with_market: float
    
    # Prediction risk
    prediction_uncertainty: float
    model_confidence: float
    
    # Composite scores
    overall_risk_score: float
    risk_grade: str

class RiskAssessment:
    """
    Comprehensive risk assessment for stock predictions and portfolios
    """
    
    def __init__(self, confidence_levels: List[float] = None):
        """
        Initialize risk assessment
        
        Args:
            confidence_levels: List of confidence levels for VaR calculations
        """
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.market_data = None
        
    def calculate_risk_metrics(self,
                             returns: Union[pd.Series, np.ndarray],
                             predictions: Optional[np.ndarray] = None,
                             prices: Optional[pd.Series] = None,
                             market_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Historical or predicted returns
            predictions: Model predictions (optional)
            prices: Price series (optional)
            market_returns: Market benchmark returns (optional)
            
        Returns:
            RiskMetrics object
        """
        try:
            logger.info("Calculating comprehensive risk metrics")
            
            if isinstance(returns, pd.Series):
                returns_array = returns.values
            else:
                returns_array = np.array(returns)
            
            # Remove NaN values
            returns_array = returns_array[~np.isnan(returns_array)]
            
            if len(returns_array) == 0:
                raise ValueError("No valid returns data provided")
            
            # Volatility measures
            volatility_metrics = self._calculate_volatility_metrics(returns_array)
            
            # Value at Risk measures
            var_metrics = self._calculate_var_metrics(returns_array)
            
            # Drawdown measures
            if prices is not None:
                drawdown_metrics = self._calculate_drawdown_metrics(prices)
            else:
                # Estimate from returns
                cumulative_returns = np.cumprod(1 + returns_array)
                drawdown_metrics = self._calculate_drawdown_from_returns(cumulative_returns)
            
            # Distribution measures
            distribution_metrics = self._calculate_distribution_metrics(returns_array)
            
            # Tail risk measures
            tail_risk_metrics = self._calculate_tail_risk_metrics(returns_array)
            
            # Market risk measures
            if market_returns is not None:
                market_risk_metrics = self._calculate_market_risk_metrics(returns_array, market_returns)
            else:
                market_risk_metrics = {'beta': 0.0, 'correlation_with_market': 0.0}
            
            # Prediction risk measures
            if predictions is not None:
                prediction_risk_metrics = self._calculate_prediction_risk_metrics(predictions, returns_array)
            else:
                prediction_risk_metrics = {'prediction_uncertainty': 0.0, 'model_confidence': 0.0}
            
            # Composite risk score
            composite_metrics = self._calculate_composite_risk_score(
                volatility_metrics, var_metrics, drawdown_metrics, distribution_metrics
            )
            
            # Create RiskMetrics object
            risk_metrics = RiskMetrics(
                # Volatility
                historical_volatility=volatility_metrics['historical_volatility'],
                realized_volatility=volatility_metrics['realized_volatility'],
                garch_volatility=volatility_metrics.get('garch_volatility', volatility_metrics['historical_volatility']),
                
                # VaR
                var_95=var_metrics['var_95'],
                var_99=var_metrics['var_99'],
                cvar_95=var_metrics['cvar_95'],
                cvar_99=var_metrics['cvar_99'],
                
                # Drawdown
                max_drawdown=drawdown_metrics['max_drawdown'],
                avg_drawdown=drawdown_metrics['avg_drawdown'],
                drawdown_duration=drawdown_metrics['avg_duration'],
                
                # Distribution
                skewness=distribution_metrics['skewness'],
                kurtosis=distribution_metrics['kurtosis'],
                jarque_bera_stat=distribution_metrics['jarque_bera_stat'],
                jarque_bera_p_value=distribution_metrics['jarque_bera_p_value'],
                
                # Tail risk
                tail_ratio=tail_risk_metrics['tail_ratio'],
                expected_shortfall=tail_risk_metrics['expected_shortfall'],
                
                # Market risk
                beta=market_risk_metrics['beta'],
                correlation_with_market=market_risk_metrics['correlation_with_market'],
                
                # Prediction risk
                prediction_uncertainty=prediction_risk_metrics['prediction_uncertainty'],
                model_confidence=prediction_risk_metrics['model_confidence'],
                
                # Composite
                overall_risk_score=composite_metrics['overall_risk_score'],
                risk_grade=composite_metrics['risk_grade']
            )
            
            logger.info(f"Risk assessment completed: {risk_metrics.risk_grade} risk profile")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            raise ValueError(f"Risk calculation failed: {e}")
    
    def _calculate_volatility_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate various volatility measures"""
        
        # Historical volatility (annualized)
        historical_vol = np.std(returns) * np.sqrt(252)
        
        # Realized volatility (using squared returns)
        realized_vol = np.sqrt(np.mean(returns ** 2)) * np.sqrt(252)
        
        # Simple GARCH-like volatility (exponentially weighted)
        weights = np.exp(-np.arange(len(returns)) * 0.05)[::-1]
        weights = weights / np.sum(weights)
        garch_vol = np.sqrt(np.sum(weights * returns ** 2)) * np.sqrt(252)
        
        return {
            'historical_volatility': float(historical_vol),
            'realized_volatility': float(realized_vol),
            'garch_volatility': float(garch_vol)
        }
    
    def _calculate_var_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate Value at Risk metrics"""
        
        var_metrics = {}
        
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            
            # Historical VaR
            var = np.percentile(returns, alpha * 100)
            
            # Conditional VaR (Expected Shortfall)
            cvar = np.mean(returns[returns <= var])
            
            # Store with confidence level
            var_key = f"var_{int(confidence_level * 100)}"
            cvar_key = f"cvar_{int(confidence_level * 100)}"
            
            var_metrics[var_key] = float(var)
            var_metrics[cvar_key] = float(cvar)
        
        return var_metrics
    
    def _calculate_drawdown_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate drawdown metrics from price series"""
        
        # Calculate running maximum
        peak = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - peak) / peak
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Drawdown duration
        drawdown_periods = []
        in_drawdown = False
        start_period = 0
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (1% threshold)
                in_drawdown = True
                start_period = i
            elif dd >= -0.01 and in_drawdown:  # End of drawdown
                in_drawdown = False
                drawdown_periods.append(i - start_period)
        
        avg_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': float(max_drawdown),
            'avg_drawdown': float(avg_drawdown),
            'avg_duration': float(avg_duration)
        }
    
    def _calculate_drawdown_from_returns(self, cumulative_returns: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown metrics from cumulative returns"""
        
        # Calculate running maximum
        peak = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - peak) / peak
        
        # Maximum drawdown
        max_drawdown = np.min(drawdown)
        
        # Average drawdown
        negative_drawdowns = drawdown[drawdown < 0]
        avg_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0
        
        # Simplified duration calculation
        avg_duration = len(negative_drawdowns) / len(drawdown) * 100 if len(drawdown) > 0 else 0
        
        return {
            'max_drawdown': float(max_drawdown),
            'avg_drawdown': float(avg_drawdown),
            'avg_duration': float(avg_duration)
        }
    
    def _calculate_distribution_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate distribution characteristics"""
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess kurtosis
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_p_value = stats.jarque_bera(returns)
        except:
            jb_stat, jb_p_value = 0, 1
        
        return {
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'jarque_bera_stat': float(jb_stat),
            'jarque_bera_p_value': float(jb_p_value)
        }
    
    def _calculate_tail_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate tail risk measures"""
        
        # Tail ratio (90th percentile / 10th percentile)
        p90 = np.percentile(returns, 90)
        p10 = np.percentile(returns, 10)
        tail_ratio = abs(p90 / p10) if p10 != 0 else 0
        
        # Expected shortfall (average of worst 5% returns)
        worst_5_percent = np.percentile(returns, 5)
        expected_shortfall = np.mean(returns[returns <= worst_5_percent])
        
        return {
            'tail_ratio': float(tail_ratio),
            'expected_shortfall': float(expected_shortfall)
        }
    
    def _calculate_market_risk_metrics(self, returns: np.ndarray, market_returns: pd.Series) -> Dict[str, float]:
        """Calculate market risk metrics"""
        
        try:
            # Align data
            min_length = min(len(returns), len(market_returns))
            asset_returns = returns[-min_length:]
            market_returns_array = market_returns.values[-min_length:]
            
            # Remove NaN values
            valid_indices = ~(np.isnan(asset_returns) | np.isnan(market_returns_array))
            asset_returns = asset_returns[valid_indices]
            market_returns_array = market_returns_array[valid_indices]
            
            if len(asset_returns) < 10:  # Need minimum data points
                return {'beta': 0.0, 'correlation_with_market': 0.0}
            
            # Beta calculation
            covariance = np.cov(asset_returns, market_returns_array)[0, 1]
            market_variance = np.var(market_returns_array)
            beta = covariance / market_variance if market_variance != 0 else 0
            
            # Correlation
            correlation = np.corrcoef(asset_returns, market_returns_array)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            return {
                'beta': float(beta),
                'correlation_with_market': float(correlation)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating market risk metrics: {e}")
            return {'beta': 0.0, 'correlation_with_market': 0.0}
    
    def _calculate_prediction_risk_metrics(self, predictions: np.ndarray, actual_returns: np.ndarray) -> Dict[str, float]:
        """Calculate prediction-specific risk metrics"""
        
        try:
            # Prediction uncertainty (standard deviation of prediction errors)
            min_length = min(len(predictions), len(actual_returns))
            pred_subset = predictions[-min_length:]
            actual_subset = actual_returns[-min_length:]
            
            prediction_errors = pred_subset - actual_subset
            prediction_uncertainty = np.std(prediction_errors)
            
            # Model confidence (inverse of normalized RMSE)
            rmse = np.sqrt(np.mean(prediction_errors ** 2))
            actual_std = np.std(actual_subset)
            normalized_rmse = rmse / actual_std if actual_std != 0 else float('inf')
            model_confidence = 1 / (1 + normalized_rmse)  # Bounded between 0 and 1
            
            return {
                'prediction_uncertainty': float(prediction_uncertainty),
                'model_confidence': float(model_confidence)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating prediction risk metrics: {e}")
            return {'prediction_uncertainty': 0.0, 'model_confidence': 0.0}
    
    def _calculate_composite_risk_score(self, 
                                      volatility_metrics: Dict,
                                      var_metrics: Dict,
                                      drawdown_metrics: Dict,
                                      distribution_metrics: Dict) -> Dict[str, Union[float, str]]:
        """Calculate composite risk score and grade"""
        
        # Normalize components (higher score = higher risk)
        vol_score = min(volatility_metrics['historical_volatility'] / 0.5, 1.0)  # Cap at 50% vol
        var_score = min(abs(var_metrics['var_95']) / 0.1, 1.0)  # Cap at 10% daily VaR
        drawdown_score = min(abs(drawdown_metrics['max_drawdown']) / 0.5, 1.0)  # Cap at 50% drawdown
        
        # Tail risk score
        tail_score = min(abs(distribution_metrics['kurtosis']) / 10, 1.0)  # High kurtosis = high tail risk
        
        # Weighted composite score
        weights = [0.3, 0.3, 0.25, 0.15]  # Vol, VaR, Drawdown, Tail
        composite_score = (
            weights[0] * vol_score +
            weights[1] * var_score +
            weights[2] * drawdown_score +
            weights[3] * tail_score
        )
        
        # Risk grade
        if composite_score <= 0.3:
            risk_grade = "Low"
        elif composite_score <= 0.6:
            risk_grade = "Medium"
        elif composite_score <= 0.8:
            risk_grade = "High"
        else:
            risk_grade = "Very High"
        
        return {
            'overall_risk_score': float(composite_score),
            'risk_grade': risk_grade
        }
    
    def generate_risk_report(self, risk_metrics: RiskMetrics) -> Dict[str, any]:
        """Generate comprehensive risk report"""
        
        report = {
            'executive_summary': {
                'overall_risk_grade': risk_metrics.risk_grade,
                'risk_score': risk_metrics.overall_risk_score,
                'key_concerns': self._identify_key_concerns(risk_metrics),
                'recommendations': self._generate_risk_recommendations(risk_metrics)
            },
            'volatility_analysis': {
                'historical_volatility': f"{risk_metrics.historical_volatility:.2%}",
                'annualized_volatility': f"{risk_metrics.historical_volatility:.2%}",
                'volatility_regime': self._classify_volatility_regime(risk_metrics.historical_volatility)
            },
            'downside_risk': {
                'var_95': f"{risk_metrics.var_95:.2%}",
                'var_99': f"{risk_metrics.var_99:.2%}",
                'expected_shortfall_95': f"{risk_metrics.cvar_95:.2%}",
                'max_drawdown': f"{risk_metrics.max_drawdown:.2%}",
                'drawdown_interpretation': self._interpret_drawdown(risk_metrics.max_drawdown)
            },
            'distribution_analysis': {
                'skewness': risk_metrics.skewness,
                'skewness_interpretation': self._interpret_skewness(risk_metrics.skewness),
                'kurtosis': risk_metrics.kurtosis,
                'kurtosis_interpretation': self._interpret_kurtosis(risk_metrics.kurtosis),
                'normality_test_p_value': risk_metrics.jarque_bera_p_value,
                'is_normal_distribution': risk_metrics.jarque_bera_p_value > 0.05
            },
            'market_risk': {
                'beta': risk_metrics.beta,
                'beta_interpretation': self._interpret_beta(risk_metrics.beta),
                'market_correlation': risk_metrics.correlation_with_market,
                'correlation_interpretation': self._interpret_correlation(risk_metrics.correlation_with_market)
            },
            'model_risk': {
                'prediction_uncertainty': risk_metrics.prediction_uncertainty,
                'model_confidence': f"{risk_metrics.model_confidence:.2%}",
                'confidence_interpretation': self._interpret_model_confidence(risk_metrics.model_confidence)
            },
            'risk_limits': self._suggest_risk_limits(risk_metrics),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return report
    
    def _identify_key_concerns(self, risk_metrics: RiskMetrics) -> List[str]:
        """Identify key risk concerns"""
        concerns = []
        
        if risk_metrics.historical_volatility > 0.4:
            concerns.append("High volatility - expect significant price swings")
        
        if abs(risk_metrics.max_drawdown) > 0.3:
            concerns.append("Large historical drawdowns - potential for significant losses")
        
        if abs(risk_metrics.var_95) > 0.05:
            concerns.append("High daily Value at Risk - daily losses could exceed 5%")
        
        if risk_metrics.kurtosis > 3:
            concerns.append("Fat tails in return distribution - higher probability of extreme events")
        
        if risk_metrics.skewness < -0.5:
            concerns.append("Negative skew - bias toward larger losses than gains")
        
        if risk_metrics.model_confidence < 0.6:
            concerns.append("Low model confidence - predictions may be unreliable")
        
        return concerns or ["No major risk concerns identified"]
    
    def _generate_risk_recommendations(self, risk_metrics: RiskMetrics) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_metrics.overall_risk_score > 0.7:
            recommendations.append("Consider reducing position size due to high overall risk")
        
        if abs(risk_metrics.max_drawdown) > 0.2:
            recommendations.append("Implement stop-loss mechanisms to limit drawdowns")
        
        if risk_metrics.historical_volatility > 0.3:
            recommendations.append("Use volatility-based position sizing")
        
        if risk_metrics.beta > 1.5:
            recommendations.append("High market sensitivity - consider hedging during market downturns")
        
        if risk_metrics.model_confidence < 0.7:
            recommendations.append("Low model confidence - use wider stop-losses and smaller positions")
        
        if risk_metrics.kurtosis > 5:
            recommendations.append("High tail risk - consider options strategies for downside protection")
        
        return recommendations or ["Risk profile is acceptable for current strategy"]
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility < 0.15:
            return "Low Volatility"
        elif volatility < 0.25:
            return "Normal Volatility"
        elif volatility < 0.40:
            return "High Volatility"
        else:
            return "Extreme Volatility"
    
    def _interpret_drawdown(self, max_drawdown: float) -> str:
        """Interpret maximum drawdown"""
        dd = abs(max_drawdown)
        if dd < 0.1:
            return "Low drawdown - well-controlled losses"
        elif dd < 0.2:
            return "Moderate drawdown - acceptable loss levels"
        elif dd < 0.3:
            return "High drawdown - significant loss potential"
        else:
            return "Extreme drawdown - severe loss risk"
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness"""
        if skewness > 0.5:
            return "Positive skew - bias toward larger gains"
        elif skewness < -0.5:
            return "Negative skew - bias toward larger losses"
        else:
            return "Approximately symmetric distribution"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis"""
        if kurtosis < 1:
            return "Thin tails - lower probability of extreme events"
        elif kurtosis < 3:
            return "Normal tails - typical extreme event probability"
        elif kurtosis < 6:
            return "Fat tails - higher probability of extreme events"
        else:
            return "Very fat tails - significant extreme event risk"
    
    def _interpret_beta(self, beta: float) -> str:
        """Interpret beta coefficient"""
        if abs(beta) < 0.5:
            return "Low market sensitivity"
        elif abs(beta) < 1.5:
            return "Normal market sensitivity"
        else:
            return "High market sensitivity"
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret market correlation"""
        corr = abs(correlation)
        if corr < 0.3:
            return "Low market correlation - good diversification"
        elif corr < 0.7:
            return "Moderate market correlation"
        else:
            return "High market correlation - limited diversification"
    
    def _interpret_model_confidence(self, confidence: float) -> str:
        """Interpret model confidence"""
        if confidence > 0.8:
            return "High confidence - reliable predictions"
        elif confidence > 0.6:
            return "Moderate confidence - reasonably reliable"
        elif confidence > 0.4:
            return "Low confidence - use with caution"
        else:
            return "Very low confidence - predictions unreliable"
    
    def _suggest_risk_limits(self, risk_metrics: RiskMetrics) -> Dict[str, str]:
        """Suggest risk limits based on metrics"""
        
        # Suggested position size (as percentage of portfolio)
        if risk_metrics.overall_risk_score < 0.3:
            max_position = "15-20%"
        elif risk_metrics.overall_risk_score < 0.6:
            max_position = "10-15%"
        elif risk_metrics.overall_risk_score < 0.8:
            max_position = "5-10%"
        else:
            max_position = "2-5%"
        
        # Stop loss level
        stop_loss = f"{abs(risk_metrics.var_99) * 2:.1%}"
        
        # Rebalancing frequency
        if risk_metrics.historical_volatility > 0.3:
            rebalance = "Daily"
        elif risk_metrics.historical_volatility > 0.2:
            rebalance = "Weekly"
        else:
            rebalance = "Monthly"
        
        return {
            'max_position_size': max_position,
            'suggested_stop_loss': stop_loss,
            'rebalancing_frequency': rebalance,
            'max_drawdown_limit': f"{abs(risk_metrics.max_drawdown) * 0.8:.1%}"
        }
    
    def calculate_portfolio_risk(self, 
                               positions: Dict[str, float],
                               individual_risks: Dict[str, RiskMetrics],
                               correlation_matrix: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """
        Calculate portfolio-level risk metrics
        
        Args:
            positions: Dictionary of {symbol: weight} for portfolio positions
            individual_risks: Dictionary of {symbol: RiskMetrics} for each asset
            correlation_matrix: Correlation matrix between assets (optional)
            
        Returns:
            Portfolio-level RiskMetrics
        """
        try:
            # Simple portfolio risk calculation (assuming equal correlation if matrix not provided)
            weights = np.array(list(positions.values()))
            volatilities = np.array([individual_risks[symbol].historical_volatility for symbol in positions.keys()])
            
            # Portfolio volatility
            if correlation_matrix is not None:
                # Use full correlation matrix
                symbols = list(positions.keys())
                corr_subset = correlation_matrix.loc[symbols, symbols]
                cov_matrix = np.outer(volatilities, volatilities) * corr_subset.values
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
            else:
                # Assume average correlation of 0.6
                avg_correlation = 0.6
                portfolio_variance = np.dot(weights**2, volatilities**2) + \
                                  avg_correlation * np.sum([weights[i] * weights[j] * volatilities[i] * volatilities[j] 
                                                           for i in range(len(weights)) for j in range(len(weights)) if i != j])
                portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Portfolio VaR (simplified)
            portfolio_var_95 = -1.65 * portfolio_volatility / np.sqrt(252)  # Daily VaR
            portfolio_var_99 = -2.33 * portfolio_volatility / np.sqrt(252)
            
            # Weighted average of other metrics
            weighted_max_dd = np.sum([positions[symbol] * abs(individual_risks[symbol].max_drawdown) 
                                    for symbol in positions.keys()])
            
            weighted_beta = np.sum([positions[symbol] * individual_risks[symbol].beta 
                                  for symbol in positions.keys()])
            
            # Composite risk score
            risk_scores = [individual_risks[symbol].overall_risk_score for symbol in positions.keys()]
            weighted_risk_score = np.sum([positions[symbol] * risk_scores[i] for i, symbol in enumerate(positions.keys())])
            
            # Determine risk grade
            if weighted_risk_score <= 0.3:
                risk_grade = "Low"
            elif weighted_risk_score <= 0.6:
                risk_grade = "Medium"
            elif weighted_risk_score <= 0.8:
                risk_grade = "High"
            else:
                risk_grade = "Very High"
            
            # Create portfolio risk metrics
            portfolio_risk = RiskMetrics(
                historical_volatility=portfolio_volatility,
                realized_volatility=portfolio_volatility,
                garch_volatility=portfolio_volatility,
                var_95=portfolio_var_95,
                var_99=portfolio_var_99,
                cvar_95=portfolio_var_95 * 1.2,  # Simplified
                cvar_99=portfolio_var_99 * 1.2,
                max_drawdown=-weighted_max_dd,
                avg_drawdown=-weighted_max_dd * 0.5,
                drawdown_duration=0,
                skewness=0,  # Would need return series to calculate
                kurtosis=0,
                jarque_bera_stat=0,
                jarque_bera_p_value=1,
                tail_ratio=0,
                expected_shortfall=portfolio_var_95 * 1.3,
                beta=weighted_beta,
                correlation_with_market=0,
                prediction_uncertainty=0,
                model_confidence=0,
                overall_risk_score=weighted_risk_score,
                risk_grade=risk_grade
            )
            
            logger.info(f"Portfolio risk calculated: {risk_grade} risk profile")
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            raise ValueError(f"Portfolio risk calculation failed: {e}")

def calculate_risk_adjusted_returns(returns: np.ndarray, 
                                   risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate risk-adjusted return metrics
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Dictionary of risk-adjusted metrics
    """
    try:
        # Convert annual risk-free rate to period rate
        period_rf_rate = risk_free_rate / 252  # Assuming daily returns
        
        # Excess returns
        excess_returns = returns - period_rf_rate
        
        # Sharpe ratio
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < period_rf_rate] - period_rf_rate
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        annual_return = np.mean(returns) * 252
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Treynor ratio (needs beta - simplified version)
        treynor_ratio = sharpe_ratio  # Simplified, would need market beta
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'treynor_ratio': float(treynor_ratio),
            'excess_return': float(np.mean(excess_returns) * 252)
        }
        
    except Exception as e:
        logger.error(f"Error calculating risk-adjusted returns: {e}")
        return {
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'treynor_ratio': 0,
            'excess_return': 0
        }