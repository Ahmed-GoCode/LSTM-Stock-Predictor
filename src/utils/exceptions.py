"""
Custom exceptions for the stock price predictor
"""

class StockPredictorError(Exception):
    """Base exception for all stock predictor errors"""
    pass

class DataFetchError(StockPredictorError):
    """Raised when data fetching fails"""
    pass

class ValidationError(StockPredictorError):
    """Raised when data validation fails"""
    pass

class ModelError(StockPredictorError):
    """Raised when model operations fail"""
    pass

class TrainingError(ModelError):
    """Raised when model training fails"""
    pass

class PredictionError(ModelError):
    """Raised when prediction fails"""
    pass

class PreprocessingError(StockPredictorError):
    """Raised when data preprocessing fails"""
    pass

class ConfigurationError(StockPredictorError):
    """Raised when configuration is invalid"""
    pass

class BacktestError(StockPredictorError):
    """Raised when backtesting fails"""
    pass

class VisualizationError(StockPredictorError):
    """Raised when visualization generation fails"""
    pass