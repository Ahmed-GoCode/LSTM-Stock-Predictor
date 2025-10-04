# Stock Price Predictor Configuration

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Configuration for LSTM model"""
    sequence_length: int = 60  # Number of days to look back
    lstm_units: List[int] = None  # LSTM layer units
    dropout_rate: float = 0.2
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    learning_rate: float = 0.001
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [50, 50]

@dataclass
class DataConfig:
    """Configuration for data fetching and preprocessing"""
    default_period: str = "2y"  # Default data period
    default_interval: str = "1d"  # Default data interval
    train_split: float = 0.8
    features: List[str] = None  # Features to use for prediction
    target: str = "Close"  # Target variable
    
    def __post_init__(self):
        if self.features is None:
            self.features = ["Open", "High", "Low", "Close", "Volume"]

@dataclass
class APIConfig:
    """Configuration for API keys and endpoints"""
    alpha_vantage_key: Optional[str] = None
    
    def __post_init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

@dataclass
class AppConfig:
    """Main application configuration"""
    model: ModelConfig = None
    data: DataConfig = None
    api: APIConfig = None
    
    # File paths
    models_dir: str = "models"
    data_dir: str = "data" 
    outputs_dir: str = "outputs"
    
    # Visualization
    plot_style: str = "plotly_dark"
    figure_size: tuple = (12, 8)
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.api is None:
            self.api = APIConfig()

# Default configuration instance
config = AppConfig()