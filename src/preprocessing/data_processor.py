"""
Data preprocessing and feature engineering for stock price prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass, field
import joblib
import os

from ..utils.exceptions import PreprocessingError, ValidationError
from ..config.config import config

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingResult:
    """Result object for preprocessing operations"""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: Any
    feature_names: List[str]
    sequence_length: int
    train_size: int
    test_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataProcessor:
    """
    Comprehensive data processor for stock price prediction
    """
    
    def __init__(self, 
                 sequence_length: int = None,
                 scaler_type: str = "minmax",
                 train_split: float = None):
        """
        Initialize the data processor
        
        Args:
            sequence_length: Number of time steps to look back
            scaler_type: Type of scaler ('minmax', 'standard', 'robust')
            train_split: Proportion of data for training
        """
        self.sequence_length = sequence_length or config.model.sequence_length
        self.train_split = train_split or config.data.train_split
        self.scaler_type = scaler_type
        
        # Initialize scaler
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
        self.feature_names = []
        
    def prepare_data(self, 
                    data: pd.DataFrame,
                    target_column: str = None,
                    feature_columns: List[str] = None,
                    validation_split: float = None) -> PreprocessingResult:
        """
        Prepare data for LSTM training
        
        Args:
            data: Input DataFrame with stock data
            target_column: Target column name for prediction
            feature_columns: List of feature columns to use
            validation_split: Proportion of training data for validation
            
        Returns:
            PreprocessingResult object with processed data
        """
        try:
            logger.info("Starting data preparation")
            
            # Set defaults
            target_column = target_column or config.data.target
            feature_columns = feature_columns or config.data.features
            validation_split = validation_split or config.model.validation_split
            
            # Validate input data
            self._validate_input_data(data, target_column, feature_columns)
            
            # Clean and prepare the data
            cleaned_data = self._clean_data(data.copy())
            
            # Feature engineering
            engineered_data = self._engineer_features(cleaned_data)
            
            # Select features
            feature_data = self._select_features(engineered_data, feature_columns, target_column)
            
            # Scale the data
            scaled_data = self._scale_data(feature_data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, target_column)
            
            # Split the data
            X_train, X_test, y_train, y_test = self._split_data(X, y, validation_split)
            
            # Create result object
            result = PreprocessingResult(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                scaler=self.scaler,
                feature_names=self.feature_names,
                sequence_length=self.sequence_length,
                train_size=len(X_train),
                test_size=len(X_test),
                metadata={
                    "original_shape": data.shape,
                    "processed_shape": scaled_data.shape,
                    "scaler_type": self.scaler_type,
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                    "train_split": self.train_split,
                    "validation_split": validation_split
                }
            )
            
            logger.info(f"Data preparation completed. Train: {len(X_train)}, Test: {len(X_test)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise PreprocessingError(f"Data preparation failed: {e}")
    
    def _validate_input_data(self, 
                           data: pd.DataFrame, 
                           target_column: str, 
                           feature_columns: List[str]) -> None:
        """Validate input data"""
        if data.empty:
            raise ValidationError("Input data is empty")
        
        if target_column not in data.columns:
            raise ValidationError(f"Target column '{target_column}' not found in data")
        
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValidationError(f"Missing feature columns: {missing_features}")
        
        if len(data) < self.sequence_length + 1:
            raise ValidationError(f"Insufficient data points: {len(data)} (minimum: {self.sequence_length + 1})")
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the input data"""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Sort by index (date)
        data = data.sort_index()
        
        # Handle missing values
        # Forward fill first, then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        # Remove outliers using IQR method for price columns
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                data[col] = data[col].clip(lower_bound, upper_bound)
        
        # Validate price relationships
        if all(col in data.columns for col in price_columns):
            # Ensure High >= Low, Open, Close and Low <= Open, Close
            data['High'] = data[['High', 'Open', 'Close', 'Low']].max(axis=1)
            data['Low'] = data[['Low', 'Open', 'Close', 'High']].min(axis=1)
        
        logger.info(f"Data cleaned. Shape: {data.shape}")
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features"""
        df = data.copy()
        
        try:
            # Price-based features
            if 'Close' in df.columns:
                # Returns
                df['Returns'] = df['Close'].pct_change()
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                
                # Moving averages
                for window in [5, 10, 20, 50]:
                    df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
                    df[f'Close_MA_{window}_Ratio'] = df['Close'] / df[f'MA_{window}']
                
                # Exponential moving averages
                for span in [12, 26]:
                    df[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()
                
                # Price momentum
                for period in [1, 5, 10, 20]:
                    df[f'Price_Change_{period}d'] = df['Close'].pct_change(periods=period)
                    df[f'Price_Momentum_{period}d'] = df['Close'] / df['Close'].shift(period) - 1
            
            # OHLC features
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                # Price ranges
                df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
                df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close']
                df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
                df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
                
                # Gap analysis
                df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
                df['Gap_Up'] = (df['Gap'] > 0).astype(int)
                df['Gap_Down'] = (df['Gap'] < 0).astype(int)
            
            # Volume features
            if 'Volume' in df.columns:
                # Volume moving averages
                for window in [5, 10, 20]:
                    df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
                    df[f'Volume_Ratio_{window}'] = df['Volume'] / df[f'Volume_MA_{window}']
                
                # Price-Volume features
                if 'Close' in df.columns:
                    df['Price_Volume'] = df['Close'] * df['Volume']
                    df['VWAP'] = df['Price_Volume'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
                    df['Volume_Price_Trend'] = df['Volume'] * df['Returns']
            
            # Volatility features
            if 'Returns' in df.columns:
                for window in [5, 10, 20]:
                    df[f'Volatility_{window}d'] = df['Returns'].rolling(window=window).std()
                    df[f'Volatility_Ratio_{window}d'] = df[f'Volatility_{window}d'] / df[f'Volatility_{window}d'].rolling(window=50).mean()
            
            # Technical indicators
            df = self._add_technical_indicators(df)
            
            # Cyclical features (day of week, month, etc.)
            if isinstance(df.index, pd.DatetimeIndex):
                df['DayOfWeek'] = df.index.dayofweek
                df['Month'] = df.index.month
                df['Quarter'] = df.index.quarter
                df['IsMonthEnd'] = df.index.is_month_end.astype(int)
                df['IsMonthStart'] = df.index.is_month_start.astype(int)
            
            # Lag features
            if 'Close' in df.columns:
                for lag in [1, 2, 3, 5, 10]:
                    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
                    if 'Returns' in df.columns:
                        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Drop any new NaN values created by feature engineering
            df = df.dropna()
            
            logger.info(f"Feature engineering completed. New shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise PreprocessingError(f"Feature engineering failed: {e}")
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        df = data.copy()
        
        try:
            if 'Close' in df.columns:
                # RSI
                df['RSI'] = self._calculate_rsi(df['Close'])
                
                # MACD
                exp1 = df['Close'].ewm(span=12).mean()
                exp2 = df['Close'].ewm(span=26).mean()
                df['MACD'] = exp1 - exp2
                df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
                
                # Bollinger Bands
                df['BB_Middle'] = df['Close'].rolling(window=20).mean()
                bb_std = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
                df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
                df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                
                # Stochastic Oscillator
                if all(col in df.columns for col in ['High', 'Low']):
                    lowest_low = df['Low'].rolling(window=14).min()
                    highest_high = df['High'].rolling(window=14).max()
                    df['Stoch_K'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
                    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
                
                # Williams %R
                if all(col in df.columns for col in ['High', 'Low']):
                    highest_high_14 = df['High'].rolling(window=14).max()
                    lowest_low_14 = df['Low'].rolling(window=14).min()
                    df['Williams_R'] = -100 * (highest_high_14 - df['Close']) / (highest_high_14 - lowest_low_14)
                
                # Average True Range (ATR)
                if all(col in df.columns for col in ['High', 'Low']):
                    df['True_Range'] = self._calculate_true_range(df)
                    df['ATR'] = df['True_Range'].rolling(window=14).mean()
                    df['ATR_Ratio'] = df['True_Range'] / df['ATR']
                
                # Commodity Channel Index (CCI)
                if all(col in df.columns for col in ['High', 'Low']):
                    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                    sma_tp = typical_price.rolling(window=20).mean()
                    mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
                    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding some technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high_low = data['High'] - data['Low']
        high_close_prev = abs(data['High'] - data['Close'].shift(1))
        low_close_prev = abs(data['Low'] - data['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range
    
    def _select_features(self, 
                        data: pd.DataFrame, 
                        feature_columns: List[str], 
                        target_column: str) -> pd.DataFrame:
        """Select and validate features for modeling"""
        
        # Start with specified feature columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        # Add engineered features that might be useful
        engineered_features = [col for col in data.columns if any(keyword in col for keyword in [
            'MA_', 'EMA_', 'Returns', 'Volatility', 'RSI', 'MACD', 'BB_', 'Stoch_', 'Williams_R', 
            'ATR', 'CCI', 'Price_Change_', 'Volume_Ratio_', 'Gap', 'Price_Range'
        ])]
        
        # Combine all features
        all_features = list(set(available_features + engineered_features))
        
        # Ensure target column is included
        if target_column not in all_features:
            all_features.append(target_column)
        
        # Select only available columns
        selected_features = [col for col in all_features if col in data.columns]
        
        # Remove features with too many NaN values (>50%)
        selected_features = [col for col in selected_features 
                           if data[col].isna().sum() / len(data) < 0.5]
        
        self.feature_names = [col for col in selected_features if col != target_column]
        
        result_data = data[selected_features].copy()
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        return result_data
    
    def _scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale the data"""
        # Fit the scaler if not already fitted
        if not self.is_fitted:
            self.scaler.fit(data)
            self.is_fitted = True
        
        # Transform the data
        scaled_array = self.scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_array, columns=data.columns, index=data.index)
        
        logger.info(f"Data scaled using {self.scaler_type} scaler")
        return scaled_data
    
    def _create_sequences(self, 
                         data: pd.DataFrame, 
                         target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        # Separate features and target
        feature_columns = [col for col in data.columns if col != target_column]
        features = data[feature_columns].values
        target = data[target_column].values
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            # Features: sequence of past values
            X.append(features[i-self.sequence_length:i])
            # Target: next value
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created sequences. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def _split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray, 
                   validation_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets"""
        
        # Time series split - use last portion for testing
        split_idx = int(len(X) * self.train_split)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Data split completed. Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_predictions(self, 
                                    predictions: np.ndarray, 
                                    feature_data: pd.DataFrame,
                                    target_column: str) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale
        
        Args:
            predictions: Scaled predictions
            feature_data: Original feature data for inverse scaling
            target_column: Name of target column
            
        Returns:
            Predictions in original scale
        """
        try:
            # Create a dummy array with the same structure as training data
            dummy_data = np.zeros((len(predictions), len(feature_data.columns)))
            
            # Find target column index
            target_idx = feature_data.columns.get_loc(target_column)
            
            # Place predictions in the target column
            dummy_data[:, target_idx] = predictions.flatten()
            
            # Inverse transform
            inverse_transformed = self.scaler.inverse_transform(dummy_data)
            
            # Extract only the target column
            return inverse_transformed[:, target_idx]
            
        except Exception as e:
            logger.error(f"Error in inverse transform: {e}")
            raise PreprocessingError(f"Inverse transform failed: {e}")
    
    def save_processor(self, filepath: str) -> None:
        """Save the processor state"""
        try:
            processor_state = {
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'train_split': self.train_split,
                'scaler_type': self.scaler_type,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(processor_state, filepath)
            logger.info(f"Processor saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving processor: {e}")
            raise PreprocessingError(f"Failed to save processor: {e}")
    
    def load_processor(self, filepath: str) -> None:
        """Load the processor state"""
        try:
            processor_state = joblib.load(filepath)
            
            self.scaler = processor_state['scaler']
            self.sequence_length = processor_state['sequence_length']
            self.train_split = processor_state['train_split']
            self.scaler_type = processor_state['scaler_type']
            self.feature_names = processor_state['feature_names']
            self.is_fitted = processor_state['is_fitted']
            
            logger.info(f"Processor loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading processor: {e}")
            raise PreprocessingError(f"Failed to load processor: {e}")