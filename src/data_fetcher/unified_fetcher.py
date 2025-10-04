"""
Unified data fetcher that combines multiple data sources
"""

import pandas as pd
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .yahoo_finance import YahooFinanceAPI, StockData
from .alpha_vantage import AlphaVantageAPI, AlphaVantageData
from ..utils.exceptions import DataFetchError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class UnifiedStockData:
    """Unified data structure combining multiple sources"""
    symbol: str
    primary_data: pd.DataFrame
    supplementary_data: Dict[str, pd.DataFrame]
    metadata: Dict
    sources: List[str]
    last_updated: datetime

class UnifiedDataFetcher:
    """
    Unified data fetcher that combines Yahoo Finance and Alpha Vantage APIs
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize the unified data fetcher
        
        Args:
            alpha_vantage_key: Alpha Vantage API key (optional)
        """
        self.yahoo_api = YahooFinanceAPI()
        self.alpha_vantage_api = AlphaVantageAPI(alpha_vantage_key) if alpha_vantage_key else None
        
    def fetch_stock_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        include_technical_indicators: bool = True,
        include_fundamentals: bool = False
    ) -> UnifiedStockData:
        """
        Fetch comprehensive stock data from multiple sources
        
        Args:
            symbol: Stock ticker symbol
            period: Data period for primary data
            interval: Data interval for primary data
            include_technical_indicators: Whether to fetch technical indicators
            include_fundamentals: Whether to fetch fundamental data
            
        Returns:
            UnifiedStockData object with combined data
        """
        sources_used = []
        supplementary_data = {}
        metadata = {"symbol": symbol, "fetched_at": datetime.now().isoformat()}
        
        try:
            # Primary data from Yahoo Finance
            logger.info(f"Fetching primary data for {symbol} from Yahoo Finance")
            yahoo_data = self.yahoo_api.get_stock_data(symbol, period, interval)
            primary_data = yahoo_data.data.copy()
            sources_used.append("yahoo_finance")
            
            # Add Yahoo Finance metadata
            metadata.update({
                "yahoo_info": yahoo_data.info,
                "period": period,
                "interval": interval
            })
            
        except Exception as e:
            logger.error(f"Failed to fetch primary data from Yahoo Finance: {e}")
            raise DataFetchError(f"Failed to fetch data for {symbol}: {e}")
        
        # Technical indicators from Alpha Vantage (if available)
        if include_technical_indicators and self.alpha_vantage_api:
            try:
                logger.info(f"Fetching technical indicators for {symbol}")
                technical_data = self._fetch_technical_indicators(symbol)
                supplementary_data.update(technical_data)
                sources_used.append("alpha_vantage_technical")
                
            except Exception as e:
                logger.warning(f"Failed to fetch technical indicators: {e}")
        
        # Fundamental data from Alpha Vantage (if available)
        if include_fundamentals and self.alpha_vantage_api:
            try:
                logger.info(f"Fetching fundamental data for {symbol}")
                fundamental_data = self._fetch_fundamental_data(symbol)
                metadata.update({"fundamentals": fundamental_data})
                sources_used.append("alpha_vantage_fundamentals")
                
            except Exception as e:
                logger.warning(f"Failed to fetch fundamental data: {e}")
        
        # Enhance primary data with calculated indicators
        primary_data = self._add_calculated_indicators(primary_data)
        
        return UnifiedStockData(
            symbol=symbol,
            primary_data=primary_data,
            supplementary_data=supplementary_data,
            metadata=metadata,
            sources=sources_used,
            last_updated=datetime.now()
        )
    
    def _fetch_technical_indicators(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch various technical indicators from Alpha Vantage
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary of technical indicator DataFrames
        """
        indicators = {}
        
        # List of indicators to fetch
        indicator_configs = [
            {"indicator": "SMA", "time_period": 20, "name": "SMA_20"},
            {"indicator": "SMA", "time_period": 50, "name": "SMA_50"},
            {"indicator": "EMA", "time_period": 12, "name": "EMA_12"},
            {"indicator": "EMA", "time_period": 26, "name": "EMA_26"},
            {"indicator": "RSI", "time_period": 14, "name": "RSI_14"},
            {"indicator": "MACD", "name": "MACD"},
            {"indicator": "BBANDS", "time_period": 20, "name": "BBANDS_20"},
        ]
        
        for config in indicator_configs:
            try:
                if config["indicator"] == "MACD":
                    # MACD has different parameters
                    data = self.alpha_vantage_api.get_technical_indicators(
                        symbol, "MACD", interval="daily",
                        fastperiod=12, slowperiod=26, signalperiod=9
                    )
                elif config["indicator"] == "BBANDS":
                    # Bollinger Bands
                    data = self.alpha_vantage_api.get_technical_indicators(
                        symbol, "BBANDS", interval="daily",
                        time_period=config["time_period"],
                        nbdevup=2, nbdevdn=2
                    )
                else:
                    data = self.alpha_vantage_api.get_technical_indicators(
                        symbol, config["indicator"], interval="daily",
                        time_period=config["time_period"]
                    )
                
                indicators[config["name"]] = data.data
                
            except Exception as e:
                logger.warning(f"Failed to fetch {config['name']}: {e}")
                continue
        
        return indicators
    
    def _fetch_fundamental_data(self, symbol: str) -> Dict:
        """
        Fetch fundamental data from Alpha Vantage
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing fundamental data
        """
        fundamental_data = {}
        
        try:
            # Company overview
            overview = self.alpha_vantage_api.get_company_overview(symbol)
            fundamental_data["overview"] = overview
            
        except Exception as e:
            logger.warning(f"Failed to fetch company overview: {e}")
        
        try:
            # Earnings data
            earnings = self.alpha_vantage_api.get_earnings(symbol)
            fundamental_data["earnings"] = earnings
            
        except Exception as e:
            logger.warning(f"Failed to fetch earnings data: {e}")
        
        return fundamental_data
    
    def _add_calculated_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated technical indicators to the data
        
        Args:
            data: Stock price DataFrame
            
        Returns:
            Enhanced DataFrame with technical indicators
        """
        df = data.copy()
        
        try:
            # Price-based indicators
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            df['RSI'] = self._calculate_rsi(df['Close'])
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Price patterns
            df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
            df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
            
            # Volatility
            df['True_Range'] = self._calculate_true_range(df)
            df['ATR'] = df['True_Range'].rolling(window=14).mean()
            
            # Support and Resistance levels
            df['Resistance'] = df['High'].rolling(window=20).max()
            df['Support'] = df['Low'].rolling(window=20).min()
            
        except Exception as e:
            logger.warning(f"Error calculating some indicators: {e}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range for ATR calculation"""
        high_low = data['High'] - data['Low']
        high_close_prev = abs(data['High'] - data['Close'].shift(1))
        low_close_prev = abs(data['Low'] - data['Close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range
    
    def validate_data(self, data: UnifiedStockData) -> bool:
        """
        Validate the fetched data for completeness and quality
        
        Args:
            data: UnifiedStockData object to validate
            
        Returns:
            True if data is valid, False otherwise
            
        Raises:
            ValidationError: If critical validation fails
        """
        try:
            # Check if we have basic required data
            if data.primary_data.empty:
                raise ValidationError("Primary data is empty")
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.primary_data.columns]
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")
            
            # Check for sufficient data points
            min_required_points = 50  # Minimum for meaningful analysis
            if len(data.primary_data) < min_required_points:
                logger.warning(f"Limited data points: {len(data.primary_data)} (minimum recommended: {min_required_points})")
            
            # Check for data quality issues
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if data.primary_data[col].isna().sum() > len(data.primary_data) * 0.1:  # More than 10% missing
                    logger.warning(f"High percentage of missing values in {col}")
                
                if (data.primary_data[col] <= 0).sum() > 0 and col != 'Volume':  # Negative prices
                    logger.warning(f"Found non-positive values in {col}")
            
            # Validate price relationships
            invalid_prices = data.primary_data[
                (data.primary_data['High'] < data.primary_data['Low']) |
                (data.primary_data['High'] < data.primary_data['Open']) |
                (data.primary_data['High'] < data.primary_data['Close']) |
                (data.primary_data['Low'] > data.primary_data['Open']) |
                (data.primary_data['Low'] > data.primary_data['Close'])
            ]
            
            if len(invalid_prices) > 0:
                logger.warning(f"Found {len(invalid_prices)} rows with invalid price relationships")
            
            logger.info(f"Data validation completed for {data.symbol}")
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def get_available_symbols(self, exchange: str = "US") -> List[str]:
        """
        Get list of available symbols for a given exchange
        
        Args:
            exchange: Exchange identifier
            
        Returns:
            List of available symbols
        """
        # This would typically come from a dedicated API or database
        # For now, return a sample list of popular US stocks
        popular_symbols = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'SHOP', 'SQ',
            'JPM', 'BAC', 'WFC', 'GS', 'C', 'V', 'MA', 'AXP',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'CVS', 'TMO', 'DHR',
            'SPY', 'QQQ', 'IWM', 'EFA', 'VTI', 'VOO', 'ARKK', 'XLF'
        ]
        
        return popular_symbols