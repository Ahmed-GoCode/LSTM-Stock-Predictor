"""
Yahoo Finance API implementation for stock data fetching
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Data structure for stock information"""
    symbol: str
    data: pd.DataFrame
    info: Dict
    period: str
    interval: str
    last_updated: datetime

class YahooFinanceAPI:
    """
    Yahoo Finance API wrapper for fetching stock data
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance API wrapper"""
        self.session = None
        
    def get_stock_data(
        self, 
        symbol: str, 
        period: str = "2y", 
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> StockData:
        """
        Fetch stock data for a given symbol
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            
        Returns:
            StockData object containing the fetched data
            
        Raises:
            ValueError: If symbol is invalid or no data is found
            ConnectionError: If there's an issue with the API connection
        """
        try:
            logger.info(f"Fetching data for {symbol} with period={period}, interval={interval}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            if start and end:
                hist_data = ticker.history(start=start, end=end, interval=interval)
            else:
                hist_data = ticker.history(period=period, interval=interval)
            
            if hist_data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Get stock info
            try:
                info = ticker.info
            except Exception as e:
                logger.warning(f"Could not fetch info for {symbol}: {e}")
                info = {"symbol": symbol}
            
            # Clean and validate data
            hist_data = self._clean_data(hist_data)
            
            # Create StockData object
            stock_data = StockData(
                symbol=symbol.upper(),
                data=hist_data,
                info=info,
                period=period,
                interval=interval,
                last_updated=datetime.now()
            )
            
            logger.info(f"Successfully fetched {len(hist_data)} records for {symbol}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise ConnectionError(f"Failed to fetch data for {symbol}: {e}")
    
    def get_multiple_stocks(
        self, 
        symbols: List[str], 
        period: str = "2y", 
        interval: str = "1d"
    ) -> Dict[str, StockData]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock ticker symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to their StockData objects
        """
        results = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                results[symbol] = self.get_stock_data(symbol, period, interval)
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for: {failed_symbols}")
        
        return results
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def search_stocks(self, query: str) -> List[Dict]:
        """
        Search for stocks by name or symbol
        
        Args:
            query: Search query
            
        Returns:
            List of matching stocks
        """
        try:
            # This is a simplified search - in a real implementation,
            # you might want to use a proper search API
            ticker = yf.Ticker(query)
            info = ticker.info
            
            if info.get('symbol'):
                return [info]
            else:
                return []
        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return []
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the fetched data
        
        Args:
            data: Raw data from Yahoo Finance
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill missing values
        data = data.fillna(method='ffill')
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        # Ensure all required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                logger.warning(f"Missing column: {col}")
        
        # Add technical indicators columns (will be filled by preprocessing)
        data['Returns'] = data['Close'].pct_change()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate volatility
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        return data
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return 'symbol' in info or 'shortName' in info
        except Exception:
            return False
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price of a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Current price or None if not available
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_market_status(self) -> Dict[str, str]:
        """
        Get market status information
        
        Returns:
            Dictionary containing market status information
        """
        # This is a placeholder - in a real implementation,
        # you might want to use a dedicated market status API
        now = datetime.now()
        hour = now.hour
        
        if 9 <= hour < 16:  # Simplified market hours
            status = "open"
        else:
            status = "closed"
        
        return {
            "status": status,
            "timestamp": now.isoformat(),
            "timezone": "US/Eastern"
        }