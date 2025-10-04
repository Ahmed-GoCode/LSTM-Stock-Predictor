"""
Alpha Vantage API implementation for additional stock data
"""

import requests
import pandas as pd
import json
from typing import Optional, Dict, List
from datetime import datetime
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class AlphaVantageData:
    """Data structure for Alpha Vantage stock information"""
    symbol: str
    data: pd.DataFrame
    metadata: Dict
    api_function: str
    last_updated: datetime

class AlphaVantageAPI:
    """
    Alpha Vantage API wrapper for fetching stock data and additional metrics
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Alpha Vantage API wrapper
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.last_request_time = 0
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 5 requests per minute
        
    def _make_request(self, params: Dict) -> Dict:
        """
        Make a rate-limited request to Alpha Vantage API
        
        Args:
            params: Request parameters
            
        Returns:
            JSON response
            
        Raises:
            ConnectionError: If API request fails
            ValueError: If API key is missing or invalid
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        params['apikey'] = self.api_key
        
        try:
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            self.last_request_time = time.time()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
            
            if "Note" in data:
                raise ConnectionError(f"Alpha Vantage API Rate Limit: {data['Note']}")
            
            return data
            
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to Alpha Vantage API: {e}")
    
    def get_intraday_data(
        self, 
        symbol: str, 
        interval: str = "15min", 
        outputsize: str = "compact"
    ) -> AlphaVantageData:
        """
        Get intraday stock data
        
        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            outputsize: Data size (compact or full)
            
        Returns:
            AlphaVantageData object
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize
        }
        
        data = self._make_request(params)
        
        # Parse the response
        time_series_key = f'Time Series ({interval})'
        if time_series_key not in data:
            raise ValueError(f"No intraday data found for {symbol}")
        
        time_series = data[time_series_key]
        metadata = data.get('Meta Data', {})
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Clean column names and convert to numeric
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col])
        df['Volume'] = pd.to_numeric(df['Volume'])
        
        return AlphaVantageData(
            symbol=symbol,
            data=df,
            metadata=metadata,
            api_function='TIME_SERIES_INTRADAY',
            last_updated=datetime.now()
        )
    
    def get_daily_data(
        self, 
        symbol: str, 
        outputsize: str = "compact"
    ) -> AlphaVantageData:
        """
        Get daily stock data
        
        Args:
            symbol: Stock ticker symbol
            outputsize: Data size (compact or full)
            
        Returns:
            AlphaVantageData object
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize
        }
        
        data = self._make_request(params)
        
        time_series = data.get('Time Series (Daily)', {})
        metadata = data.get('Meta Data', {})
        
        if not time_series:
            raise ValueError(f"No daily data found for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Clean column names and convert to numeric
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col])
        df['Volume'] = pd.to_numeric(df['Volume'])
        
        return AlphaVantageData(
            symbol=symbol,
            data=df,
            metadata=metadata,
            api_function='TIME_SERIES_DAILY',
            last_updated=datetime.now()
        )
    
    def get_technical_indicators(
        self, 
        symbol: str, 
        indicator: str, 
        interval: str = "daily",
        time_period: int = 20,
        **kwargs
    ) -> AlphaVantageData:
        """
        Get technical indicators for a stock
        
        Args:
            symbol: Stock ticker symbol
            indicator: Technical indicator (SMA, EMA, RSI, MACD, etc.)
            interval: Time interval
            time_period: Period for the indicator
            **kwargs: Additional parameters for the indicator
            
        Returns:
            AlphaVantageData object
        """
        params = {
            'function': indicator.upper(),
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            **kwargs
        }
        
        data = self._make_request(params)
        
        # The key varies by indicator
        tech_indicator_key = None
        for key in data.keys():
            if 'Technical Analysis' in key:
                tech_indicator_key = key
                break
        
        if not tech_indicator_key:
            raise ValueError(f"No technical indicator data found for {symbol}")
        
        time_series = data[tech_indicator_key]
        metadata = data.get('Meta Data', {})
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        return AlphaVantageData(
            symbol=symbol,
            data=df,
            metadata=metadata,
            api_function=indicator.upper(),
            last_updated=datetime.now()
        )
    
    def get_company_overview(self, symbol: str) -> Dict:
        """
        Get company overview and fundamentals
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing company information
        """
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        return self._make_request(params)
    
    def get_earnings(self, symbol: str) -> Dict:
        """
        Get earnings data for a company
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing earnings information
        """
        params = {
            'function': 'EARNINGS',
            'symbol': symbol
        }
        
        return self._make_request(params)
    
    def search_symbols(self, keywords: str) -> List[Dict]:
        """
        Search for symbols using keywords
        
        Args:
            keywords: Search keywords
            
        Returns:
            List of matching symbols
        """
        params = {
            'function': 'SYMBOL_SEARCH',
            'keywords': keywords
        }
        
        data = self._make_request(params)
        return data.get('bestMatches', [])
    
    def get_forex_rate(self, from_currency: str, to_currency: str) -> Dict:
        """
        Get real-time forex exchange rate
        
        Args:
            from_currency: From currency code (e.g., 'USD')
            to_currency: To currency code (e.g., 'EUR')
            
        Returns:
            Dictionary containing exchange rate information
        """
        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_currency,
            'to_currency': to_currency
        }
        
        data = self._make_request(params)
        return data.get('Realtime Currency Exchange Rate', {})
    
    def get_crypto_data(
        self, 
        symbol: str, 
        market: str = "USD"
    ) -> AlphaVantageData:
        """
        Get cryptocurrency data
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            market: Market currency (e.g., 'USD')
            
        Returns:
            AlphaVantageData object
        """
        params = {
            'function': 'DIGITAL_CURRENCY_DAILY',
            'symbol': symbol,
            'market': market
        }
        
        data = self._make_request(params)
        
        time_series_key = f'Time Series (Digital Currency Daily)'
        if time_series_key not in data:
            raise ValueError(f"No crypto data found for {symbol}")
        
        time_series = data[time_series_key]
        metadata = data.get('Meta Data', {})
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Select main price columns
        main_cols = [col for col in df.columns if f'({market})' in col and 'volume' not in col.lower()]
        df = df[main_cols]
        
        # Rename columns for consistency
        df.columns = ['Open', 'High', 'Low', 'Close']
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        return AlphaVantageData(
            symbol=symbol,
            data=df,
            metadata=metadata,
            api_function='DIGITAL_CURRENCY_DAILY',
            last_updated=datetime.now()
        )