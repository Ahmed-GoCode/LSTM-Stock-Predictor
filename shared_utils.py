"""
Shared utilities for LSTM Trading Platform
All the common stuff we use across different pages
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Different types of assets we can trade
ASSETS = {
    "🏢 Stocks": {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "TSLA": "Tesla Inc.",
        "MSFT": "Microsoft Corp.",
        "NVDA": "NVIDIA Corp.",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms",
        "JPM": "JPMorgan Chase",
        "V": "Visa Inc.",
        "JNJ": "Johnson & Johnson"
    },
    "🥇 Metals": {
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures", 
        "PL=F": "Platinum Futures",
        "HG=F": "Copper Futures",
        "PA=F": "Palladium Futures"
    },
    "📊 Indices": {
        "^DJI": "Dow Jones Industrial",
        "^IXIC": "NASDAQ Composite",
        "^GSPC": "S&P 500",
        "^RUT": "Russell 2000",
        "^VIX": "VIX Volatility Index"
    },
    "💱 Currency": {
        "DX-Y.NYB": "US Dollar Index",
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY",
        "GBPJPY=X": "GBP/JPY"
    },
    "₿ Crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "ADA-USD": "Cardano",
        "SOL-USD": "Solana",
        "DOT-USD": "Polkadot"
    }
}

@st.cache_data(ttl=300)  # Keep data fresh for 5 minutes
def fetch_real_data(symbol, period="1y", interval="1d"):
    """Get stock data from Yahoo Finance - handles errors gracefully"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Time periods we support (from 1 month to 5+ years)
        period_map = {
            "1mo": "1mo", "3mo": "3mo", "6mo": "6mo", "1y": "1y", 
            "2y": "2y", "5y": "5y", "ytd": "ytd", "max": "max"
        }
        
        # If user enters something weird, try to fix it
        if period not in period_map.values():
            if "mo" in period or "month" in period.lower():
                period = "6mo"
            elif "y" in period or "year" in period.lower():
                period = "2y"
            else:
                period = "1y"
        
        # Try different periods if the first one fails
        periods_to_try = [period, "1y", "2y", "5y", "max"]
        
        for p in periods_to_try:
            try:
                data = ticker.history(period=p, interval=interval)
                if not data.empty and len(data) >= 20:  # Need at least 20 days
                    # Get some extra info about the stock
                    info = ticker.info
                    data.attrs['symbol'] = symbol
                    data.attrs['company_name'] = info.get('longName', symbol)
                    data.attrs['currency'] = info.get('currency', 'USD')
                    data.attrs['market_cap'] = info.get('marketCap', 'N/A')
                    break
            except Exception as e:
                continue
        else:
            # If everything fails, try different intervals
            for intv in ["1d", "5d", "1wk"]:
                try:
                    data = ticker.history(period="1y", interval=intv)
                    if not data.empty and len(data) >= 10:
                        break
                except:
                    continue
            else:
                st.error(f"❌ No data available for {symbol}")
                return None
            
        # Clean up column names
        data.columns = data.columns.str.strip()
        
        # Tell user how much data we got
        if len(data) >= 250:  # 1+ years
            st.success(f"✅ Great! Got {len(data)} days for {symbol}")
        elif len(data) >= 90:  # 3+ months
            st.info(f"ℹ️ Good data: {len(data)} days for {symbol}")
        elif len(data) >= 30:  # 1+ month
            st.warning(f"⚠️ Limited data: {len(data)} days for {symbol}")
        else:
            st.error(f"❌ Not enough data: only {len(data)} days for {symbol}")
            
        return data
        
    except Exception as e:
        st.error(f"❌ Error getting data for {symbol}: {str(e)}")
        return None

# Trading signals based on technical analysis
@st.cache_data(ttl=60)  # Cache for 1 minute for real-time feel
def get_signals(symbol, data):
    """Generate real-time trading signals based on technical analysis"""
    try:
        if data is None or len(data) < 50:
            return {"error": "Insufficient data for signal generation"}
        
        # Calculate technical indicators for signals
        data_with_indicators = calc_indicators(data)
        
        signals = []
        confidence_scores = []
        
        # Get recent data for signal calculation
        recent_data = data_with_indicators.tail(20)
        current_price = data['Close'].iloc[-1]
        
        # 1. RSI Signals
        if 'RSI' in data_with_indicators.columns:
            current_rsi = data_with_indicators['RSI'].iloc[-1]
            if current_rsi < 30:
                signals.append({"type": "BUY", "indicator": "RSI", "reason": f"Oversold (RSI: {current_rsi:.1f})", "strength": "Strong"})
                confidence_scores.append(0.8)
            elif current_rsi > 70:
                signals.append({"type": "SELL", "indicator": "RSI", "reason": f"Overbought (RSI: {current_rsi:.1f})", "strength": "Strong"})
                confidence_scores.append(0.8)
        
        # 2. MACD Signals
        if 'MACD' in data_with_indicators.columns and 'MACD_Signal' in data_with_indicators.columns:
            macd = data_with_indicators['MACD'].iloc[-1]
            macd_signal = data_with_indicators['MACD_Signal'].iloc[-1]
            prev_macd = data_with_indicators['MACD'].iloc[-2]
            prev_macd_signal = data_with_indicators['MACD_Signal'].iloc[-2]
            
            # MACD crossover
            if macd > macd_signal and prev_macd <= prev_macd_signal:
                signals.append({"type": "BUY", "indicator": "MACD", "reason": "Bullish crossover", "strength": "Medium"})
                confidence_scores.append(0.6)
            elif macd < macd_signal and prev_macd >= prev_macd_signal:
                signals.append({"type": "SELL", "indicator": "MACD", "reason": "Bearish crossover", "strength": "Medium"})
                confidence_scores.append(0.6)
        
        # 3. Moving Average Signals
        if 'MA_20' in data_with_indicators.columns and 'MA_50' in data_with_indicators.columns:
            ma_20 = data_with_indicators['MA_20'].iloc[-1]
            ma_50 = data_with_indicators['MA_50'].iloc[-1]
            prev_ma_20 = data_with_indicators['MA_20'].iloc[-2]
            prev_ma_50 = data_with_indicators['MA_50'].iloc[-2]
            
            # Golden Cross / Death Cross
            if ma_20 > ma_50 and prev_ma_20 <= prev_ma_50:
                signals.append({"type": "BUY", "indicator": "MA Cross", "reason": "Golden Cross (20>50 MA)", "strength": "Strong"})
                confidence_scores.append(0.7)
            elif ma_20 < ma_50 and prev_ma_20 >= prev_ma_50:
                signals.append({"type": "SELL", "indicator": "MA Cross", "reason": "Death Cross (20<50 MA)", "strength": "Strong"})
                confidence_scores.append(0.7)
            
            # Price vs MA signals
            if current_price > ma_20 > ma_50:
                signals.append({"type": "BUY", "indicator": "MA Trend", "reason": "Uptrend confirmed", "strength": "Medium"})
                confidence_scores.append(0.5)
            elif current_price < ma_20 < ma_50:
                signals.append({"type": "SELL", "indicator": "MA Trend", "reason": "Downtrend confirmed", "strength": "Medium"})
                confidence_scores.append(0.5)
        
        # 4. Bollinger Bands Signals
        if 'BB_Upper' in data_with_indicators.columns and 'BB_Lower' in data_with_indicators.columns:
            bb_upper = data_with_indicators['BB_Upper'].iloc[-1]
            bb_lower = data_with_indicators['BB_Lower'].iloc[-1]
            
            if current_price <= bb_lower:
                signals.append({"type": "BUY", "indicator": "Bollinger Bands", "reason": "Price at lower band", "strength": "Medium"})
                confidence_scores.append(0.6)
            elif current_price >= bb_upper:
                signals.append({"type": "SELL", "indicator": "Bollinger Bands", "reason": "Price at upper band", "strength": "Medium"})
                confidence_scores.append(0.6)
        
        # 5. Volume Analysis
        volume_ma = data['Volume'].rolling(20).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        
        if current_volume > volume_ma * 1.5:
            # High volume can confirm other signals
            volume_boost = 0.1
            for i in range(len(confidence_scores)):
                confidence_scores[i] = min(1.0, confidence_scores[i] + volume_boost)
            
            signals.append({"type": "INFO", "indicator": "Volume", "reason": f"High volume spike ({current_volume/volume_ma:.1f}x avg)", "strength": "High"})
        
        # Overall signal strength
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            overall_sentiment = "BULLISH" if sum(1 for s in signals if s["type"] == "BUY") > sum(1 for s in signals if s["type"] == "SELL") else "BEARISH"
        else:
            avg_confidence = 0.5
            overall_sentiment = "NEUTRAL"
        
        return {
            "signals": signals,
            "overall_sentiment": overall_sentiment,
            "confidence": avg_confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": current_price
        }
        
    except Exception as e:
        return {"error": f"Signal generation error: {str(e)}"}

@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_fear_greed():
    """Simulate Fear & Greed Index (in real app, would use CNN API)"""
    import random
    
    # Simulate fear & greed index
    index_value = random.randint(0, 100)
    
    if index_value <= 25:
        sentiment = "Extreme Fear"
        color = "#ff4444"
        recommendation = "Strong Buy Opportunity"
    elif index_value <= 45:
        sentiment = "Fear"
        color = "#ff8800"
        recommendation = "Buy Opportunity"
    elif index_value <= 55:
        sentiment = "Neutral"
        color = "#ffdd00"
        recommendation = "Hold/Monitor"
    elif index_value <= 75:
        sentiment = "Greed"
        color = "#88ff00"
        recommendation = "Consider Taking Profits"
    else:
        sentiment = "Extreme Greed"
        color = "#44ff44"
        recommendation = "Sell/Take Profits"
    
    return {
        "index": index_value,
        "sentiment": sentiment,
        "color": color,
        "recommendation": recommendation,
        "last_updated": datetime.now().strftime("%H:%M:%S")
    }

def calc_indicators(data):
    """Calculate comprehensive technical indicators"""
    try:
        if data is None or data.empty:
            return data
            
        df = data.copy()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_period = 20
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error calculating indicators: {str(e)}")
        return data

def make_prediction(data, days_ahead=30):
    """Generate AI predictions using LSTM neural network"""
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        if data is None or len(data) < 60:
            return None, "❌ Insufficient data for prediction (need at least 60 days)"
        
        # Prepare data
        prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)
        
        # Create training data
        sequence_length = 60
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        if len(X) == 0:
            return None, "❌ Not enough data for training sequences"
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        with st.spinner("🧠 Training LSTM model..."):
            model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Make predictions
        last_sequence = scaled_data[-sequence_length:]
        predictions = []
        
        for _ in range(days_ahead):
            pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
            predictions.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred.reshape(1, 1), axis=0)
        
        # Transform back to original scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions.flatten(), "✅ Prediction generated successfully"
        
    except Exception as e:
        return None, f"❌ Error in AI prediction: {str(e)}"

def make_chart(data, symbol, show_volume=True, show_indicators=True):
    """Create advanced interactive chart with technical indicators"""
    try:
        if data is None or data.empty:
            return None
            
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[f'{symbol} Price Chart', 'Technical Indicators', 'Volume'],
                row_width=[0.2, 0.2, 0.1]
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[f'{symbol} Price Chart', 'Technical Indicators'],
                row_width=[0.2, 0.2]
            )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#00d4aa',
                decreasing_line_color='#ff6b6b'
            ),
            row=1, col=1
        )
        
        if show_indicators and 'SMA_20' in data.columns:
            # Moving averages
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#ffa500', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#ff69b4', width=1)
                ),
                row=1, col=1
            )
            
            # Bollinger Bands
            if 'BB_Upper' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                        fill=None
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='rgba(173, 216, 230, 0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(173, 216, 230, 0.1)'
                    ),
                    row=1, col=1
                )
            
            # RSI
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#9370db', width=2)
                    ),
                    row=2, col=1
                )
                
                # RSI overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Volume chart
        if show_volume:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Advanced Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark',
            font=dict(color='white'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating chart: {str(e)}")
        return None

@st.cache_resource
def setup_db():
    """Initialize SQLite database for data storage"""
    conn = sqlite3.connect('trading_platform.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            quantity REAL,
            avg_price REAL,
            date_added TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            condition_type TEXT NOT NULL,
            target_value REAL NOT NULL,
            current_value REAL,
            status TEXT DEFAULT 'active',
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            triggered_date TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            date TEXT,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER
        )
    ''')
    
    # Create data management tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            data_type TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            data_json TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            api_key TEXT,
            is_active BOOLEAN DEFAULT 1,
            rate_limit INTEGER DEFAULT 1000,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn

def get_connection():
    """Get database connection"""
    return sqlite3.connect('trading_platform.db', check_same_thread=False)

# Chart Export Functions
def export_chart_as_html(fig, filename):
    """Export Plotly chart as HTML"""
    import io
    
    html_string = fig.to_html(include_plotlyjs='cdn')
    
    # Create download button
    st.download_button(
        label="📄 Download HTML",
        data=html_string,
        file_name=f"{filename}.html",
        mime="text/html"
    )

def export_chart_as_image(fig, filename, format_type="png"):
    """Export chart as PNG, JPEG, or SVG"""
    import io
    
    try:
        # Convert figure to image bytes
        img_bytes = fig.to_image(format=format_type, width=1200, height=800, scale=2)
        
        # Create download button
        st.download_button(
            label=f"🖼️ Download {format_type.upper()}",
            data=img_bytes,
            file_name=f"{filename}.{format_type}",
            mime=f"image/{format_type}"
        )
    except Exception as e:
        st.error(f"Error exporting as {format_type}: {str(e)}")
        st.info("Note: Image export requires kaleido package. Install with: pip install kaleido")

def export_data_as_csv(data, filename):
    """Export DataFrame as CSV"""
    import io
    
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=True)
    csv_string = csv_buffer.getvalue()
    
    st.download_button(
        label="📊 Download CSV",
        data=csv_string,
        file_name=f"{filename}.csv",
        mime="text/csv"
    )

def export_data_as_excel(data, filename):
    """Export DataFrame as Excel"""
    import io
    
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Data', index=True)
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="📗 Download Excel",
            data=excel_data,
            file_name=f"{filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error exporting Excel: {str(e)}")
        st.info("Note: Excel export requires openpyxl package. Install with: pip install openpyxl")

# AI Pattern Detection Functions
def detect_patterns(data):
    """Detect common trading patterns"""
    patterns = {
        'Support_Resistance': detect_support_resistance(data),
        'Trend_Direction': detect_trend(data),
        'Chart_Patterns': detect_chart_patterns(data),
        'Volume_Analysis': analyze_volume_patterns(data)
    }
    return patterns

def detect_support_resistance(data):
    """Detect support and resistance levels"""
    try:
        if len(data) < 50:
            return {"message": "Insufficient data for support/resistance analysis"}
        
        # Calculate local minima and maxima
        high_prices = data['High'].values
        low_prices = data['Low'].values
        
        # Find support levels (local minima)
        support_levels = []
        for i in range(10, len(low_prices) - 10):
            if all(low_prices[i] <= low_prices[i-j] for j in range(1, 11)) and \
               all(low_prices[i] <= low_prices[i+j] for j in range(1, 11)):
                support_levels.append(low_prices[i])
        
        # Find resistance levels (local maxima)
        resistance_levels = []
        for i in range(10, len(high_prices) - 10):
            if all(high_prices[i] >= high_prices[i-j] for j in range(1, 11)) and \
               all(high_prices[i] >= high_prices[i+j] for j in range(1, 11)):
                resistance_levels.append(high_prices[i])
        
        return {
            'support_levels': support_levels[-3:] if support_levels else [],
            'resistance_levels': resistance_levels[-3:] if resistance_levels else [],
            'strength': 'High' if len(support_levels) > 2 and len(resistance_levels) > 2 else 'Medium'
        }
    except Exception as e:
        return {"error": str(e)}

def detect_trend(data):
    """Detect overall trend direction"""
    try:
        if len(data) < 20:
            return {"message": "Insufficient data for trend analysis"}
        
        # Calculate moving averages
        short_ma = data['Close'].rolling(window=10).mean()
        long_ma = data['Close'].rolling(window=20).mean()
        
        # Current trend
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        
        # Trend strength
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100
        
        if current_short > current_long and price_change > 2:
            trend = "Strong Uptrend"
        elif current_short > current_long and price_change > 0:
            trend = "Uptrend"
        elif current_short < current_long and price_change < -2:
            trend = "Strong Downtrend"
        elif current_short < current_long and price_change < 0:
            trend = "Downtrend"
        else:
            trend = "Sideways"
        
        return {
            'direction': trend,
            'strength': abs(price_change),
            'confidence': 'High' if abs(price_change) > 3 else 'Medium'
        }
    except Exception as e:
        return {"error": str(e)}

def detect_chart_patterns(data):
    """Detect common chart patterns"""
    try:
        if len(data) < 30:
            return {"message": "Insufficient data for pattern analysis"}
        
        patterns_found = []
        
        # Simple pattern detection
        recent_highs = data['High'].tail(10)
        recent_lows = data['Low'].tail(10)
        
        # Double top pattern
        if len(recent_highs) >= 5:
            max_indices = recent_highs.nlargest(2).index
            if len(max_indices) == 2:
                patterns_found.append("Potential Double Top")
        
        # Double bottom pattern
        if len(recent_lows) >= 5:
            min_indices = recent_lows.nsmallest(2).index
            if len(min_indices) == 2:
                patterns_found.append("Potential Double Bottom")
        
        # Triangle pattern (converging highs and lows)
        if len(data) >= 20:
            recent_data = data.tail(20)
            high_trend = np.polyfit(range(len(recent_data)), recent_data['High'], 1)[0]
            low_trend = np.polyfit(range(len(recent_data)), recent_data['Low'], 1)[0]
            
            if high_trend < -0.1 and low_trend > 0.1:
                patterns_found.append("Symmetrical Triangle")
        
        return {
            'patterns': patterns_found if patterns_found else ["No clear patterns detected"],
            'confidence': 'Medium'
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_volume_patterns(data):
    """Analyze volume patterns"""
    try:
        if 'Volume' not in data.columns or len(data) < 20:
            return {"message": "Insufficient volume data"}
        
        # Volume moving average
        vol_ma = data['Volume'].rolling(window=10).mean()
        current_vol = data['Volume'].iloc[-1]
        avg_vol = vol_ma.iloc[-1]
        
        # Volume trend
        vol_change = (current_vol - avg_vol) / avg_vol * 100
        
        if vol_change > 50:
            volume_signal = "High Volume Spike"
        elif vol_change > 20:
            volume_signal = "Above Average Volume"
        elif vol_change < -30:
            volume_signal = "Low Volume"
        else:
            volume_signal = "Normal Volume"
        
        return {
            'signal': volume_signal,
            'volume_change': vol_change,
            'interpretation': "Strong momentum" if abs(vol_change) > 30 else "Normal activity"
        }
    except Exception as e:
        return {"error": str(e)}

# Market News Integration
def get_market_news(symbol=None):
    """Get market news (placeholder for real implementation)"""
    # This would typically integrate with a news API
    news_items = [
        {
            'title': 'Market Update: Tech Stocks Show Strong Performance',
            'summary': 'Technology sector leads gains in today\'s trading session...',
            'time': '2 hours ago',
            'relevance': 'High'
        },
        {
            'title': 'Federal Reserve Announces Interest Rate Decision',
            'summary': 'The Fed maintains current interest rates amid economic uncertainty...',
            'time': '4 hours ago',
            'relevance': 'High'
        },
        {
            'title': 'Quarterly Earnings Season Updates',
            'summary': 'Major companies report Q3 earnings with mixed results...',
            'time': '6 hours ago',
            'relevance': 'Medium'
        }
    ]
    
    return news_items