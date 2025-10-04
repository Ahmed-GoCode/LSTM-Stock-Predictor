"""
LSTM Stock Trading Platform
Main app with multiple pages for trading analysis
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import json
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Set up the page
st.set_page_config(
    page_title="LSTM Trading Platform", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional LSTM Trading Platform - Advanced real-time analysis"
    }
)

# Professional CSS styling
professional_css = """
<style>
/* Dark theme colors */
:root {
    --primary-color: #00d4aa;
    --secondary-color: #ff6b6b;
    --background-dark: #0e1117;
    --surface-dark: #1a1a1a;
    --text-light: #fafafa;
    --accent-blue: #4fc3f7;
    --accent-orange: #ff9800;
}

/* Hide footer and deploy button */
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Custom metrics styling */
.metric-card {
    background: linear-gradient(135deg, var(--surface-dark), #2d2d2d);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid var(--primary-color);
    margin: 0.5rem 0;
}

/* Professional header */
.main-header {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}

/* Navigation styling */
.nav-pills {
    background: var(--surface-dark);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

/* Chart container */
.chart-container {
    background: var(--surface-dark);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #333;
}

/* Alert boxes */
.alert-success {
    background: linear-gradient(90deg, #00d4aa, #00bfa5);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.alert-warning {
    background: linear-gradient(90deg, #ff9800, #f57c00);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.alert-danger {
    background: linear-gradient(90deg, #ff6b6b, #e53e3e);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

/* Sidebar enhancements */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, var(--surface-dark), #2d2d2d);
}

/* Professional buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), #00bfa5);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 212, 170, 0.3);
}

/* Loading animations */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Table styling */
.dataframe {
    background: var(--surface-dark);
    border-radius: 8px;
    overflow: hidden;
}

.dataframe th {
    background: var(--primary-color);
    color: white;
}

/* Status indicators */
.status-online {
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #00d4aa;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
</style>
"""

st.markdown(professional_css, unsafe_allow_html=True)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'GOOGL', 'TSLA']

# Asset categories with enhanced selection
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

def init_database():
    """Initialize SQLite database for data storage"""
    conn = sqlite3.connect('trading_platform.db')
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
    conn.close()

def fetch_real_data(symbol, period="1y", interval="1d"):
    """Enhanced data fetching with caching"""
    try:
        # Method 1: Use proven working yfinance.ticker method
        import yfinance.ticker
        ticker = yfinance.ticker.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if not data.empty and len(data) > 5:
            return data
            
        # Method 2: Fallback to standard yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        
        if not data.empty and len(data) > 5:
            return data
            
        # Method 3: Try download method
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=True, show_errors=False)
        
        if not data.empty and len(data) > 5:
            return data
            
        return None
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_advanced_indicators(data):
    """Calculate comprehensive technical indicators"""
    # Basic indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # EMA
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
    
    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
    data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
    
    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    # Williams %R
    data['Williams_R'] = -100 * ((high_14 - data['Close']) / (high_14 - low_14))
    
    # VWAP (Volume Weighted Average Price)
    if 'Volume' in data.columns:
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # ATR (Average True Range)
    data['TR'] = np.maximum(
        data['High'] - data['Low'],
        np.maximum(
            abs(data['High'] - data['Close'].shift()),
            abs(data['Low'] - data['Close'].shift())
        )
    )
    data['ATR'] = data['TR'].rolling(window=14).mean()
    
    return data

def generate_ai_prediction(data, days_ahead=30):
    """Generate LSTM predictions"""
    try:
        if len(data) < 60:
            return None, "Insufficient data for prediction"
        
        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        # Create sequences
        sequence_length = 60
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Simple LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model (quick training for demo)
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        
        # Generate predictions
        last_sequence = scaled_data[-sequence_length:]
        predictions = []
        
        for _ in range(days_ahead):
            pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
            predictions.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions.flatten(), "Prediction generated successfully"
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def create_advanced_chart(data, symbol, show_volume=True, show_indicators=True):
    """Create professional candlestick chart with indicators"""
    
    # Create subplots
    if show_volume and show_indicators:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=(f'{symbol} Price Chart', 'Volume', 'RSI', 'MACD')
        )
    elif show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} Price Chart', 'Volume')
        )
    else:
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(title=f'{symbol} Price Chart')
    
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
    
    # Add moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                name='SMA 50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB_upper' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                opacity=0.3
            ),
            row=1, col=1
        )
    
    # Volume
    if show_volume and 'Volume' in data.columns:
        colors = ['#00d4aa' if close >= open else '#ff6b6b' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # RSI
    if show_indicators and 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # MACD
    if show_indicators and 'MACD' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_signal'],
                name='MACD Signal',
                line=dict(color='red', width=1)
            ),
            row=4, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD_histogram'],
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.6
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        height=800 if show_volume and show_indicators else (600 if show_volume else 500),
        showlegend=True,
        xaxis_rangeslider_visible=False,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Initialize database
init_database()

# Main header
st.markdown('''
<div class="main-header">
    <h1>🚀 Professional LSTM Trading Platform</h1>
    <p>Advanced AI-Powered Financial Analysis & Portfolio Management</p>
    <div class="status-online"></div> Live Market Data Connected
</div>
''', unsafe_allow_html=True)

# Navigation
pages = [
    "📊 Market Analysis",
    "💼 Portfolio Manager", 
    "🤖 AI Predictions",
    "📈 Advanced Charts",
    "⚡ Real-time Dashboard",
    "🔔 Alert System",
    "💾 Data Manager",
    "⚙️ Settings"
]

selected_page = st.selectbox("🧭 Navigate", pages, key="main_nav")

# Load the selected page
if selected_page == "📊 Market Analysis":
    exec(open("pages/market_analysis.py", encoding='utf-8').read())
elif selected_page == "💼 Portfolio Manager":
    exec(open("pages/portfolio_manager.py", encoding='utf-8').read())
elif selected_page == "🤖 AI Predictions":
    exec(open("pages/ai_predictions.py", encoding='utf-8').read())
elif selected_page == "📈 Advanced Charts":
    exec(open("pages/advanced_charts.py", encoding='utf-8').read())
elif selected_page == "⚡ Real-time Dashboard":
    exec(open("pages/realtime_dashboard.py", encoding='utf-8').read())
elif selected_page == "🔔 Alert System":
    exec(open("pages/alert_system.py", encoding='utf-8').read())
elif selected_page == "💾 Data Manager":
    exec(open("pages/data_manager.py", encoding='utf-8').read())
elif selected_page == "⚙️ Settings":
    exec(open("pages/settings.py", encoding='utf-8').read())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Professional LSTM Trading Platform</strong> | Real-time Market Analysis | AI-Powered Predictions</p>
    <p>⚡ Live Data • 🔒 Secure • 📊 Professional Grade</p>
</div>
""", unsafe_allow_html=True)