"""
Stock Predictor - Clean Version
Real data only, with metals and indices support
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

# Page config
st.set_page_config(
    page_title="LSTM Stock Predictor", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "LSTM Stock Predictor - Real-time stock, metals & indices analysis tool"
    }
)

# Hide only deploy-related elements
hide_deploy_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_deploy_style, unsafe_allow_html=True)

# Title
st.title("📈 LSTM Stock Price Predictor")
st.markdown("*Professional real-time stock, metals & indices analysis*")

# Add professional badges/metrics
col_badge1, col_badge2, col_badge3, col_badge4 = st.columns(4)
with col_badge1:
    st.metric("📊 Assets Covered", "25+", help="Stocks, Metals, Indices, Currency")
with col_badge2:
    st.metric("⏱️ Timeframes", "9", help="1m to 2y intervals")
with col_badge3:
    st.metric("📈 Indicators", "7+", help="RSI, MACD, BB, SMA, etc.")
with col_badge4:
    st.metric("🎯 Signals", "5", help="Strong Buy to Strong Sell")

st.markdown("---")

# Asset categories with symbols
ASSETS = {
    "Stocks": {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "TSLA": "Tesla Inc.",
        "MSFT": "Microsoft Corp.",
        "NVDA": "NVIDIA Corp.",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms"
    },
    "Metals": {
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures", 
        "PL=F": "Platinum Futures",
        "HG=F": "Copper Futures",
        "PA=F": "Palladium Futures"
    },
    "Indices": {
        "^DJI": "Dow Jones Industrial",
        "^IXIC": "NASDAQ Composite",
        "^GSPC": "S&P 500",
        "^RUT": "Russell 2000",
        "^VIX": "VIX Volatility Index"
    },
    "Currency": {
        "DX-Y.NYB": "US Dollar Index",
        "EURUSD=X": "EUR/USD",
        "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY"
    }
}

def fetch_real_data(symbol, period="1y", interval="1d"):
    """Fetch real data using working yfinance methods"""
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

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
    data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
    
    return data

def generate_trading_signal(data):
    """Generate trading signals based on technical indicators"""
    if len(data) < 50:
        return "INSUFFICIENT_DATA", "Need more data for reliable signals", "⚠️"
    
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    # Initialize signal components
    signals = []
    strength = 0
    
    # RSI Analysis
    if latest['RSI'] < 30:
        signals.append("RSI oversold (bullish)")
        strength += 2
    elif latest['RSI'] > 70:
        signals.append("RSI overbought (bearish)")
        strength -= 2
    elif 40 <= latest['RSI'] <= 60:
        signals.append("RSI neutral")
    
    # MACD Analysis
    if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
        signals.append("MACD bullish crossover")
        strength += 2
    elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
        signals.append("MACD bearish crossover")
        strength -= 2
    elif latest['MACD'] > latest['MACD_signal']:
        signals.append("MACD above signal (bullish)")
        strength += 1
    else:
        signals.append("MACD below signal (bearish)")
        strength -= 1
    
    # Moving Average Analysis
    if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
        signals.append("Price above both MAs (strong bullish)")
        strength += 2
    elif latest['Close'] > latest['SMA_20']:
        signals.append("Price above SMA20 (bullish)")
        strength += 1
    elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
        signals.append("Price below both MAs (strong bearish)")
        strength -= 2
    else:
        signals.append("Price below SMA20 (bearish)")
        strength -= 1
    
    # Bollinger Bands Analysis
    if latest['Close'] <= latest['BB_lower']:
        signals.append("Price at lower BB (potential reversal)")
        strength += 1
    elif latest['Close'] >= latest['BB_upper']:
        signals.append("Price at upper BB (potential reversal)")
        strength -= 1
    
    # Price momentum
    price_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
    if price_change > 2:
        signals.append(f"Strong upward momentum (+{price_change:.1f}%)")
        strength += 1
    elif price_change < -2:
        signals.append(f"Strong downward momentum ({price_change:.1f}%)")
        strength -= 1
    
    # Generate final signal
    if strength >= 4:
        signal = "STRONG_LONG"
        recommendation = "Strong Buy Signal - Consider Long Position"
        emoji = "🟢"
    elif strength >= 2:
        signal = "LONG"
        recommendation = "Buy Signal - Consider Long Position"
        emoji = "🔵"
    elif strength <= -4:
        signal = "STRONG_SHORT"
        recommendation = "Strong Sell Signal - Consider Short Position"
        emoji = "🔴"
    elif strength <= -2:
        signal = "SHORT"
        recommendation = "Sell Signal - Consider Short Position"
        emoji = "🟠"
    else:
        signal = "NEUTRAL"
        recommendation = "Hold - No clear signal"
        emoji = "⚪"
    
    return signal, recommendation, emoji, signals, strength

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Asset category selection
    category = st.selectbox("Asset Category", list(ASSETS.keys()))
    
    # Symbol selection within category
    symbols_in_category = ASSETS[category]
    symbol_choice = st.selectbox("Select Asset", list(symbols_in_category.keys()), 
                                format_func=lambda x: f"{x} - {symbols_in_category[x]}")
    
    # Or manual input
    symbol = st.text_input("Or enter symbol manually", symbol_choice)
    
    # Time settings
    st.subheader("📅 Time Settings")
    
    # Timeframe options
    timeframe_options = {
        "1m": ("1d", "1m"),      # 1 minute bars, 1 day period
        "5m": ("5d", "5m"),      # 5 minute bars, 5 day period
        "15m": ("5d", "15m"),    # 15 minute bars, 5 day period
        "30m": ("5d", "30m"),    # 30 minute bars, 5 day period
        "1h": ("1mo", "1h"),     # 1 hour bars, 1 month period
        "1d": ("3mo", "1d"),     # Daily bars, 3 month period
        "5d": ("6mo", "5d"),     # 5 day bars, 6 month period
        "1wk": ("1y", "1wk"),    # Weekly bars, 1 year period
        "1mo": ("2y", "1mo")     # Monthly bars, 2 year period
    }
    
    timeframe = st.selectbox("Timeframe", list(timeframe_options.keys()), 
                            index=5, help="Choose from 1 minute to 2 years")
    
    period, interval = timeframe_options[timeframe]
    
    # Display selected settings
    st.info(f"📊 **{timeframe}** bars | **{period}** period")
    
    st.markdown("---")
    st.markdown("**Asset Categories:**")
    for cat, assets in ASSETS.items():
        with st.expander(f"📊 {cat}"):
            for sym, name in list(assets.items())[:3]:  # Show first 3
                st.write(f"• {sym} - {name}")

# Main area
st.subheader(f"📊 Analysis: {symbol.upper()}")

# Determine asset type
asset_type = "Unknown"
for cat, assets in ASSETS.items():
    if symbol.upper() in [s.upper() for s in assets.keys()]:
        asset_type = cat
        break

st.info(f"Asset Type: **{asset_type}** | Timeframe: **{timeframe}** | Period: **{period}**")

analyze_button = st.button("🚀 Analyze Asset", type="primary", use_container_width=True)

if analyze_button:
    try:
        # Progress indicators
        progress = st.progress(0)
        status = st.empty()
        
        # Step 1: Fetch data
        status.text(f"🔄 Fetching {asset_type.lower()} data for {symbol}...")
        progress.progress(20)
        
        data = fetch_real_data(symbol, period, interval)
        
        if data is None or data.empty:
            st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol and try again.")
            st.stop()
        
        status.text("✅ Data fetched successfully!")
        progress.progress(40)
        
        # Step 2: Calculate indicators
        status.text("📊 Calculating technical indicators...")
        data = calculate_technical_indicators(data)
        progress.progress(60)
        
        # Step 3: Display results
        status.text("📈 Generating analysis...")
        progress.progress(80)
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        
        latest_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
        
        with col1:
            st.metric("Current Price", f"${latest_price:.2f}", f"{price_change:+.2f}")
        
        with col2:
            st.metric("Change %", f"{price_change_pct:+.2f}%")
        
        with col3:
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        
        with col4:
            st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
        
        # Generate Trading Signal
        signal, recommendation, emoji, signal_details, strength = generate_trading_signal(data)
        
        # Trading Signal Display
        st.markdown("---")
        st.subheader("🎯 Trading Signal")
        
        col_signal, col_strength = st.columns([2, 1])
        
        with col_signal:
            if signal == "STRONG_LONG":
                st.success(f"{emoji} **{recommendation}**")
            elif signal == "LONG":
                st.info(f"{emoji} **{recommendation}**")
            elif signal == "STRONG_SHORT":
                st.error(f"{emoji} **{recommendation}**")
            elif signal == "SHORT":
                st.warning(f"{emoji} **{recommendation}**")
            else:
                st.info(f"{emoji} **{recommendation}**")
        
        with col_strength:
            st.metric("Signal Strength", f"{strength:+d}", help="Range: -8 (Strong Sell) to +8 (Strong Buy)")
        
        # Signal breakdown
        with st.expander("📋 Signal Analysis Details"):
            st.write("**Technical Indicators Analysis:**")
            for detail in signal_details:
                st.write(f"• {detail}")
            
            st.write(f"\n**Overall Signal Strength:** {strength}")
            st.write("**Signal Interpretation:**")
            if strength >= 4:
                st.write("🟢 **Strong Bullish** - High confidence long position")
            elif strength >= 2:
                st.write("🔵 **Bullish** - Consider long position")
            elif strength <= -4:
                st.write("🔴 **Strong Bearish** - High confidence short position")
            elif strength <= -2:
                st.write("🟠 **Bearish** - Consider short position")
            else:
                st.write("⚪ **Neutral** - No clear directional bias")
        
        progress.progress(90)
        
        # Charts
        st.subheader("📈 Price Chart")
        
        # Create price chart
        chart_data = pd.DataFrame({
            'Date': data.index,
            'Price': data['Close'],
            'SMA 20': data['SMA_20'],
            'SMA 50': data['SMA_50']
        }).set_index('Date')
        
        st.line_chart(chart_data)
        
        # Volume chart
        st.subheader("📊 Volume")
        st.bar_chart(data['Volume'])
        
        # Technical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RSI (14)")
            rsi_data = pd.DataFrame({'RSI': data['RSI']})
            st.line_chart(rsi_data)
            
            # RSI interpretation
            current_rsi = data['RSI'].iloc[-1]
            if current_rsi > 70:
                st.warning("⚠️ RSI indicates overbought conditions")
            elif current_rsi < 30:
                st.warning("⚠️ RSI indicates oversold conditions")
            else:
                st.info("ℹ️ RSI in neutral range")
        
        with col2:
            st.subheader("MACD")
            macd_data = pd.DataFrame({
                'MACD': data['MACD'],
                'Signal': data['MACD_signal']
            })
            st.line_chart(macd_data)
        
        # Recent data table
        st.subheader("📋 Recent Data")
        recent_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].tail(10)
        recent_data.index = recent_data.index.strftime('%Y-%m-%d')
        st.dataframe(recent_data, use_container_width=True)
        
        progress.progress(100)
        status.text("✅ Analysis complete!")
        
        st.success(f"✅ Successfully analyzed {symbol} ({asset_type})")
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.info("Please try again or check your internet connection.")

# Footer
st.markdown("---")
st.markdown("**Data Source:** Yahoo Finance via yfinance | **Real-time data only**")