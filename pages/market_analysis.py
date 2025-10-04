import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Import shared utilities
from shared_utils import (ASSETS, fetch_real_data, calc_indicators, 
                         make_chart, get_market_news, get_signals,
                         get_fear_greed)

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# Market Analysis Page
st.header("📊 Advanced Market Analysis")

# Sidebar for asset selection
with st.sidebar:
    st.subheader("🎯 Asset Selection")
    
    # Asset category selection
    category = st.selectbox("Asset Category", list(ASSETS.keys()))
    
    # Symbol selection within category
    symbols_in_category = ASSETS[category]
    symbol_choice = st.selectbox("Select Asset", list(symbols_in_category.keys()), 
                                format_func=lambda x: f"{x} - {symbols_in_category[x]}")
    
    # Or manual input
    symbol = st.text_input("Or enter symbol manually", symbol_choice)
    
    # Time settings
    st.subheader("⏰ Time Settings")
    
    # Enhanced timeframe options (months to 5 years)
    timeframe_options = {
        "1 Month": ("1mo", "1d"),
        "3 Months": ("3mo", "1d"), 
        "6 Months": ("6mo", "1d"),
        "1 Year": ("1y", "1d"),
        "2 Years": ("2y", "1d"),
        "5 Years": ("5y", "1d"),
        "Year to Date": ("ytd", "1d"),
        "Maximum": ("max", "1d")
    }
    
    selected_timeframe = st.selectbox("Select Timeframe", list(timeframe_options.keys()), index=2)
    period, interval = timeframe_options[selected_timeframe]
    
    # Display selected settings
    st.info(f"📊 **{selected_timeframe}** | **{period}** period")
    
    # Advanced options
    st.subheader("🔧 Analysis Options")
    show_volume = st.checkbox("Show Volume", True)
    show_indicators = st.checkbox("Show Technical Indicators", True)
    enable_alerts = st.checkbox("Enable Price Alerts", False)

# Main analysis area
col1, col2 = st.columns([3, 1])

with col2:
    # Quick stats and controls
    st.subheader("⚡ Quick Actions")
    
    if st.button("🚀 Analyze Now", type="primary", width="stretch"):
        with st.spinner("Fetching and analyzing data..."):
            # Fetch data
            data = fetch_real_data(symbol, period, interval)
            
            if data is not None:
                # Calculate indicators
                data = calc_indicators(data)
                
                # Store in session state
                st.session_state.current_data = data
                st.session_state.current_symbol = symbol
                
                st.success("✅ Analysis complete!")
            else:
                st.error("❌ Failed to fetch data")
    
    # Add to watchlist
    if st.button("👁️ Add to Watchlist", width="stretch"):
        if symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(symbol)
            st.success(f"Added {symbol} to watchlist")
    
    # Quick portfolio actions
    st.subheader("💼 Portfolio Actions")
    buy_quantity = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.1)
    
    col_buy, col_sell = st.columns(2)
    with col_buy:
        if st.button("📈 Buy", width="stretch"):
            # Add to portfolio logic here
            st.success("Buy order simulated")
    
    with col_sell:
        if st.button("📉 Sell", width="stretch"):
            # Sell from portfolio logic here
            st.success("Sell order simulated")

with col1:
    # Main analysis display
    if 'current_data' in st.session_state and 'current_symbol' in st.session_state:
        data = st.session_state.current_data
        symbol = st.session_state.current_symbol
        
        # Price metrics
        latest_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
        
        # Display metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("💰 Current Price", f"${latest_price:.2f}", f"{price_change:+.2f}")
        
        with col_m2:
            st.metric("📊 Change %", f"{price_change_pct:+.2f}%")
        
        with col_m3:
            if 'Volume' in data.columns:
                st.metric("📈 Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            else:
                st.metric("📈 Volume", "N/A")
        
        with col_m4:
            if 'RSI' in data.columns:
                current_rsi = data['RSI'].iloc[-1]
                rsi_status = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "⚪ Neutral"
                st.metric("📊 RSI", f"{current_rsi:.1f}", rsi_status)
        
        # 🆕 Real-Time Trading Signals
        st.subheader("⚡ Real-Time Trading Signals")
        
        # Get real-time signals
        signals_data = get_signals(symbol, data)
        
        if 'error' not in signals_data:
            signal_col1, signal_col2, signal_col3 = st.columns([2, 1, 1])
            
            with signal_col1:
                st.subheader("📊 Active Signals")
                if signals_data['signals']:
                    for signal in signals_data['signals']:
                        if signal['type'] == 'BUY':
                            st.success(f"🟢 **{signal['type']}** - {signal['indicator']}: {signal['reason']} ({signal['strength']})")
                        elif signal['type'] == 'SELL':
                            st.error(f"🔴 **{signal['type']}** - {signal['indicator']}: {signal['reason']} ({signal['strength']})")
                        else:
                            st.info(f"ℹ️ **{signal['indicator']}**: {signal['reason']}")
                else:
                    st.info("No active signals detected")
            
            with signal_col2:
                st.metric("🎯 Overall Sentiment", 
                         signals_data['overall_sentiment'],
                         f"Confidence: {signals_data['confidence']:.1%}")
                
                st.metric("🕐 Last Updated", 
                         signals_data['timestamp'].split(' ')[1])
            
            with signal_col3:
                # Fear & Greed Index
                fear_greed = get_fear_greed()
                st.metric("😱 Fear & Greed", 
                         f"{fear_greed['index']}/100",
                         fear_greed['sentiment'])
                
                st.markdown(f"<p style='color: {fear_greed['color']}'>{fear_greed['recommendation']}</p>", 
                           unsafe_allow_html=True)
        else:
            st.error(f"Signals Error: {signals_data['error']}")
        
        st.markdown("---")
        
        # Advanced chart
        st.subheader("📈 Professional Chart")
        fig = make_chart(data, symbol, show_volume, show_indicators)
        st.plotly_chart(fig, config={"displayModeBar": True})
        
        # Technical analysis summary
        st.subheader("🔬 Technical Analysis Summary")
        
        col_tech1, col_tech2 = st.columns(2)
        
        with col_tech1:
            st.write("**Trend Analysis:**")
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]:
                    st.success("🔸 Short-term uptrend (SMA20 > SMA50)")
                else:
                    st.error("🔸 Short-term downtrend (SMA20 < SMA50)")
            
            if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
                bb_position = (latest_price - data['BB_lower'].iloc[-1]) / (data['BB_upper'].iloc[-1] - data['BB_lower'].iloc[-1])
                if bb_position > 0.8:
                    st.warning("🔸 Near upper Bollinger Band")
                elif bb_position < 0.2:
                    st.info("🔸 Near lower Bollinger Band")
                else:
                    st.info("🔸 Within Bollinger Band range")
        
        with col_tech2:
            st.write("**Momentum Indicators:**")
            if 'RSI' in data.columns:
                rsi_val = data['RSI'].iloc[-1]
                if rsi_val > 70:
                    st.error("🔸 RSI indicates overbought")
                elif rsi_val < 30:
                    st.success("🔸 RSI indicates oversold")
                else:
                    st.info("🔸 RSI in neutral territory")
            
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                    st.success("🔸 MACD bullish signal")
                else:
                    st.error("🔸 MACD bearish signal")
        
        # Recent data table
        st.subheader("📋 Recent OHLCV Data")
        display_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
        if 'RSI' in data.columns:
            display_data['RSI'] = data['RSI'].tail(10)
        if 'MACD' in data.columns:
            display_data['MACD'] = data['MACD'].tail(10)
        
        display_data.index = display_data.index.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_data, width="stretch")
        
        # Export options
        st.subheader("💾 Export Data")
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            if st.button("📁 Export CSV"):
                csv = data.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{symbol}_data.csv",
                    mime="text/csv"
                )
        
        with col_exp2:
            if st.button("📊 Export Excel"):
                # Excel export would go here
                st.info("Excel export feature coming soon")
        
        with col_exp3:
            if st.button("📈 Export Chart"):
                # Chart export would go here
                st.info("Chart export feature coming soon")
    
    else:
        # Welcome message
        st.info("👋 Welcome to Advanced Market Analysis! Select an asset and click 'Analyze Now' to get started.")
        
        # Market overview
        st.subheader("🌍 Market Overview")
        
        # Quick market data for major indices
        indices_data = {}
        major_indices = ['^GSPC', '^DJI', '^IXIC']
        
        for idx in major_indices:
            try:
                ticker_data = fetch_real_data(idx, "1d", "1d")
                if ticker_data is not None and len(ticker_data) >= 2:
                    latest = ticker_data['Close'].iloc[-1]
                    prev = ticker_data['Close'].iloc[-2]
                    change = ((latest - prev) / prev) * 100
                    indices_data[idx] = {'price': latest, 'change': change}
            except:
                pass
        
        if indices_data:
            st.write("**Major Indices Performance:**")
            col_idx1, col_idx2, col_idx3 = st.columns(3)
            
            names = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ'}
            
            for i, (idx, data) in enumerate(indices_data.items()):
                col = [col_idx1, col_idx2, col_idx3][i]
                with col:
                    st.metric(
                        names.get(idx, idx),
                        f"{data['price']:.2f}",
                        f"{data['change']:+.2f}%"
                    )
        
        # 🆕 Real-time Market News
        st.subheader("📰 Latest Market News")
        
        news_items = get_market_news()
        
        for i, news in enumerate(news_items):
            with st.expander(f"📰 {news['title']} • {news['time']}", expanded=(i==0)):
                st.write(news['summary'])
                
                # News relevance badge
                relevance_color = {
                    'High': '🔴',
                    'Medium': '🟡', 
                    'Low': '🟢'
                }.get(news['relevance'], '⚪')
                
                st.markdown(f"**Relevance**: {relevance_color} {news['relevance']}")
        
        # Market sentiment indicator
        st.subheader("📊 Market Sentiment")
        
        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
        
        with sentiment_col1:
            st.metric("Fear & Greed Index", "65", "+5 (Greed)")
        
        with sentiment_col2:
            st.metric("VIX Level", "18.5", "-2.3 (Lower)")
        
        with sentiment_col3:
            st.metric("Market Trend", "Bullish", "↗️ Strong")

# Watchlist sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("👁️ Watchlist")
    
    for watch_symbol in st.session_state.watchlist:
        col_w1, col_w2 = st.columns([3, 1])
        with col_w1:
            st.write(watch_symbol)
        with col_w2:
            if st.button("❌", key=f"remove_{watch_symbol}", help="Remove from watchlist"):
                st.session_state.watchlist.remove(watch_symbol)
                st.rerun()
