import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

#                 if fig:
                    st.plotly_chart(fig, config={"displayModeBar": True})
                
                # Additional analysis
                st.subheader("📊 Quick Analysis")
                
                if len(data) >= 2:
                    current_price = data['Close'].iloc[-1]
                    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                    with col3:
                        if len(data) >= 20:
                            volume_avg = data['Volume'].rolling(20).mean().iloc[-1]
                            st.metric("Avg Volume (20d)", f"{volume_avg:,.0f}")
                        else:
                            st.metric("Avg Volume", f"{data['Volume'].mean():,.0f}")
                else:
                    st.warning("⚠️ Insufficient data for analysis")lities
from shared_utils import ASSETS, fetch_real_data, create_advanced_chart

# Advanced Charts Page
st.header("📈 Advanced Charting")

# Chart configuration section
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Chart Configuration")
    
    # Asset selection
    category = st.selectbox("Asset Category", list(ASSETS.keys()))
    symbols_in_category = ASSETS[category]
    symbol = st.selectbox("Select Asset", list(symbols_in_category.keys()), 
                         format_func=lambda x: f"{x} - {symbols_in_category[x]}")
    
    # Chart settings
    st.subheader("🎨 Chart Settings")
    chart_types = ["Candlestick", "Line", "Area", "OHLC", "Mountain"]
    selected_chart = st.selectbox("Chart Type", chart_types)
    
    # Timeframe
    timeframes = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
    timeframe = st.selectbox("Timeframe", timeframes, index=5)
    
    # Technical indicators
    st.subheader("📊 Technical Indicators")
    show_volume = st.checkbox("Show Volume", value=True)
    show_sma = st.checkbox("Simple Moving Average", value=True)
    show_ema = st.checkbox("Exponential Moving Average")
    show_bollinger = st.checkbox("Bollinger Bands")
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD")
    
    # Chart styling
    st.subheader("🎨 Styling")
    chart_theme = st.selectbox("Theme", ["Dark", "Light", "Plotly"])
    chart_height = st.slider("Chart Height", 400, 1000, 600)

with col2:
    if st.button("🚀 Generate Chart", type="primary", width="stretch"):
        with st.spinner("Fetching data and generating advanced chart..."):
            # Fetch data
            data = fetch_real_data(symbol, timeframe, "1d")
            
            if data is not None:
                # Calculate indicators
                from shared_utils import calculate_advanced_indicators
                data_with_indicators = calculate_advanced_indicators(data)
                
                # Create different chart types
                if selected_chart == "Candlestick":
                    fig = create_advanced_chart(data_with_indicators, symbol, show_volume, True)
                
                elif selected_chart == "Line":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#00d4aa', width=2)
                    ))
                    fig.update_layout(
                        title=f'{symbol} - Line Chart',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template='plotly_dark' if chart_theme == 'Dark' else 'plotly_white',
                        height=chart_height
                    )
                
                elif selected_chart == "Area":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        fill='tonexty',
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#00d4aa'),
                        fillcolor='rgba(0, 212, 170, 0.3)'
                    ))
                    fig.update_layout(
                        title=f'{symbol} - Area Chart',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template='plotly_dark' if chart_theme == 'Dark' else 'plotly_white',
                        height=chart_height
                    )
                
                elif selected_chart == "OHLC":
                    fig = go.Figure()
                    fig.add_trace(go.Ohlc(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='OHLC'
                    ))
                    fig.update_layout(
                        title=f'{symbol} - OHLC Chart',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template='plotly_dark' if chart_theme == 'Dark' else 'plotly_white',
                        height=chart_height
                    )
                
                elif selected_chart == "Mountain":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        fill='tozeroy',
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#ff6b6b'),
                        fillcolor='rgba(255, 107, 107, 0.3)'
                    ))
                    fig.update_layout(
                        title=f'{symbol} - Mountain Chart',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template='plotly_dark' if chart_theme == 'Dark' else 'plotly_white',
                        height=chart_height
                    )
                
                if fig:
                    st.plotly_chart(fig, config={"displayModeBar": True})
                
                # Additional analysis
                st.subheader("� Quick Analysis")
                current_price = data['Close'].iloc[-1]
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                with col3:
                    volume_avg = data['Volume'].rolling(20).mean().iloc[-1]
                    st.metric("Avg Volume (20d)", f"{volume_avg:,.0f}")

# Additional chart tools
st.subheader("🛠️ Chart Tools")

tools_col1, tools_col2, tools_col3 = st.columns(3)

with tools_col1:
    st.subheader("📊 Compare Assets")
    if st.button("Multi-Asset Comparison"):
        st.info("💡 Feature coming soon: Compare multiple assets on one chart")

with tools_col2:
    st.subheader("🔍 Pattern Recognition")
    if st.button("Detect Patterns"):
        st.info("💡 Feature coming soon: AI-powered pattern detection")

with tools_col3:
    st.subheader("📈 Custom Indicators")
    if st.button("Add Custom Indicator"):
        st.info("💡 Feature coming soon: Create your own technical indicators")
