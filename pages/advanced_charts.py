import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Import shared utilities
from shared_utils import (ASSETS, fetch_real_data, make_chart, 
                         export_chart_as_html, export_chart_as_image, 
                         export_data_as_csv, export_data_as_excel,
                         detect_patterns)

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
            try:
                # Fetch data
                data = fetch_real_data(symbol, timeframe, "1d")
                
                if data is not None and not data.empty:
                    # Calculate indicators
                    from shared_utils import calculate_advanced_indicators
                    data_with_indicators = calculate_advanced_indicators(data)
                    
                    fig = None
                    
                    # Create different chart types
                    if selected_chart == "Candlestick":
                        fig = make_chart(data_with_indicators, symbol, show_volume, True)
                    
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
                    
                    # Display chart
                    if fig:
                        st.plotly_chart(fig, config={'displayModeBar': True})
                        
                        # 🆕 Export functionality
                        st.subheader("📤 Export Options")
                        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                        
                        with export_col1:
                            if st.button("🖼️ Export PNG"):
                                export_chart_as_image(fig, f"{symbol}_chart", "png")
                        
                        with export_col2:
                            if st.button("📄 Export HTML"):
                                export_chart_as_html(fig, f"{symbol}_chart")
                        
                        with export_col3:
                            if st.button("📊 Export CSV"):
                                export_data_as_csv(data, f"{symbol}_data")
                        
                        with export_col4:
                            if st.button("📗 Export Excel"):
                                export_data_as_excel(data, f"{symbol}_data")
                    
                    # Additional analysis with proper error handling
                    st.subheader("📊 Quick Analysis")
                    
                    if len(data) >= 2:
                        current_price = data['Close'].iloc[-1]
                        previous_price = data['Close'].iloc[-2]
                        price_change = current_price - previous_price
                        price_change_pct = (price_change / previous_price) * 100
                        
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
                                volume_avg = data['Volume'].mean()
                                st.metric("Avg Volume", f"{volume_avg:,.0f}")
                    else:
                        st.warning("⚠️ Insufficient data for detailed analysis")
                        if len(data) >= 1:
                            current_price = data['Close'].iloc[-1]
                            st.metric("Current Price", f"${current_price:.2f}")
                
                else:
                    st.error("❌ No data available for the selected symbol and timeframe")
                    
            except Exception as e:
                st.error(f"❌ Error generating chart: {str(e)}")
                st.info("💡 Try selecting a different symbol or timeframe")

# Additional chart tools
st.subheader("🛠️ Chart Tools")

tools_col1, tools_col2, tools_col3 = st.columns(3)

with tools_col1:
    st.subheader("📊 Compare Assets")
    if st.button("Multi-Asset Comparison"):
        st.info("💡 Feature coming soon: Compare multiple assets on one chart")

with tools_col2:
    st.subheader("🔍 AI Pattern Recognition")
    if st.button("🤖 Detect Patterns"):
        if 'data' in locals() and data is not None and not data.empty:
            with st.spinner("Analyzing patterns..."):
                patterns = detect_patterns(data)
                
                # Display pattern analysis
                for pattern_type, analysis in patterns.items():
                    with st.expander(f"📊 {pattern_type.replace('_', ' ')}"):
                        if isinstance(analysis, dict):
                            if 'error' in analysis:
                                st.error(f"Error: {analysis['error']}")
                            elif 'message' in analysis:
                                st.info(analysis['message'])
                            else:
                                for key, value in analysis.items():
                                    st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                        else:
                            st.write(analysis)
        else:
            st.warning("Please generate a chart first to detect patterns")

with tools_col3:
    st.subheader("📈 Multi-Asset Comparison")
    if st.button("🔄 Compare Assets"):
        # Multi-select for comparison
        st.subheader("Select Assets to Compare")
        available_symbols = []
        for category in ASSETS.values():
            available_symbols.extend(list(category.keys()))
        
        compare_symbols = st.multiselect(
            "Choose assets to compare (max 5):",
            available_symbols,
            default=[symbol],
            max_selections=5
        )
        
        if len(compare_symbols) > 1:
            st.subheader("� Comparison Chart")
            comparison_fig = go.Figure()
            
            for comp_symbol in compare_symbols:
                comp_data = fetch_real_data(comp_symbol, timeframe)
                if comp_data is not None and not comp_data.empty:
                    # Normalize prices to percentage change from first day
                    normalized_prices = (comp_data['Close'] / comp_data['Close'].iloc[0] - 1) * 100
                    
                    comparison_fig.add_trace(go.Scatter(
                        x=comp_data.index,
                        y=normalized_prices,
                        mode='lines',
                        name=comp_symbol,
                        line=dict(width=2)
                    ))
            
            comparison_fig.update_layout(
                title="Asset Performance Comparison (% Change)",
                xaxis_title="Date",
                yaxis_title="Percentage Change (%)",
                template='plotly_dark' if chart_theme == 'Dark' else 'plotly_white',
                height=400
            )
            
            st.plotly_chart(comparison_fig, config={'displayModeBar': True})
        else:
            st.info("Select at least 2 assets to compare")

# Additional features
st.subheader("�️ Advanced Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.subheader("🎯 Custom Technical Indicators")
    indicator_type = st.selectbox("Add Custom Indicator", 
                                ["None", "Custom Moving Average", "Price Bands", "Momentum Oscillator"])
    
    if indicator_type == "Custom Moving Average":
        ma_period = st.number_input("Moving Average Period", min_value=5, max_value=200, value=50)
        if st.button("Add Custom MA"):
            if 'data' in locals() and data is not None:
                custom_ma = data['Close'].rolling(window=ma_period).mean()
                st.line_chart(custom_ma.tail(100))
                st.success(f"Added {ma_period}-day Moving Average")

with feature_col2:
    st.subheader("⚙️ Chart Configuration")
    if st.button("� Save Configuration"):
        config = {
            'symbol': symbol,
            'chart_type': selected_chart,
            'timeframe': timeframe,
            'theme': chart_theme,
            'height': chart_height
        }
        st.session_state.saved_chart_config = config
        st.success("✅ Chart configuration saved!")
    
    if st.button("� Load Configuration"):
        if 'saved_chart_config' in st.session_state:
            config = st.session_state.saved_chart_config
            st.json(config)
            st.success("✅ Configuration loaded!")
        else:
            st.info("No saved configuration found")
