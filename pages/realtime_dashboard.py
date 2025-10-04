import streamlit as st
import pandas as pd
from datetime import datetime

# Import shared utilities
from shared_utils import ASSETS, fetch_real_data

# Real-time Dashboard Page
st.header("⚡ Real-time Dashboard")

# Real-time market data display
st.subheader("📊 Live Market Overview")

# Create columns for different market sections
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏢 Major Stocks")
    # Placeholder for real-time stock data
    stocks_data = {
        'Symbol': ['AAPL', 'GOOGL', 'TSLA', 'MSFT'],
        'Price': [175.23, 2734.56, 248.91, 378.45],
        'Change': ['+2.34', '-15.67', '+12.45', '+5.67'],
        'Change %': ['+1.35%', '-0.57%', '+5.26%', '+1.52%']
    }
    st.dataframe(pd.DataFrame(stocks_data), width="stretch")

with col2:
    st.subheader("🥇 Metals")
    metals_data = {
        'Symbol': ['GC=F', 'SI=F', 'PL=F', 'HG=F'],
        'Price': [1924.50, 23.45, 987.60, 3.87],
        'Change': ['+12.30', '+0.45', '-5.40', '+0.12'],
        'Change %': ['+0.64%', '+1.95%', '-0.54%', '+3.20%']
    }
    st.dataframe(pd.DataFrame(metals_data), width="stretch")

with col3:
    st.subheader("📊 Indices")
    indices_data = {
        'Symbol': ['^GSPC', '^DJI', '^IXIC', '^VIX'],
        'Price': [4567.89, 34567.12, 14234.56, 18.45],
        'Change': ['+23.45', '+156.78', '+89.34', '-2.67'],
        'Change %': ['+0.52%', '+0.46%', '+0.63%', '-12.65%']
    }
    st.dataframe(pd.DataFrame(indices_data), width="stretch")

# Auto-refresh functionality
st.subheader("🔄 Auto-Refresh Settings")
auto_refresh = st.checkbox("Enable Auto-Refresh")
if auto_refresh:
    refresh_interval = st.selectbox("Refresh Interval", ["30 seconds", "1 minute", "5 minutes", "15 minutes"])
    st.info(f"Auto-refreshing every {refresh_interval}")

# Market status
st.subheader("🔔 Market Status")
market_status = "🟢 OPEN" if datetime.now().hour < 16 else "🔴 CLOSED"
st.metric("Market Status", market_status)

# News feed placeholder
st.subheader("📰 Market News")
st.info("📰 Latest market news would be displayed here in real-time")
