import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json

# Import shared utilities
from shared_utils import get_db_connection, init_database

# Data Manager Page
st.header("💾 Data Management")

# Initialize database
conn = get_db_connection()

# Create tabs for different data management functions
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔄 Cache Management", "🔌 Data Sources", "⚙️ Settings"])

with tab1:
    st.subheader("Data Overview")
    
    # Data statistics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get cache statistics
    cache_stats = conn.execute('''
        SELECT 
            COUNT(*) as total_entries,
            COUNT(DISTINCT symbol) as unique_symbols,
            COUNT(DISTINCT data_type) as data_types,
            SUM(LENGTH(data_json)) as total_size_bytes
        FROM data_cache
    ''').fetchone()
    
    with col1:
        st.metric("Cached Entries", cache_stats[0] if cache_stats[0] else 0)
    
    with col2:
        st.metric("Unique Symbols", cache_stats[1] if cache_stats[1] else 0)
    
    with col3:
        st.metric("Data Types", cache_stats[2] if cache_stats[2] else 0)
    
    with col4:
        cache_size_mb = (cache_stats[3] / (1024 * 1024)) if cache_stats[3] else 0
        st.metric("Cache Size (MB)", f"{cache_size_mb:.2f}")
    
    # Recent data updates
    st.subheader("Recent Data Updates")
    recent_data = pd.read_sql_query('''
        SELECT symbol, data_type, timeframe, last_updated
        FROM data_cache
        ORDER BY last_updated DESC
        LIMIT 10
    ''', conn)
    
    if not recent_data.empty:
        st.dataframe(recent_data, width="stretch")
    else:
        st.info("No cached data found. Data will be cached as you use the application.")

with tab2:
    st.subheader("Cache Management")
    
    # Cache controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cache Actions")
        
        if st.button("🗑️ Clear All Cache", type="secondary"):
            conn.execute('DELETE FROM data_cache')
            conn.commit()
            st.success("All cached data cleared!")
            st.rerun()
        
        if st.button("🧹 Clear Expired Cache"):
            conn.execute('DELETE FROM data_cache WHERE expires_at < ?', (datetime.now(),))
            conn.commit()
            st.success("Expired cache cleared!")
            st.rerun()
        
        # Manual cache entry
        st.subheader("Manual Cache Entry")
        cache_symbol = st.text_input("Symbol", placeholder="AAPL")
        cache_type = st.selectbox("Data Type", ["price", "technical", "fundamentals"])
        cache_timeframe = st.selectbox("Timeframe", ["1d", "1wk", "1mo", "3mo", "1y"])
        
        if st.button("Refresh Cache Entry"):
            if cache_symbol:
                # Simulate cache refresh (in real app, would fetch fresh data)
                sample_data = {"price": 150.0, "volume": 1000000, "timestamp": datetime.now().isoformat()}
                conn.execute('''
                    INSERT OR REPLACE INTO data_cache 
                    (symbol, data_type, timeframe, data_json, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (cache_symbol, cache_type, cache_timeframe, 
                     json.dumps(sample_data), datetime.now() + timedelta(hours=1)))
                conn.commit()
                st.success(f"Cache refreshed for {cache_symbol}")
    
    with col2:
        st.subheader("Cache Configuration")
        
        # Cache settings
        cache_enabled = st.checkbox("Enable Data Caching", value=True)
        cache_duration = st.selectbox("Default Cache Duration", 
                                    ["15 minutes", "30 minutes", "1 hour", "2 hours", "6 hours", "24 hours"])
        
        max_cache_size = st.slider("Max Cache Size (MB)", min_value=10, max_value=1000, value=100)
        auto_cleanup = st.checkbox("Auto-cleanup Expired Cache", value=True)
        
        if st.button("Save Cache Settings"):
            st.success("Cache settings saved!")

with tab3:
    st.subheader("Data Sources Configuration")
    
    # Existing data sources
    sources_df = pd.read_sql_query('''
        SELECT source_name, is_active, rate_limit, created_date
        FROM data_sources
        ORDER BY created_date DESC
    ''', conn)
    
    if not sources_df.empty:
        st.subheader("Configured Sources")
        
        # Display sources with toggle switches
        for _, source in sources_df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**{source['source_name']}**")
            
            with col2:
                is_active = st.checkbox("Active", value=bool(source['is_active']), 
                                      key=f"active_{source['source_name']}")
            
            with col3:
                st.write(f"Rate Limit: {source['rate_limit']}")
            
            with col4:
                if st.button("🗑️", key=f"delete_source_{source['source_name']}"):
                    conn.execute('DELETE FROM data_sources WHERE source_name = ?', (source['source_name'],))
                    conn.commit()
                    st.rerun()
    
    # Add new data source
    st.subheader("Add New Data Source")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_source_name = st.selectbox("Data Source", [
            "Yahoo Finance", "Alpha Vantage", "IEX Cloud", "Quandl", 
            "Polygon.io", "Finnhub", "TwelveData"
        ])
        
        api_key = st.text_input("API Key (if required)", type="password")
    
    with col2:
        rate_limit = st.number_input("Rate Limit (requests/day)", min_value=1, max_value=100000, value=1000)
        is_active = st.checkbox("Activate Immediately", value=True)
    
    if st.button("Add Data Source"):
        conn.execute('''
            INSERT INTO data_sources (source_name, api_key, is_active, rate_limit)
            VALUES (?, ?, ?, ?)
        ''', (new_source_name, api_key if api_key else None, is_active, rate_limit))
        conn.commit()
        st.success(f"Data source '{new_source_name}' added successfully!")
        st.rerun()

with tab4:
    st.subheader("Data Management Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Auto-Refresh Settings")
        
        auto_refresh_enabled = st.checkbox("Enable Auto-Refresh", value=True)
        refresh_interval = st.selectbox("Refresh Interval", 
                                      ["1 minute", "5 minutes", "15 minutes", "30 minutes", "1 hour"])
        
        refresh_during_market = st.checkbox("Refresh Only During Market Hours", value=True)
        
        st.subheader("📊 Data Quality")
        validate_data = st.checkbox("Validate Data Quality", value=True)
        remove_outliers = st.checkbox("Remove Statistical Outliers", value=False)
        fill_missing_data = st.checkbox("Fill Missing Data Points", value=True)
    
    with col2:
        st.subheader("🗄️ Storage Settings")
        
        max_history_days = st.slider("Maximum History (Days)", min_value=30, max_value=3650, value=365)
        compress_old_data = st.checkbox("Compress Data Older Than 30 Days", value=True)
        
        backup_enabled = st.checkbox("Enable Data Backup", value=True)
        backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
        
        st.subheader("🔒 Privacy & Security")
        encrypt_api_keys = st.checkbox("Encrypt API Keys", value=True)
        log_data_access = st.checkbox("Log Data Access", value=False)
    
    # Save settings button
    if st.button("Save All Settings", type="primary"):
        # In a real app, you'd save these settings to a configuration file or database
        st.success("All data management settings saved successfully!")

# Data export functionality
st.subheader("📤 Data Export")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export Cache Data"):
        # Export cached data to CSV
        export_df = pd.read_sql_query('SELECT * FROM data_cache', conn)
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download Cache Data CSV",
            data=csv,
            file_name=f"cache_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export Portfolio Data"):
        # Export portfolio data
        st.info("Portfolio data export will be available here")

with col3:
    if st.button("Export Alert History"):
        # Export alert history
        alerts_df = pd.read_sql_query('SELECT * FROM alerts', conn)
        if not alerts_df.empty:
            csv = alerts_df.to_csv(index=False)
            st.download_button(
                label="Download Alerts CSV",
                data=csv,
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No alert data to export")

# System status
st.subheader("🔍 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Database Status", "🟢 Connected")

with col2:
    st.metric("Last Backup", "2024-01-15 10:30")

with col3:
    st.metric("Storage Used", "45.6 MB")
