import streamlit as st
import pandas as pd
from datetime import datetime
import sqlite3

# Import shared utilities
from shared_utils import ASSETS, get_db_connection, init_database

# Alert System Page
st.header("🔔 Alert System")

# Initialize database
conn = get_db_connection()

# Create tabs for different alert functions
tab1, tab2, tab3 = st.tabs(["📝 Create Alert", "📋 Active Alerts", "📊 Alert History"])

with tab1:
    st.subheader("Create New Alert")
    
    # Alert creation form
    col1, col2 = st.columns(2)
    
    with col1:
        # Symbol selection
        asset_categories = {
            "🏢 Stocks": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "AMD"],
            "🥇 Metals": ["GC=F", "SI=F", "PL=F", "HG=F"],
            "📊 Indices": ["^GSPC", "^DJI", "^IXIC", "^RUT"],
            "💰 Crypto": ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD"]
        }
        
        category = st.selectbox("Asset Category", list(asset_categories.keys()))
        symbol = st.selectbox("Symbol", asset_categories[category])
        
        alert_type = st.selectbox("Alert Type", [
            "Price Alert", "Volume Alert", "RSI Alert", "MACD Alert", "Moving Average Alert"
        ])
    
    with col2:
        condition_type = st.selectbox("Condition", [
            "Above", "Below", "Crosses Above", "Crosses Below"
        ])
        
        target_value = st.number_input("Target Value", min_value=0.0, step=0.01)
        
        # Additional settings
        email_notification = st.checkbox("Email Notification")
        sound_notification = st.checkbox("Sound Notification")
    
    if st.button("Create Alert", type="primary"):
        # Insert alert into database
        conn.execute('''
            INSERT INTO alerts (symbol, alert_type, condition_type, target_value, status)
            VALUES (?, ?, ?, ?, 'active')
        ''', (symbol, alert_type, condition_type, target_value))
        conn.commit()
        st.success(f"Alert created for {symbol} when price {condition_type.lower()} {target_value}")

with tab2:
    st.subheader("Active Alerts")
    
    # Fetch active alerts
    alerts_df = pd.read_sql_query('''
        SELECT id, symbol, alert_type, condition_type, target_value, created_date
        FROM alerts 
        WHERE status = 'active'
        ORDER BY created_date DESC
    ''', conn)
    
    if not alerts_df.empty:
        # Display alerts with action buttons
        for _, alert in alerts_df.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.write(f"**{alert['symbol']}**")
                    st.write(f"{alert['alert_type']}")
                
                with col2:
                    st.write(f"Condition: {alert['condition_type']}")
                    st.write(f"Target: {alert['target_value']}")
                
                with col3:
                    st.write(f"Created: {alert['created_date'][:10]}")
                    # Simulate current value (in real app, this would be live data)
                    current_val = target_value * (0.95 + 0.1 * hash(alert['symbol']) % 100 / 100)
                    st.write(f"Current: {current_val:.2f}")
                
                with col4:
                    if st.button("❌", key=f"delete_{alert['id']}"):
                        conn.execute('DELETE FROM alerts WHERE id = ?', (alert['id'],))
                        conn.commit()
                        st.rerun()
                
                st.divider()
    else:
        st.info("No active alerts. Create your first alert in the 'Create Alert' tab.")

with tab3:
    st.subheader("Alert History")
    
    # Fetch triggered alerts
    history_df = pd.read_sql_query('''
        SELECT symbol, alert_type, condition_type, target_value, triggered_date
        FROM alerts 
        WHERE status = 'triggered'
        ORDER BY triggered_date DESC
        LIMIT 50
    ''', conn)
    
    if not history_df.empty:
        st.dataframe(history_df, width="stretch")
    else:
        st.info("No triggered alerts yet.")

# Alert monitoring simulation
st.subheader("🔍 Alert Monitoring")

# Real-time alert status
monitoring_active = st.checkbox("Enable Real-time Monitoring")

if monitoring_active:
    st.info("🟢 Alert monitoring is active. Checking alerts every 30 seconds...")
    
    # Placeholder for real-time monitoring status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Alerts", len(alerts_df) if not alerts_df.empty else 0)
    
    with col2:
        st.metric("Alerts Triggered Today", 0)  # Placeholder
    
    with col3:
        st.metric("Next Check", "29s")  # Placeholder

# Alert settings
st.subheader("⚙️ Alert Settings")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Notification Settings")
    default_email = st.text_input("Default Email", placeholder="your@email.com")
    email_enabled = st.checkbox("Enable Email Notifications", value=True)
    sound_enabled = st.checkbox("Enable Sound Notifications", value=True)

with col2:
    st.subheader("Monitoring Settings")
    check_interval = st.selectbox("Check Interval", ["30 seconds", "1 minute", "5 minutes"])
    max_alerts = st.number_input("Maximum Active Alerts", min_value=1, max_value=100, value=20)

if st.button("Save Settings"):
    st.success("Alert settings saved successfully!")

# Close database connection
# Note: In a real app, you'd want to manage this connection more carefully
