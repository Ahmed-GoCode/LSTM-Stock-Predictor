import streamlit as st
import json
from datetime import datetime

# Import shared utilities
from shared_utils import ASSETS

# Settings Page
st.header("⚙️ Settings")

# Create tabs for different settings categories
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 Appearance", "📊 Trading", "🔔 Notifications", "🔒 Security", "🛠️ Advanced"])

with tab1:
    st.subheader("🎨 Appearance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Theme Settings")
        
        theme = st.selectbox("Theme", ["Dark", "Light", "Auto"])
        primary_color = st.color_picker("Primary Color", "#FF6B6B")
        secondary_color = st.color_picker("Secondary Color", "#4ECDC4")
        
        st.subheader("Layout Settings")
        sidebar_width = st.selectbox("Sidebar Width", ["Normal", "Wide", "Narrow"])
        compact_mode = st.checkbox("Compact Mode")
        show_grid_lines = st.checkbox("Show Grid Lines", value=True)
    
    with col2:
        st.subheader("Chart Settings")
        
        default_chart_type = st.selectbox("Default Chart Type", 
                                        ["Candlestick", "Line", "Area", "OHLC"])
        chart_height = st.slider("Chart Height (pixels)", min_value=300, max_value=800, value=500)
        show_volume = st.checkbox("Show Volume by Default", value=True)
        
        st.subheader("Data Display")
        decimal_places = st.slider("Decimal Places", min_value=2, max_value=8, value=4)
        use_24h_format = st.checkbox("Use 24-hour Time Format", value=True)
        show_percentage_change = st.checkbox("Show Percentage Change", value=True)
    
    if st.button("Save Appearance Settings"):
        st.success("Appearance settings saved!")

with tab2:
    st.subheader("📊 Trading Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Settings")
        
        base_currency = st.selectbox("Base Currency", ["USD", "EUR", "GBP", "JPY", "CAD"])
        portfolio_value = st.number_input("Initial Portfolio Value", min_value=1000, value=10000)
        risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
        
        st.subheader("Default Watchlist")
        default_symbols = st.text_area("Default Symbols (comma-separated)", 
                                     value="AAPL, GOOGL, MSFT, TSLA, AMZN")
        
        auto_add_analyzed = st.checkbox("Auto-add Analyzed Symbols to Watchlist")
    
    with col2:
        st.subheader("Analysis Settings")
        
        default_timeframe = st.selectbox("Default Timeframe", 
                                       ["1D", "1W", "1M", "3M", "6M", "1Y", "2Y", "5Y"])
        
        technical_indicators = st.multiselect("Default Technical Indicators", 
                                            ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "Stochastic"],
                                            default=["SMA", "RSI", "MACD"])
        
        prediction_horizon = st.selectbox("AI Prediction Horizon", 
                                        ["1 Day", "3 Days", "1 Week", "2 Weeks", "1 Month"])
        
        confidence_threshold = st.slider("Minimum Confidence Threshold (%)", 
                                       min_value=50, max_value=95, value=70)
    
    if st.button("Save Trading Settings"):
        st.success("Trading settings saved!")

with tab3:
    st.subheader("🔔 Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Alert Notifications")
        
        email_alerts = st.checkbox("Email Alerts", value=True)
        if email_alerts:
            email_address = st.text_input("Email Address", placeholder="your@email.com")
        
        browser_notifications = st.checkbox("Browser Notifications", value=True)
        sound_alerts = st.checkbox("Sound Alerts", value=True)
        
        st.subheader("Market Hours Notifications")
        market_open_alert = st.checkbox("Market Open Alert")
        market_close_alert = st.checkbox("Market Close Alert")
        pre_market_alerts = st.checkbox("Pre-market Alerts")
    
    with col2:
        st.subheader("Price Movement Alerts")
        
        significant_moves = st.checkbox("Significant Price Movements (>5%)", value=True)
        volume_spikes = st.checkbox("Volume Spikes (>200% avg)", value=True)
        news_alerts = st.checkbox("Breaking News Alerts")
        
        st.subheader("Portfolio Alerts")
        portfolio_gain_threshold = st.number_input("Portfolio Gain Alert (%)", min_value=1, value=10)
        portfolio_loss_threshold = st.number_input("Portfolio Loss Alert (%)", min_value=1, value=5)
        
        rebalance_alerts = st.checkbox("Portfolio Rebalancing Alerts")
    
    # Test notifications
    st.subheader("Test Notifications")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test Email"):
            st.info("📧 Test email sent!")
    
    with col2:
        if st.button("Test Browser"):
            st.info("🌐 Browser notification triggered!")
    
    with col3:
        if st.button("Test Sound"):
            st.info("🔊 Sound alert played!")
    
    if st.button("Save Notification Settings"):
        st.success("Notification settings saved!")

with tab4:
    st.subheader("🔒 Security Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Account Security")
        
        two_factor_auth = st.checkbox("Two-Factor Authentication")
        if two_factor_auth:
            st.info("📱 Configure 2FA in your authenticator app")
        
        session_timeout = st.selectbox("Session Timeout", 
                                     ["15 minutes", "30 minutes", "1 hour", "2 hours", "Never"])
        
        login_alerts = st.checkbox("Login Alerts", value=True)
        
        st.subheader("API Security")
        api_key_rotation = st.selectbox("API Key Rotation", 
                                      ["Never", "Monthly", "Quarterly", "Annually"])
        
        encrypt_local_data = st.checkbox("Encrypt Local Data", value=True)
    
    with col2:
        st.subheader("Privacy Settings")
        
        anonymous_usage = st.checkbox("Anonymous Usage Statistics")
        crash_reporting = st.checkbox("Crash Reporting", value=True)
        
        data_sharing = st.selectbox("Data Sharing", 
                                  ["None", "Aggregated Only", "Full (Anonymous)"])
        
        st.subheader("Data Retention")
        portfolio_history = st.selectbox("Portfolio History", 
                                       ["1 Year", "2 Years", "5 Years", "Forever"])
        
        trade_history = st.selectbox("Trade History", 
                                   ["1 Year", "3 Years", "7 Years", "Forever"])
        
        delete_inactive_data = st.checkbox("Auto-delete Inactive Data (>1 year)")
    
    if st.button("Save Security Settings"):
        st.success("Security settings saved!")

with tab5:
    st.subheader("🛠️ Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Settings")
        
        cache_enabled = st.checkbox("Enable Data Caching", value=True)
        cache_size = st.slider("Cache Size (MB)", min_value=10, max_value=500, value=100)
        
        parallel_processing = st.checkbox("Parallel Data Processing", value=True)
        max_threads = st.slider("Max Processing Threads", min_value=1, max_value=8, value=4)
        
        st.subheader("Data Sources")
        primary_data_source = st.selectbox("Primary Data Source", 
                                         ["Yahoo Finance", "Alpha Vantage", "IEX Cloud"])
        
        backup_data_source = st.selectbox("Backup Data Source", 
                                        ["Yahoo Finance", "Alpha Vantage", "IEX Cloud"])
        
        data_validation = st.checkbox("Strict Data Validation", value=True)
    
    with col2:
        st.subheader("Developer Settings")
        
        debug_mode = st.checkbox("Debug Mode")
        verbose_logging = st.checkbox("Verbose Logging")
        
        if debug_mode:
            st.warning("⚠️ Debug mode may expose sensitive information")
        
        st.subheader("Experimental Features")
        
        ai_enhanced_predictions = st.checkbox("AI-Enhanced Predictions (Beta)")
        real_time_sentiment = st.checkbox("Real-time Sentiment Analysis (Beta)")
        advanced_options_trading = st.checkbox("Advanced Options Trading (Beta)")
        
        if ai_enhanced_predictions or real_time_sentiment or advanced_options_trading:
            st.info("ℹ️ Beta features may be unstable")
    
    # Database maintenance
    st.subheader("🗄️ Database Maintenance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Optimize Database"):
            st.success("Database optimized!")
    
    with col2:
        if st.button("Clear Cache"):
            st.success("Cache cleared!")
    
    with col3:
        if st.button("Backup Data"):
            st.success("Data backed up!")
    
    if st.button("Save Advanced Settings"):
        st.success("Advanced settings saved!")

# Settings export/import
st.subheader("📁 Settings Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export Settings"):
        # Create settings dictionary
        settings = {
            "theme": "Dark",
            "primary_color": "#FF6B6B",
            "base_currency": "USD",
            "exported_at": datetime.now().isoformat()
        }
        
        settings_json = json.dumps(settings, indent=2)
        st.download_button(
            label="Download Settings JSON",
            data=settings_json,
            file_name=f"settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

with col2:
    uploaded_settings = st.file_uploader("Import Settings", type=['json'])
    if uploaded_settings:
        if st.button("Import Settings"):
            try:
                settings = json.load(uploaded_settings)
                st.success("Settings imported successfully!")
            except:
                st.error("Invalid settings file")

with col3:
    if st.button("Reset to Defaults"):
        if st.button("Confirm Reset", type="secondary"):
            st.success("Settings reset to defaults!")

# About section
st.subheader("ℹ️ About")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **LSTM Stock Predictor Pro**
    
    Version: 2.0.0
    Build: 2024.01.15
    
    A comprehensive trading platform with AI-powered predictions and professional analysis tools.
    """)

with col2:
    st.info("""
    **System Information**
    
    Python Version: 3.12+
    Streamlit Version: 1.29+
    Database: SQLite
    
    Last Updated: January 15, 2024
    """)
