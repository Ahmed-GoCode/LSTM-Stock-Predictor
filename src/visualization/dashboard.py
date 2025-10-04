"""
Streamlit Dashboard for Stock Price Predictor
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Also add the src directory to path
src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

try:
    # Try different import approaches
    try:
        from src.main import StockPredictor
        from src.config.config import config
        from src.utils.exceptions import StockPredictorError
    except ImportError:
        # Alternative import method
        sys.path.insert(0, os.path.join(project_root, 'src'))
        from main import StockPredictor
        from config.config import config
        from utils.exceptions import StockPredictorError
        
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure you're running from the project root directory")
    st.error(f"Current working directory: {os.getcwd()}")
    st.error(f"Project root: {project_root}")
    st.error(f"Python path: {sys.path}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

def initialize_predictor():
    """Initialize the predictor with API key if provided"""
    alpha_vantage_key = st.session_state.get('alpha_vantage_key', '')
    if alpha_vantage_key:
        return StockPredictor(alpha_vantage_key=alpha_vantage_key)
    else:
        return StockPredictor()

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by LSTM Neural Networks")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        st.subheader("API Settings")
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API Key (Optional)",
            type="password",
            help="Enter your Alpha Vantage API key for additional data sources"
        )
        st.session_state.alpha_vantage_key = alpha_vantage_key
        
        # Initialize predictor
        if st.button("Initialize Predictor"):
            try:
                st.session_state.predictor = initialize_predictor()
                st.success("Predictor initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize predictor: {e}")
        
        st.divider()
        
        # Model Configuration
        st.subheader("Model Settings")
        
        # LSTM Configuration
        st.write("**LSTM Architecture**")
        lstm_units_str = st.text_input("LSTM Units (comma-separated)", value="50,50")
        try:
            lstm_units = [int(x.strip()) for x in lstm_units_str.split(',')]
        except:
            lstm_units = [50, 50]
            
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        sequence_length = st.slider("Sequence Length", 30, 120, 60, 5)
        
        # Training Configuration
        st.write("**Training Settings**")
        epochs = st.slider("Epochs", 50, 500, 100, 10)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
        
        # Update config
        if st.button("Update Model Config"):
            config.model.lstm_units = lstm_units
            config.model.dropout_rate = dropout_rate
            config.model.learning_rate = learning_rate
            config.model.sequence_length = sequence_length
            config.model.epochs = epochs
            config.model.batch_size = batch_size
            config.model.validation_split = validation_split
            st.success("Configuration updated!")
    
    # Main content
    tabs = st.tabs(["🏠 Home", "🔧 Train Model", "🔮 Predictions", "📊 Backtesting", "📈 Visualization", "📋 Model Info"])
    
    with tabs[0]:
        show_home_page()
    
    with tabs[1]:
        show_training_page()
    
    with tabs[2]:
        show_prediction_page()
    
    with tabs[3]:
        show_backtesting_page()
    
    with tabs[4]:
        show_visualization_page()
    
    with tabs[5]:
        show_model_info_page()

def show_home_page():
    """Show the home page"""
    st.header("Welcome to Stock Price Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Features")
        st.markdown("""
        - **LSTM Neural Networks**: Advanced deep learning for time series prediction
        - **Multiple Data Sources**: Yahoo Finance and Alpha Vantage integration
        - **Technical Indicators**: 20+ built-in technical analysis indicators
        - **Real-time Predictions**: Make future price predictions with confidence intervals
        - **Backtesting**: Evaluate model performance on historical data
        - **Risk Assessment**: Comprehensive risk metrics and analysis
        - **Interactive Visualizations**: Beautiful charts and dashboards
        - **Export Capabilities**: Save predictions and reports
        """)
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Initialize**: Click 'Initialize Predictor' in the sidebar
        2. **Configure**: Adjust model settings as needed
        3. **Train**: Go to 'Train Model' tab and train on your stock
        4. **Predict**: Use 'Predictions' tab to forecast future prices
        5. **Analyze**: Explore results in 'Visualization' tab
        """)
        
        # Quick stats
        if st.session_state.predictor:
            st.success("✅ Predictor Ready")
        else:
            st.warning("⚠️ Predictor Not Initialized")
            
        if st.session_state.trained_model:
            st.success("✅ Model Trained")
        else:
            st.warning("⚠️ No Model Trained")
    
    # Recent activity
    st.subheader("📊 Recent Activity")
    if st.session_state.training_history:
        df_history = pd.DataFrame(st.session_state.training_history)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No recent training activity")

def show_training_page():
    """Show the training page"""
    st.header("🔧 Train LSTM Model")
    
    if not st.session_state.predictor:
        st.warning("Please initialize the predictor first in the sidebar.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock selection
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, GOOGL, TSLA)")
        
        # Training parameters
        period = st.selectbox("Training Period", ["1y", "2y", "5y", "max"], index=1)
        
        model_name = st.text_input("Model Name (Optional)", help="Custom name for saving the model")
        
        save_model = st.checkbox("Save Model After Training", value=True)
        
    with col2:
        st.subheader("Training Status")
        if st.session_state.trained_model:
            st.success("Model Trained ✅")
            model_info = st.session_state.predictor.get_model_summary()
            if model_info.get('current_symbol'):
                st.write(f"**Symbol:** {model_info['current_symbol']}")
            if model_info.get('training_time'):
                st.write(f"**Training Time:** {model_info['training_time']:.1f}s")
        else:
            st.info("No model trained yet")
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        if not symbol:
            st.error("Please enter a stock symbol")
            return
        
        with st.spinner(f"Training model for {symbol}..."):
            try:
                # Training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Start training
                status_text.text("Fetching data...")
                progress_bar.progress(20)
                
                result = st.session_state.predictor.train(
                    symbol=symbol.upper(),
                    period=period,
                    save_model=save_model,
                    model_name=model_name
                )
                
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                # Update session state
                st.session_state.trained_model = result
                
                # Add to history
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol.upper(),
                    "period": period,
                    "rmse": result.model_metrics.rmse,
                    "mae": result.model_metrics.mae,
                    "directional_accuracy": result.model_metrics.directional_accuracy,
                    "training_time": result.training_time
                }
                st.session_state.training_history.append(history_entry)
                
                # Show results
                st.success("Training completed successfully!")
                
                # Metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{result.model_metrics.rmse:.6f}")
                with col2:
                    st.metric("MAE", f"{result.model_metrics.mae:.6f}")
                with col3:
                    st.metric("MAPE", f"{result.model_metrics.mape:.2f}%")
                with col4:
                    st.metric("Directional Accuracy", f"{result.model_metrics.directional_accuracy:.1f}%")
                
                # Training details
                with st.expander("Training Details"):
                    st.json({
                        "Model Config": result.model_config,
                        "Preprocessing Info": result.preprocessing_info,
                        "Training Time": f"{result.training_time:.2f} seconds"
                    })
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)

def show_prediction_page():
    """Show the prediction page"""
    st.header("🔮 Stock Price Predictions")
    
    if not st.session_state.predictor:
        st.warning("Please initialize the predictor first.")
        return
    
    if not st.session_state.trained_model:
        st.warning("Please train a model first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction parameters
        symbol = st.text_input("Stock Symbol", value=st.session_state.trained_model.symbol if st.session_state.trained_model else "")
        days = st.slider("Prediction Horizon (Days)", 1, 90, 30)
        confidence_intervals = st.checkbox("Include Confidence Intervals", value=True)
        
    with col2:
        st.subheader("Prediction Settings")
        st.write(f"**Current Model:** {st.session_state.trained_model.symbol if st.session_state.trained_model else 'None'}")
        st.write(f"**Model Accuracy:** {st.session_state.trained_model.model_metrics.directional_accuracy:.1f}%" if st.session_state.trained_model else "N/A")
    
    # Prediction button
    if st.button("🎯 Generate Predictions", type="primary"):
        if not symbol:
            st.error("Please enter a stock symbol")
            return
        
        with st.spinner("Generating predictions..."):
            try:
                result = st.session_state.predictor.predict(
                    symbol=symbol.upper(),
                    days=days,
                    confidence_intervals=confidence_intervals
                )
                
                st.session_state.prediction_result = result
                
                # Show prediction summary
                st.success("Predictions generated successfully!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${result.metrics['last_price']:.2f}")
                with col2:
                    st.metric("Predicted Final Price", f"${result.predictions[-1]:.2f}")
                with col3:
                    st.metric("Expected Return", f"{result.metrics['predicted_return']:.2f}%")
                with col4:
                    st.metric("Trend", result.metrics['trend'].title())
                
                # Prediction chart
                fig = create_prediction_chart(result)
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction table
                st.subheader("📋 Prediction Data")
                prediction_df = create_prediction_dataframe(result)
                st.dataframe(prediction_df, use_container_width=True)
                
                # Export options
                st.subheader("📥 Export Predictions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Export CSV"):
                        csv = prediction_df.to_csv(index=False)
                        st.download_button("Download CSV", csv, f"{symbol}_predictions.csv", "text/csv")
                
                with col2:
                    if st.button("Export JSON"):
                        json_data = prediction_df.to_json(orient="records", date_format="iso")
                        st.download_button("Download JSON", json_data, f"{symbol}_predictions.json", "application/json")
                
                with col3:
                    if st.button("Export Excel"):
                        # Note: This would need xlsxwriter or openpyxl
                        st.info("Excel export requires additional dependencies")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

def show_backtesting_page():
    """Show the backtesting page"""
    st.header("📊 Model Backtesting")
    
    if not st.session_state.predictor:
        st.warning("Please initialize the predictor first.")
        return
    
    if not st.session_state.trained_model:
        st.warning("Please train a model first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Backtest Configuration")
        symbol = st.text_input("Symbol", value=st.session_state.trained_model.symbol)
        
        # Date range
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        with col_end:
            end_date = st.date_input("End Date", value=datetime.now() - timedelta(days=30))
        
        prediction_horizon = st.slider("Prediction Horizon (Days)", 1, 30, 5)
        
    with col2:
        st.subheader("Expected Results")
        st.info("""
        **Backtesting will evaluate:**
        - Prediction accuracy on historical data
        - Trading performance metrics
        - Risk-adjusted returns
        - Directional accuracy
        """)
    
    if st.button("🔄 Run Backtest", type="primary"):
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
        
        with st.spinner("Running backtest..."):
            try:
                result = st.session_state.predictor.backtest(
                    symbol=symbol.upper(),
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    prediction_horizon=prediction_horizon
                )
                
                # Display results
                st.success("Backtesting completed!")
                
                # Performance metrics
                eval_metrics = result.get('evaluation', {}).get('basic_metrics', {})
                trading_metrics = result.get('evaluation', {}).get('trading_metrics', {})
                
                if eval_metrics:
                    st.subheader("📈 Prediction Accuracy")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSE", f"{eval_metrics.get('rmse', 0):.6f}")
                    with col2:
                        st.metric("MAE", f"{eval_metrics.get('mae', 0):.6f}")
                    with col3:
                        st.metric("MAPE", f"{eval_metrics.get('mape', 0):.2f}%")
                    with col4:
                        st.metric("Directional Accuracy", f"{eval_metrics.get('directional_accuracy', 0):.1f}%")
                
                if trading_metrics:
                    st.subheader("💰 Trading Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{trading_metrics.get('total_return', 0):.2f}%")
                    with col2:
                        st.metric("Sharpe Ratio", f"{trading_metrics.get('sharpe_ratio', 0):.3f}")
                    with col3:
                        st.metric("Max Drawdown", f"{trading_metrics.get('max_drawdown', 0):.2f}%")
                    with col4:
                        st.metric("Win Rate", f"{trading_metrics.get('win_rate', 0):.1f}%")
                
                # Backtest visualization
                if 'backtest_data' in result:
                    fig = create_backtest_chart(result['backtest_data'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                with st.expander("Detailed Backtest Results"):
                    st.json(result, expanded=False)
                
            except Exception as e:
                st.error(f"Backtesting failed: {e}")
                st.exception(e)

def show_visualization_page():
    """Show advanced visualizations"""
    st.header("📈 Advanced Visualizations")
    
    if not st.session_state.prediction_result:
        st.warning("Please generate predictions first to see visualizations.")
        return
    
    result = st.session_state.prediction_result
    
    # Visualization options
    viz_type = st.selectbox(
        "Visualization Type",
        ["Price Prediction", "Confidence Intervals", "Return Distribution", "Risk Metrics"]
    )
    
    if viz_type == "Price Prediction":
        fig = create_prediction_chart(result)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Confidence Intervals":
        if result.confidence_intervals:
            fig = create_confidence_chart(result)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Confidence intervals not available for this prediction.")
            
    elif viz_type == "Return Distribution":
        fig = create_return_distribution_chart(result)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Risk Metrics":
        show_risk_metrics(result)

def show_model_info_page():
    """Show model information and statistics"""
    st.header("📋 Model Information")
    
    if not st.session_state.predictor:
        st.warning("Please initialize the predictor first.")
        return
    
    model_summary = st.session_state.predictor.get_model_summary()
    
    if model_summary.get('status') == 'not_trained':
        st.warning("No model trained yet.")
        return
    
    # Model overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏗️ Model Architecture")
        st.write(f"**Type:** LSTM Neural Network")
        st.write(f"**Symbol:** {model_summary.get('current_symbol', 'N/A')}")
        st.write(f"**Features:** {model_summary.get('feature_count', 'N/A')}")
        st.write(f"**Sequence Length:** {model_summary.get('sequence_length', 'N/A')}")
        st.write(f"**Training Time:** {model_summary.get('training_time', 0):.1f} seconds")
    
    with col2:
        st.subheader("📊 Performance Metrics")
        if st.session_state.trained_model:
            metrics = st.session_state.trained_model.model_metrics
            st.write(f"**RMSE:** {metrics.rmse:.6f}")
            st.write(f"**MAE:** {metrics.mae:.6f}")
            st.write(f"**MAPE:** {metrics.mape:.2f}%")
            st.write(f"**R²:** {metrics.r2_score:.4f}")
            st.write(f"**Directional Accuracy:** {metrics.directional_accuracy:.1f}%")
    
    # Model configuration
    with st.expander("⚙️ Model Configuration"):
        st.json(model_summary.get('model_config', {}))
    
    # Preprocessing information
    with st.expander("🔧 Preprocessing Information"):
        st.json(model_summary.get('preprocessing_info', {}))
    
    # Training history
    if st.session_state.training_history:
        st.subheader("📜 Training History")
        df_history = pd.DataFrame(st.session_state.training_history)
        st.dataframe(df_history, use_container_width=True)

def create_prediction_chart(result):
    """Create a prediction chart"""
    fig = go.Figure()
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=result.dates,
        y=result.predictions,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='blue', width=2)
    ))
    
    # Confidence intervals
    if result.confidence_intervals and 'lower' in result.confidence_intervals:
        fig.add_trace(go.Scatter(
            x=result.dates,
            y=result.confidence_intervals['upper'],
            mode='lines',
            name='Upper Confidence',
            line=dict(color='lightblue', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=result.dates,
            y=result.confidence_intervals['lower'],
            mode='lines',
            name='Confidence Interval',
            line=dict(color='lightblue', width=1),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)'
        ))
    
    fig.update_layout(
        title=f"Stock Price Prediction for {result.symbol}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_confidence_chart(result):
    """Create confidence interval visualization"""
    if not result.confidence_intervals:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Predictions with Confidence Intervals', 'Prediction Uncertainty'),
        vertical_spacing=0.1
    )
    
    # Main prediction chart
    fig.add_trace(go.Scatter(
        x=result.dates,
        y=result.predictions,
        mode='lines',
        name='Prediction',
        line=dict(color='blue')
    ), row=1, col=1)
    
    if 'upper' in result.confidence_intervals:
        fig.add_trace(go.Scatter(
            x=result.dates,
            y=result.confidence_intervals['upper'],
            mode='lines',
            name='Upper 95%',
            line=dict(color='red', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=result.dates,
            y=result.confidence_intervals['lower'],
            mode='lines',
            name='Lower 95%',
            line=dict(color='red', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ), row=1, col=1)
    
    # Uncertainty plot
    if 'std' in result.confidence_intervals:
        fig.add_trace(go.Scatter(
            x=result.dates,
            y=result.confidence_intervals['std'],
            mode='lines',
            name='Prediction Std',
            line=dict(color='orange')
        ), row=2, col=1)
    
    fig.update_layout(height=700, title_text=f"Prediction Analysis for {result.symbol}")
    return fig

def create_return_distribution_chart(result):
    """Create return distribution chart"""
    returns = np.diff(result.predictions) / result.predictions[:-1] * 100
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Daily Return Distribution', 'Cumulative Returns'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=20,
        name='Daily Returns (%)',
        opacity=0.7
    ), row=1, col=1)
    
    # Cumulative returns
    cumulative_returns = np.cumprod(1 + returns/100) - 1
    fig.add_trace(go.Scatter(
        x=result.dates[1:],
        y=cumulative_returns * 100,
        mode='lines',
        name='Cumulative Return (%)',
        line=dict(color='green')
    ), row=1, col=2)
    
    fig.update_layout(height=400, title_text=f"Return Analysis for {result.symbol}")
    return fig

def create_backtest_chart(backtest_data):
    """Create backtest visualization"""
    fig = go.Figure()
    
    dates = backtest_data.get('prediction_dates', [])
    predictions = backtest_data.get('predicted_values', [])
    actuals = backtest_data.get('actual_values', [])
    
    if len(dates) > 0:
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=actuals,
            mode='markers',
            name='Actual',
            marker=dict(color='red', size=6)
        ))
    
    fig.update_layout(
        title="Backtest Results: Predictions vs Actual",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    
    return fig

def create_prediction_dataframe(result):
    """Create DataFrame for prediction data"""
    data = {
        'Date': result.dates,
        'Predicted_Price': result.predictions,
    }
    
    if result.confidence_intervals:
        if 'lower' in result.confidence_intervals:
            data['Lower_95%'] = result.confidence_intervals['lower']
            data['Upper_95%'] = result.confidence_intervals['upper']
        if 'std' in result.confidence_intervals:
            data['Prediction_Std'] = result.confidence_intervals['std']
    
    return pd.DataFrame(data)

def show_risk_metrics(result):
    """Show risk metrics and analysis"""
    st.subheader("⚠️ Risk Analysis")
    
    predictions = result.predictions
    volatility = np.std(predictions)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Volatility", f"{volatility:.4f}")
    
    with col2:
        max_loss = np.min(np.diff(predictions) / predictions[:-1]) * 100
        st.metric("Max Daily Loss", f"{max_loss:.2f}%")
    
    with col3:
        max_gain = np.max(np.diff(predictions) / predictions[:-1]) * 100
        st.metric("Max Daily Gain", f"{max_gain:.2f}%")
    
    returns = np.diff(predictions) / predictions[:-1]
    
    st.subheader("📊 Risk-Return Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Expected Return:** {np.mean(returns) * 100:.2f}% per day")
        st.write(f"**Return Volatility:** {np.std(returns) * 100:.2f}%")
        st.write(f"**Skewness:** {pd.Series(returns).skew():.3f}")
        st.write(f"**Kurtosis:** {pd.Series(returns).kurtosis():.3f}")
    
    with col2:
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        st.write(f"**Value at Risk (95%):** {var_95:.2f}%")
        st.write(f"**Value at Risk (99%):** {var_99:.2f}%")
        
        if np.std(returns) > 0:
            sharpe_proxy = np.mean(returns) / np.std(returns)
            st.write(f"**Sharpe Proxy:** {sharpe_proxy:.3f}")

if __name__ == "__main__":
    main()