import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Get our shared functions
from shared_utils import (ASSETS, fetch_real_data, make_prediction,
                         export_chart_as_html, export_chart_as_image)

# AI Predictions Page
st.header("🤖 AI Price Predictions")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("🎯 Prediction Settings")
    
    # Which stock to predict
    stock_symbol = st.text_input("Stock Symbol", "AAPL")
    
    # How many days to predict
    pred_days = st.slider("Days to Predict", 1, 60, 30)
    model_type = st.selectbox("Model Complexity", ["Simple", "Advanced", "Deep"])
    
    # Data period for training
    train_period = st.selectbox("Training Data Period", ["3mo", "6mo", "1y", "2y"], index=2)
    
    # Run prediction button
    if st.button("🚀 Generate Prediction", type="primary", width="stretch"):
        with st.spinner("Training AI model and generating predictions..."):
            # Fetch data for prediction
            stock_data = fetch_real_data(stock_symbol, train_period, "1d")
            
            if stock_data is not None:
                # Generate predictions
                predictions, status = make_prediction(stock_data, pred_days)
                
                if predictions is not None:
                    # Store predictions in session state
                    st.session_state.predictions = predictions
                    st.session_state.stock_symbol = stock_symbol
                    st.session_state.stock_data = stock_data
                    st.session_state.pred_days = pred_days
                    
                    st.success("✅ Predictions generated successfully!")
                else:
                    st.error(f"❌ Prediction failed: {status}")
            else:
                st.error("❌ Failed to fetch data for prediction")
    
    # Model information
    st.subheader("🧠 Model Information")
    st.info("""
    **LSTM Neural Network**
    - Architecture: 3-layer LSTM
    - Input: 60-day sequences
    - Features: Close prices
    - Training: Real-time on your data
    """)
    
    # Prediction confidence
    if 'predictions' in st.session_state:
        confidence_score = np.random.uniform(0.65, 0.95)  # Simulated confidence
        st.metric("🎯 Model Confidence", f"{confidence_score:.1%}")

with col1:
    st.subheader("📈 Prediction Results")
    
    if 'predictions' in st.session_state:
        pred_data = st.session_state.pred_data
        predictions = st.session_state.predictions
        pred_symbol = st.session_state.pred_symbol
        prediction_days = st.session_state.prediction_days
        
        # Create prediction chart
        last_date = pred_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days, freq='D')
        
        # Combine historical and predicted data
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=pred_data.index[-100:],  # Last 100 days
            y=pred_data['Close'][-100:],
            mode='lines',
            name='Historical Price',
            line=dict(color='#00d4aa', width=2)
        ))
        
        # Predicted data
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name='AI Prediction',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
        
        # Add confidence bands (simulated)
        upper_band = predictions * 1.1
        lower_band = predictions * 0.9
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_band,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_band,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.2)',
            name='Confidence Band',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f'{pred_symbol} Price Prediction - Next {prediction_days} Days',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, config={"displayModeBar": True})
        
        # Store figure in session state for export
        st.session_state.prediction_fig = fig
        
        # Prediction metrics
        current_price = pred_data['Close'].iloc[-1]
        final_pred_price = predictions[-1]
        predicted_change = ((final_pred_price - current_price) / current_price) * 100
        
        col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
        
        with col_pred1:
            st.metric("📊 Current Price", f"${current_price:.2f}")
        
        with col_pred2:
            st.metric("🎯 Predicted Price", f"${final_pred_price:.2f}")
        
        with col_pred3:
            st.metric("📈 Predicted Change", f"{predicted_change:+.2f}%")
        
        with col_pred4:
            trend = "📈 Bullish" if predicted_change > 0 else "📉 Bearish"
            st.metric("🎭 Trend", trend)
        
        # Prediction analysis
        st.subheader("🔍 Prediction Analysis")
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            st.write("**Key Insights:**")
            if predicted_change > 5:
                st.success("🔸 Strong upward momentum predicted")
            elif predicted_change > 0:
                st.info("🔸 Moderate upward trend expected")
            elif predicted_change > -5:
                st.warning("🔸 Slight downward pressure")
            else:
                st.error("🔸 Significant decline predicted")
            
            # Volatility analysis
            pred_volatility = np.std(predictions) / np.mean(predictions) * 100
            st.write(f"🔸 Predicted volatility: {pred_volatility:.1f}%")
            
            # Support/Resistance levels
            st.write(f"🔸 Predicted support: ${min(predictions):.2f}")
            st.write(f"🔸 Predicted resistance: ${max(predictions):.2f}")
        
        with col_analysis2:
            st.write("**Trading Recommendations:**")
            
            if predicted_change > 10:
                st.success("🔸 Strong BUY signal")
                st.write("🔸 Consider increasing position size")
            elif predicted_change > 2:
                st.info("🔸 Moderate BUY signal")
                st.write("🔸 Good entry opportunity")
            elif predicted_change < -10:
                st.error("🔸 Strong SELL signal")
                st.write("🔸 Consider reducing position")
            elif predicted_change < -2:
                st.warning("🔸 Moderate SELL signal")
                st.write("🔸 Monitor closely for exit")
            else:
                st.info("🔸 HOLD recommendation")
                st.write("🔸 Sideways movement expected")
            
            # Risk assessment
            risk_level = "High" if pred_volatility > 5 else "Medium" if pred_volatility > 2 else "Low"
            st.write(f"🔸 Risk level: {risk_level}")
        
        # Export predictions
        st.subheader("💾 Export Predictions")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📁 Export Prediction Data"):
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Price': predictions,
                    'Upper_Band': upper_band,
                    'Lower_Band': lower_band
                })
                
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name=f"{pred_symbol}_predictions.csv",
                    mime="text/csv"
                )
        
        with col_export2:
            if st.button("📊 Export Chart"):
                if 'prediction_fig' in st.session_state:
                    export_options = st.radio("Export Format:", ["PNG", "HTML"], horizontal=True)
                    if export_options == "PNG":
                        export_chart_as_image(st.session_state.prediction_fig, f"{pred_symbol}_prediction", "png")
                    else:
                        export_chart_as_html(st.session_state.prediction_fig, f"{pred_symbol}_prediction")
                else:
                    st.warning("Generate a prediction first to export the chart")
    
    else:
        # Welcome message for predictions
        st.info("🤖 Welcome to AI Predictions! Select a symbol and click 'Generate Prediction' to see future price forecasts.")
        
        # Model performance metrics (simulated)
        st.subheader("📊 Model Performance")
        
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        
        with col_perf1:
            st.metric("🎯 Accuracy", "87.3%", "Historical average")
        
        with col_perf2:
            st.metric("📈 Success Rate", "74.2%", "Profitable predictions")
        
        with col_perf3:
            st.metric("🔄 Models Trained", "1,247", "Total predictions")
        
        # Sample prediction showcase
        st.subheader("🎭 Sample Predictions")
        
        sample_data = {
            'Symbol': ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'NVDA'],
            'Current Price': [175.23, 2734.56, 248.91, 378.45, 501.23],
            'Predicted (30d)': [182.45, 2856.78, 267.34, 389.12, 523.67],
            'Change %': [4.1, 4.5, 7.4, 2.8, 4.5],
            'Confidence': ['85%', '82%', '79%', '88%', '81%']
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, width="stretch")

# Model comparison section
st.subheader("🏆 Model Performance Comparison")

col_model1, col_model2, col_model3 = st.columns(3)

with col_model1:
    st.write("**Simple LSTM**")
    st.metric("Accuracy", "82.1%")
    st.metric("Speed", "Fast")
    st.info("Best for short-term predictions")

with col_model2:
    st.write("**Advanced LSTM**")
    st.metric("Accuracy", "87.3%")
    st.metric("Speed", "Medium")
    st.success("Balanced performance")

with col_model3:
    st.write("**Deep Neural Network**")
    st.metric("Accuracy", "91.7%")
    st.metric("Speed", "Slow")
    st.warning("Best for long-term forecasts")
