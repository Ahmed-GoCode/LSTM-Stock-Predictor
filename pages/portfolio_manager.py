import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import shared utilities
from shared_utils import (ASSETS, fetch_real_data, get_db_connection,
                         export_data_as_csv, export_data_as_excel)

# Portfolio Manager Page
st.header("💼 Portfolio Manager")

# Initialize portfolio database connection
def get_portfolio_data():
    """Fetch portfolio data from database"""
    conn = sqlite3.connect('trading_platform.db')
    df = pd.read_sql_query("SELECT * FROM portfolio", conn)
    conn.close()
    return df

def add_to_portfolio(symbol, quantity, price):
    """Add position to portfolio"""
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO portfolio (symbol, quantity, avg_price, date_added)
        VALUES (?, ?, ?, ?)
    """, (symbol, quantity, price, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

# Portfolio overview
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Portfolio Overview")
    
    # Get portfolio data
    portfolio_df = get_portfolio_data()
    
    if not portfolio_df.empty:
        # Calculate current values
        portfolio_value = 0
        portfolio_data = []
        
        for _, row in portfolio_df.iterrows():
            symbol = row['symbol']
            quantity = row['quantity']
            avg_price = row['avg_price']
            
            # Get current price
            current_data = fetch_real_data(symbol, "1d", "1d")
            if current_data is not None and len(current_data) > 0:
                current_price = current_data['Close'].iloc[-1]
                position_value = quantity * current_price
                pnl = (current_price - avg_price) * quantity
                pnl_pct = ((current_price - avg_price) / avg_price) * 100
                
                portfolio_data.append({
                    'Symbol': symbol,
                    'Quantity': quantity,
                    'Avg Price': avg_price,
                    'Current Price': current_price,
                    'Position Value': position_value,
                    'P&L': pnl,
                    'P&L %': pnl_pct
                })
                
                portfolio_value += position_value
        
        if portfolio_data:
            portfolio_display_df = pd.DataFrame(portfolio_data)
            
            # Portfolio metrics
            total_pnl = portfolio_display_df['P&L'].sum()
            avg_pnl_pct = portfolio_display_df['P&L %'].mean()
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            with col_p1:
                st.metric("💰 Total Value", f"${portfolio_value:,.2f}")
            
            with col_p2:
                st.metric("📈 Total P&L", f"${total_pnl:,.2f}", f"{avg_pnl_pct:+.2f}%")
            
            with col_p3:
                st.metric("📊 Positions", len(portfolio_data))
            
            with col_p4:
                best_performer = portfolio_display_df.loc[portfolio_display_df['P&L %'].idxmax(), 'Symbol']
                st.metric("🏆 Best Performer", best_performer)
            
            # Portfolio table
            st.dataframe(portfolio_display_df, width="stretch")
            
            # Portfolio allocation chart
            st.subheader("🥧 Portfolio Allocation")
            fig_pie = px.pie(
                values=portfolio_display_df['Position Value'],
                names=portfolio_display_df['Symbol'],
                title="Portfolio Allocation by Value"
            )
            fig_pie.update_layout(template='plotly_dark')
            st.plotly_chart(fig_pie, config={"displayModeBar": True})
            
        else:
            st.info("No current price data available for portfolio positions")
    
    else:
        st.info("📝 Your portfolio is empty. Add some positions to get started!")

with col2:
    st.subheader("➕ Add Position")
    
    # Add new position form
    with st.form("add_position"):
        new_symbol = st.text_input("Symbol", placeholder="e.g., AAPL")
        new_quantity = st.number_input("Quantity", min_value=0.001, value=1.0, step=0.001)
        new_price = st.number_input("Purchase Price", min_value=0.01, value=100.0, step=0.01)
        
        if st.form_submit_button("🚀 Add Position", width="stretch"):
            if new_symbol:
                add_to_portfolio(new_symbol.upper(), new_quantity, new_price)
                st.success(f"Added {new_quantity} shares of {new_symbol.upper()} at ${new_price}")
                st.rerun()
            else:
                st.error("Please enter a symbol")
    
    st.markdown("---")
    
    # Portfolio analytics
    st.subheader("📊 Portfolio Analytics")
    
    if not portfolio_df.empty:
        # Risk metrics
        st.write("**Risk Metrics:**")
        
        # Get historical data for correlation analysis
        portfolio_symbols = portfolio_df['symbol'].unique()
        
        if len(portfolio_symbols) > 1:
            correlation_data = {}
            
            for symbol in portfolio_symbols:
                hist_data = fetch_real_data(symbol, "3mo", "1d")
                if hist_data is not None:
                    correlation_data[symbol] = hist_data['Close'].pct_change().dropna()
            
            if len(correlation_data) > 1:
                correlation_df = pd.DataFrame(correlation_data)
                correlation_matrix = correlation_df.corr()
                
                # Display correlation heatmap
                fig_corr = px.imshow(
                    correlation_matrix,
                    title="Portfolio Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                fig_corr.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig_corr, config={"displayModeBar": True})
        
        # Performance metrics
        st.write("**Performance Metrics:**")
        
        # Calculate portfolio beta (if SP500 data available)
        try:
            sp500_data = fetch_real_data("^GSPC", "3mo", "1d")
            if sp500_data is not None:
                sp500_returns = sp500_data['Close'].pct_change().dropna()
                
                portfolio_betas = []
                for symbol in portfolio_symbols:
                    hist_data = fetch_real_data(symbol, "3mo", "1d")
                    if hist_data is not None:
                        stock_returns = hist_data['Close'].pct_change().dropna()
                        # Align dates
                        aligned_data = pd.concat([stock_returns, sp500_returns], axis=1, join='inner')
                        if len(aligned_data) > 30:
                            beta = aligned_data.cov().iloc[0, 1] / aligned_data.var().iloc[1]
                            portfolio_betas.append(beta)
                
                if portfolio_betas:
                    avg_beta = np.mean(portfolio_betas)
                    st.metric("📊 Average Beta", f"{avg_beta:.2f}")
        
        except:
            st.info("Beta calculation unavailable")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("📤 Export Portfolio", width="stretch"):
        if not portfolio_df.empty:
            csv = portfolio_df.to_csv(index=False)
            st.download_button(
                label="Download Portfolio CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    if st.button("🔄 Refresh Prices", width="stretch"):
        st.rerun()
    
    if st.button("⚠️ Clear Portfolio", width="stretch"):
        # Add confirmation dialog in real implementation
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM portfolio")
        conn.commit()
        conn.close()
        st.success("Portfolio cleared")
        st.rerun()

# Portfolio performance over time
if not portfolio_df.empty:
    st.subheader("📈 Portfolio Performance")
    
    # Simulate portfolio performance (in real app, would track historical positions)
    performance_data = []
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    
    # This is a simplified simulation - real implementation would track actual position history
    base_value = 10000  # Starting portfolio value
    for i, date in enumerate(dates):
        # Simulate some portfolio growth with volatility
        daily_return = np.random.normal(0.0005, 0.02)  # 0.05% average daily return with 2% volatility
        base_value *= (1 + daily_return)
        performance_data.append({'Date': date, 'Portfolio Value': base_value})
    
    performance_df = pd.DataFrame(performance_data)
    
    fig_performance = px.line(
        performance_df, 
        x='Date', 
        y='Portfolio Value',
        title='Portfolio Performance Over Time'
    )
    fig_performance.update_layout(template='plotly_dark')
    st.plotly_chart(fig_performance, config={"displayModeBar": True})
    
    # 🆕 Portfolio Export Functionality
    st.subheader("📤 Export Portfolio Data")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📊 Export Portfolio CSV"):
            if not portfolio_df.empty:
                export_data_as_csv(portfolio_df, "portfolio_holdings")
            else:
                st.warning("No portfolio data to export")
    
    with export_col2:
        if st.button("📗 Export Portfolio Excel"):
            if not portfolio_df.empty:
                export_data_as_excel(portfolio_df, "portfolio_holdings")
            else:
                st.warning("No portfolio data to export")
    
    with export_col3:
        if st.button("📈 Export Performance Data"):
            if 'performance_df' in locals():
                export_data_as_csv(performance_df, "portfolio_performance")
            else:
                st.warning("Generate performance chart first")
