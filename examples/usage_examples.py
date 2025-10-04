"""
Example usage of the Stock Price Predictor
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import StockPredictor
from risk_assessment.risk_analyzer import RiskAssessment, calculate_risk_adjusted_returns
from export.results_exporter import ResultsExporter

def basic_prediction_example():
    """Basic stock prediction example"""
    print("=" * 60)
    print("BASIC STOCK PREDICTION EXAMPLE")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = StockPredictor()
        
        # Get prediction for Apple stock (30 days)
        print("Fetching data and training model for AAPL...")
        results = predictor.predict_stock("AAPL", days=30)
        
        # Display results
        print(f"\nPrediction Results for AAPL:")
        print(f"Current Price: ${results.get('current_price', 'N/A')}")
        print(f"30-day Prediction: ${results.get('prediction', 'N/A')}")
        print(f"Model Confidence: {results.get('confidence', 0):.2%}")
        print(f"Expected Return: {results.get('expected_return', 0):.2%}")
        
        return True
        
    except Exception as e:
        print(f"Error in basic prediction: {e}")
        return False

def comprehensive_analysis_example():
    """Comprehensive analysis workflow"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS EXAMPLE")
    print("=" * 60)
    
    try:
        symbol = "TSLA"
        predictor = StockPredictor()
        
        print(f"Running comprehensive analysis for {symbol}...")
        
        # 1. Fetch and analyze data
        print("\n1. Fetching historical data...")
        data = predictor.fetch_data(symbol, period="2y")
        print(f"   Data shape: {data.shape if data is not None else 'N/A'}")
        
        # 2. Train model
        print("\n2. Training LSTM model...")
        model_results = predictor.train_model(symbol)
        if model_results:
            print(f"   Training completed successfully")
            print(f"   Final Loss: {model_results.get('final_loss', 'N/A')}")
        
        # 3. Generate predictions
        print("\n3. Generating predictions...")
        predictions = predictor.predict_stock(symbol, days=60)
        if predictions:
            print(f"   Generated {len(predictions.get('predictions', []))} day predictions")
        
        # 4. Perform backtesting
        print("\n4. Running backtesting...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        backtest_results = predictor.backtest_model(symbol, start_date, end_date)
        if backtest_results:
            print(f"   Backtest Sharpe Ratio: {backtest_results.get('sharpe_ratio', 'N/A')}")
            print(f"   Max Drawdown: {backtest_results.get('max_drawdown', 'N/A'):.2%}")
        
        # 5. Risk assessment
        print("\n5. Assessing risk...")
        risk_metrics = predictor.assess_risk(symbol)
        if risk_metrics:
            print(f"   Risk Grade: {risk_metrics.risk_grade}")
            print(f"   Volatility: {risk_metrics.historical_volatility:.2%}")
        
        print("\nComprehensive analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        return False

def portfolio_analysis_example():
    """Portfolio risk analysis example"""
    print("\n" + "=" * 60)
    print("PORTFOLIO ANALYSIS EXAMPLE")
    print("=" * 60)
    
    try:
        # Define portfolio
        portfolio = {
            "AAPL": 0.3,   # 30% allocation
            "MSFT": 0.25,  # 25% allocation
            "GOOGL": 0.2,  # 20% allocation
            "TSLA": 0.15,  # 15% allocation
            "NVDA": 0.1    # 10% allocation
        }
        
        predictor = StockPredictor()
        risk_analyzer = RiskAssessment()
        
        print("Analyzing portfolio risk...")
        print(f"Portfolio composition: {portfolio}")
        
        # Calculate individual risks
        individual_risks = {}
        print("\nCalculating individual asset risks...")
        
        for symbol, weight in portfolio.items():
            print(f"   Analyzing {symbol} ({weight:.1%})...")
            try:
                # Generate sample returns for demonstration
                np.random.seed(42)  # For reproducible results
                returns = np.random.normal(0.001, 0.02, 252)  # Simulated daily returns
                
                risk_metrics = risk_analyzer.calculate_risk_metrics(returns)
                individual_risks[symbol] = risk_metrics
                
                print(f"     Risk Grade: {risk_metrics.risk_grade}")
                print(f"     Volatility: {risk_metrics.historical_volatility:.2%}")
                
            except Exception as e:
                print(f"     Error analyzing {symbol}: {e}")
        
        # Calculate portfolio risk
        if individual_risks:
            print("\nCalculating portfolio risk...")
            portfolio_risk = risk_analyzer.calculate_portfolio_risk(
                portfolio, individual_risks
            )
            
            print(f"\nPortfolio Risk Assessment:")
            print(f"   Overall Risk Grade: {portfolio_risk.risk_grade}")
            print(f"   Portfolio Volatility: {portfolio_risk.historical_volatility:.2%}")
            print(f"   Portfolio Beta: {portfolio_risk.beta:.2f}")
            print(f"   Max Drawdown: {portfolio_risk.max_drawdown:.2%}")
        
        return True
        
    except Exception as e:
        print(f"Error in portfolio analysis: {e}")
        return False

def export_functionality_example():
    """Demonstrate export functionality"""
    print("\n" + "=" * 60)
    print("EXPORT FUNCTIONALITY EXAMPLE")
    print("=" * 60)
    
    try:
        exporter = ResultsExporter()
        
        # Generate sample prediction data
        dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
        predictions = np.random.uniform(150, 200, 30)  # Sample predictions
        actual_prices = predictions + np.random.normal(0, 5, 30)  # Sample actuals
        
        # Export predictions
        print("Exporting sample prediction results...")
        prediction_files = exporter.export_predictions(
            predictions=predictions,
            actual_prices=actual_prices,
            dates=dates,
            symbol="SAMPLE",
            metadata={"model_type": "LSTM", "confidence": 0.85},
            format="all"
        )
        
        print("Exported prediction files:")
        for format_type, file_path in prediction_files.items():
            print(f"   {format_type.upper()}: {file_path}")
        
        # Generate and export sample risk assessment
        print("\nExporting sample risk assessment...")
        risk_analyzer = RiskAssessment()
        sample_returns = np.random.normal(0.001, 0.02, 252)
        risk_metrics = risk_analyzer.calculate_risk_metrics(sample_returns)
        risk_report = risk_analyzer.generate_risk_report(risk_metrics)
        
        risk_files = exporter.export_risk_assessment(
            risk_metrics=risk_metrics,
            risk_report=risk_report,
            symbol="SAMPLE",
            format="all"
        )
        
        print("Exported risk assessment files:")
        for format_type, file_path in risk_files.items():
            print(f"   {format_type.upper()}: {file_path}")
        
        print("\nExport functionality demonstration completed!")
        return True
        
    except Exception as e:
        print(f"Error in export example: {e}")
        return False

def risk_metrics_example():
    """Demonstrate risk metrics calculation"""
    print("\n" + "=" * 60)
    print("RISK METRICS CALCULATION EXAMPLE")
    print("=" * 60)
    
    try:
        risk_analyzer = RiskAssessment()
        
        # Generate sample returns (252 trading days)
        print("Generating sample return data...")
        np.random.seed(42)
        
        # Simulate stock returns with some realistic characteristics
        returns = np.random.normal(0.0008, 0.015, 252)  # Daily returns
        returns[:50] += np.random.normal(0, 0.01, 50)   # Add some volatility clustering
        
        # Calculate comprehensive risk metrics
        print("Calculating comprehensive risk metrics...")
        risk_metrics = risk_analyzer.calculate_risk_metrics(returns)
        
        # Generate risk report
        risk_report = risk_analyzer.generate_risk_report(risk_metrics)
        
        # Display key metrics
        print(f"\nRisk Assessment Results:")
        print(f"   Overall Risk Grade: {risk_metrics.risk_grade}")
        print(f"   Risk Score: {risk_metrics.overall_risk_score:.3f}")
        print(f"   Annual Volatility: {risk_metrics.historical_volatility:.2%}")
        print(f"   95% VaR: {risk_metrics.var_95:.2%}")
        print(f"   99% VaR: {risk_metrics.var_99:.2%}")
        print(f"   Maximum Drawdown: {risk_metrics.max_drawdown:.2%}")
        print(f"   Skewness: {risk_metrics.skewness:.3f}")
        print(f"   Kurtosis: {risk_metrics.kurtosis:.3f}")
        
        # Calculate risk-adjusted returns
        print("\nCalculating risk-adjusted return metrics...")
        risk_adjusted = calculate_risk_adjusted_returns(returns)
        
        print(f"   Sharpe Ratio: {risk_adjusted['sharpe_ratio']:.3f}")
        print(f"   Sortino Ratio: {risk_adjusted['sortino_ratio']:.3f}")
        print(f"   Calmar Ratio: {risk_adjusted['calmar_ratio']:.3f}")
        print(f"   Excess Return: {risk_adjusted['excess_return']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"Error in risk metrics example: {e}")
        return False

def main():
    """Run all examples"""
    print("STOCK PRICE PREDICTOR - USAGE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates the key functionality of the Stock Price Predictor")
    print("Note: Some examples use simulated data for demonstration purposes")
    print("=" * 60)
    
    examples = [
        ("Basic Prediction", basic_prediction_example),
        ("Comprehensive Analysis", comprehensive_analysis_example),
        ("Portfolio Analysis", portfolio_analysis_example),
        ("Export Functionality", export_functionality_example),
        ("Risk Metrics", risk_metrics_example)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\nRunning {name} example...")
            success = example_func()
            results[name] = success
        except Exception as e:
            print(f"Error running {name} example: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("EXAMPLE EXECUTION SUMMARY")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")
    
    total_passed = sum(results.values())
    total_examples = len(results)
    
    print(f"\nOverall: {total_passed}/{total_examples} examples completed successfully")
    
    if total_passed == total_examples:
        print("🎉 All examples ran successfully!")
    else:
        print("⚠️  Some examples encountered issues. Check the output above for details.")
    
    print("\nFor more information, see the README.md file.")

if __name__ == "__main__":
    main()