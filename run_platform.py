"""
LSTM Trading Platform Startup Script
Launches the professional trading platform with all features
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'yfinance',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn',
        'tensorflow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("All packages installed successfully!")
    else:
        print("All required packages are already installed.")

def launch_app():
    """Launch the Streamlit application"""
    print("\n🚀 Launching LSTM Trading Platform...")
    print("🌐 Opening in your default browser...")
    print("📊 Professional trading features enabled")
    print("\n" + "="*50)
    print("LSTM TRADING PLATFORM - PROFESSIONAL EDITION")
    print("="*50)
    print("✅ Real-time Market Data")
    print("✅ AI-Powered Predictions") 
    print("✅ Advanced Technical Analysis")
    print("✅ Portfolio Management")
    print("✅ Alert System")
    print("✅ Professional Charts")
    print("✅ Data Management")
    print("="*50 + "\n")
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main_app.py",
            "--theme.base", "dark",
            "--theme.primaryColor", "#00d4aa",
            "--theme.backgroundColor", "#0e1117",
            "--theme.secondaryBackgroundColor", "#1a1a1a",
            "--theme.textColor", "#fafafa"
        ])
    except KeyboardInterrupt:
        print("\n👋 LSTM Trading Platform closed successfully!")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("💡 Try running manually: streamlit run main_app.py")

if __name__ == "__main__":
    print("🔍 Checking requirements...")
    check_requirements()
    launch_app()