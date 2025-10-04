"""
LSTM Stock Predictor Launcher
Double-click this file to start the app!
"""

import subprocess
import sys
import os
import webbrowser
import time

def main():
    print("🚀 LSTM Stock Predictor Launcher")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {os.getcwd()}")
    print("⚡ Starting Streamlit app...")
    
    try:
        # Start the Streamlit app
        cmd = [sys.executable, "-m", "streamlit", "run", "stock_app.py", 
               "--server.port", "8507", "--server.address", "localhost"]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("🌐 App starting... Please wait...")
        time.sleep(3)  # Wait a bit for the server to start
        
        # Open browser
        url = "http://localhost:8507"
        print(f"🌍 Opening browser: {url}")
        webbrowser.open(url)
        
        print("\n✅ App is running!")
        print("🔗 URL: http://localhost:8507")
        print("\n📊 Features available:")
        print("   • Real-time stock analysis")
        print("   • LSTM price predictions") 
        print("   • Technical indicators")
        print("   • Interactive charts")
        print("   • Demo mode (always works)")
        
        print("\n💡 Usage:")
        print("   1. Enter stock symbol (AAPL, GOOGL, TSLA)")
        print("   2. Click 'Demo Mode' for instant results")
        print("   3. Or try 'Analyze Stock' for real data")
        
        print("\n⚠️  To stop the app: Close this window or press Ctrl+C")
        print("=" * 50)
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping app...")
        process.terminate()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Manual start command:")
        print("python -m streamlit run stock_app.py --server.port 8507")
        
    print("\n👋 Thanks for using LSTM Stock Predictor!")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()