#!/usr/bin/env python3
"""
Drift Analysis Dashboard - Flask Backend
========================================

A simple Flask-based dashboard for analyzing vocabulary drift between different language models.
This version works with local models and is extensible for future API key integration.
"""

import os
import sys
import argparse

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        ('flask', 'flask'),
        ('flask_cors', 'flask-cors'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('tiktoken', 'tiktoken'),
        ('sklearn', 'scikit-learn'),
        ('sentence_transformers', 'sentence-transformers'),
        ('rouge_score', 'rouge-score')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    """Main function to start the dashboard"""
    print("📊 Drift Analysis Dashboard")
    print("=" * 40)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Drift Analysis Dashboard")
    parser.add_argument('--port', type=int, default=None, help='Port to run the server on (default: 5000)')
    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        return
    
    # Determine port
    port = args.port or int(os.environ.get('FLASK_RUN_PORT', 5000))
    print(f"🚀 Starting Flask server on port {port}...")
    print(f"🌐 Dashboard will be available at: http://localhost:{port}")
    print(f"📖 API endpoints available at: http://localhost:{port}/api/")
    print("\n💡 Features:")
    print("   - Tokenization drift analysis")
    print("   - Semantic drift analysis") 
    print("   - Token counting for different models")
    print("   - Support for OpenAI and Hugging Face models")
    print("\n🔄 Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        # Start the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")

if __name__ == '__main__':
    main() 