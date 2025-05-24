#!/usr/bin/env python
"""
Entry point for the Flask emotion detection application.

This script starts the Flask development server with the emotion detection app.
For production deployment, use a WSGI server like Gunicorn instead.
"""

import os
import sys

# Add parent directory to path to access scripts module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app import app

if __name__ == '__main__':
    # Check if models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(models_dir):
        print("[WARNING] Models directory not found. Please ensure trained models are available.")
        print(f"Expected location: {models_dir}")
    
    # Check if required directories exist
    static_dirs = ['static/css', 'static/js', 'static/uploads']
    for static_dir in static_dirs:
        dir_path = os.path.join(os.path.dirname(__file__), static_dir)
        os.makedirs(dir_path, exist_ok=True)
    
    print("Starting Emotion Detection Flask App...")
    print("Navigate to http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )