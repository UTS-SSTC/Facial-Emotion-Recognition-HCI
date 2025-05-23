#!/usr/bin/env python
"""
Flask application initialization module.

This module sets up the Flask application with necessary configurations
for emotion detection demo.
"""

from flask import Flask
import os


def create_app():
    """
    Create and configure Flask application instance.
    
    Returns
    -------
    Flask
        Configured Flask application instance.
    """
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    return app