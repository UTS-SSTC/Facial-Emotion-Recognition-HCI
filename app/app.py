#!/usr/bin/env python
"""
Flask application for emotion detection demo.

This module provides a web interface for emotion detection with two main features:
1. Upload image for emotion detection
2. Real-time camera emotion detection
"""

import os
import cv2
import json
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, url_for, flash, redirect
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

from emotion_detector import EmotionDetector, get_available_models
from __init__ import create_app


# Initialize Flask app
app = create_app()

# Initialize emotion detector
detector = None
current_model = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename: str) -> bool:
    """
    Check if uploaded file has allowed extension.
    
    Parameters
    ----------
    filename : str
        Name of the uploaded file.
        
    Returns
    -------
    bool
        True if file extension is allowed, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_if_needed(model_name: str = None) -> bool:
    """
    Load emotion detection model if not already loaded or if different model requested.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the model to load. If None, loads the first available model.
        
    Returns
    -------
    bool
        True if model loaded successfully, False otherwise.
    """
    global detector, current_model
    
    available_models = get_available_models()
    if not available_models:
        print("[WARNING] No trained models found in models directory")
        return False
    
    # Use first available model if none specified
    if model_name is None:
        model_name = available_models[0]
    
    # Load model if not already loaded or different model requested
    if detector is None or current_model != model_name:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_name)
        detector = EmotionDetector(model_path=model_path)
        current_model = model_name
        print(f"[INFO] Loaded model: {model_name}")
    
    return detector.model is not None


@app.route('/')
def index():
    """
    Home page route.
    
    Returns
    -------
    str
        Rendered HTML template for the home page.
    """
    available_models = get_available_models()
    return render_template('index.html', models=available_models)


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    """
    Handle image upload and emotion detection.
    
    Returns
    -------
    str or Response
        Rendered HTML template or JSON response with detection results.
    """
    if request.method == 'GET':
        available_models = get_available_models()
        return render_template('upload.html', models=available_models)
    
    # Handle POST request
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    model_name = request.form.get('model')
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload an image file.')
        return redirect(request.url)
    
    # Load model
    if not load_model_if_needed(model_name):
        flash('Failed to load emotion detection model')
        return redirect(request.url)
    
    try:
        # Process uploaded image
        filename = secure_filename(file.filename)
        image = Image.open(file.stream).convert('RGB')
        image_array = np.array(image)
        
        # Detect emotions
        results = detector.process_image(image_array)
        
        # Convert image to base64 for display
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Check if it's an AJAX request
        if request.headers.get('Content-Type') == 'application/json' or \
           request.headers.get('Accept') == 'application/json':
            return jsonify({
                'success': True,
                'results': results,
                'image': img_str,
                'model_used': current_model
            })
        
        return render_template('upload.html', 
                             results=results, 
                             image=img_str,
                             model_used=current_model,
                             models=get_available_models())
    
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        flash(error_msg)
        
        if request.headers.get('Content-Type') == 'application/json':
            return jsonify({'success': False, 'error': error_msg})
        
        return redirect(request.url)


@app.route('/camera')
def camera():
    """
    Camera detection page route.
    
    Returns
    -------
    str
        Rendered HTML template for camera detection page.
    """
    available_models = get_available_models()
    return render_template('camera.html', models=available_models)


@app.route('/video_feed')
def video_feed():
    """
    Video streaming route for real-time emotion detection.
    
    Returns
    -------
    Response
        Streaming response with video frames.
    """
    model_name = request.args.get('model')
    
    # Load model
    if not load_model_if_needed(model_name):
        # Return error frame if model loading fails
        def error_frame():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Model Loading Failed", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        
        return Response(error_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def generate_frames():
        """
        Generate video frames with emotion detection.
        
        Yields
        ------
        bytes
            Encoded video frame with emotion annotations.
        """
        camera = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        
        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break
                
                try:
                    # Process frame for emotion detection
                    annotated_frame, results = detector.process_frame(frame)
                    
                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                except Exception as e:
                    print(f"[ERROR] Frame processing failed: {e}")
                    # Return original frame if processing fails
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        finally:
            camera.release()
    
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/models')
def api_models():
    """
    API endpoint to get available models.
    
    Returns
    -------
    Response
        JSON response with list of available models.
    """
    models = get_available_models()
    return jsonify({'models': models, 'current': current_model})


@app.route('/api/switch_model', methods=['POST'])
def api_switch_model():
    """
    API endpoint to switch emotion detection model.
    
    Returns
    -------
    Response
        JSON response indicating success or failure.
    """
    data = request.get_json()
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({'success': False, 'error': 'Model name required'})
    
    available_models = get_available_models()
    if model_name not in available_models:
        return jsonify({'success': False, 'error': 'Invalid model name'})
    
    success = load_model_if_needed(model_name)
    
    return jsonify({
        'success': success,
        'current_model': current_model,
        'message': f'Switched to model: {model_name}' if success else 'Failed to load model'
    })


if __name__ == '__main__':
    # Load default model on startup
    load_model_if_needed()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)