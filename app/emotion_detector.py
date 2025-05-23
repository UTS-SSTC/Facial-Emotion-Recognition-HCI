#!/usr/bin/env python
"""
Emotion detection module using DeepFace and pre-trained models.

This module provides functionality for face detection and emotion classification
using DeepFace for face extraction and custom trained models for emotion prediction.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from deepface import DeepFace
import json

# Import from parent scripts directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.model import HybridEmotionClassifier
from scripts.backbone import VisionFeatureExtractor


class SingleImageDataset(Dataset):
    """
    Dataset for single image prediction.
    
    Parameters
    ----------
    image : np.ndarray
        Image array in RGB format.
    transform : callable, optional
        Albumentations transform to apply to the image.
    """
    
    def __init__(self, image: np.ndarray, transform=None):
        self.image = image
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset size (always 1 for single image)."""
        return 1
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get transformed image tensor.
        
        Parameters
        ----------
        idx : int
            Index (ignored for single image dataset).
            
        Returns
        -------
        torch.Tensor
            Transformed image tensor.
        """
        image = self.image
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, torch.tensor(0, dtype=torch.long)  # dummy label


class EmotionDetector:
    """
    Emotion detection system combining face detection and emotion classification.
    
    This class integrates DeepFace for face detection with custom trained models
    for emotion classification on detected faces.
    
    Parameters
    ----------
    model_path : str, optional
        Path to the saved emotion classification model. Default is None.
    device : str, optional
        Device to run inference on ('cuda', 'cpu', 'mps'). Default is None (auto-detect).
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"[INFO] Using device: {self.device}")
        
        # Emotion class names
        self.emotion_classes = [
            'anger', 'disgust', 'fear', 'happy', 
            'sad', 'surprise', 'neutral', 'contempt'
        ]
        
        # Initialize model
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Create transform for inference
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained emotion classification model.
        
        Parameters
        ----------
        model_path : str
            Path to the saved model directory.
        """
        try:
            # Load model info to get model configuration
            model_info_path = os.path.join(model_path, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                model_name = model_info.get('model_name', 'facebook/deit-base-distilled-patch16-224')
            else:
                model_name = 'facebook/deit-base-distilled-patch16-224'
            
            # Initialize components
            feature_extractor = VisionFeatureExtractor(
                model_name=model_name,
                device=self.device
            )
            
            # Create and load hybrid model
            self.model = HybridEmotionClassifier(
                feature_extractor=feature_extractor,
                num_classes=len(self.emotion_classes)
            )
            self.model.load(model_path)
            self.model.class_names = self.emotion_classes
            
            print(f"[INFO] Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model from {model_path}: {e}")
            self.model = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in the input image using DeepFace.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in RGB format.
            
        Returns
        -------
        List[Dict]
            List of detected face regions with bounding box coordinates.
        """
        try:
            # Convert RGB to BGR for DeepFace
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect faces using DeepFace
            faces = DeepFace.extract_faces(
                img_path=image_bgr,
                target_size=(224, 224),
                detector_backend='opencv',
                enforce_detection=False
            )
            
            # Get face regions with coordinates
            face_objs = DeepFace.analyze(
                img_path=image_bgr,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False
            )
            
            if not isinstance(face_objs, list):
                face_objs = [face_objs]
            
            detected_faces = []
            for i, (face_img, face_obj) in enumerate(zip(faces, face_objs)):
                # Convert face image back to RGB and to uint8
                face_rgb = (face_img * 255).astype(np.uint8)
                
                # Get bounding box coordinates
                region = face_obj.get('region', {})
                bbox = {
                    'x': region.get('x', 0),
                    'y': region.get('y', 0),
                    'w': region.get('w', face_rgb.shape[1]),
                    'h': region.get('h', face_rgb.shape[0])
                }
                
                detected_faces.append({
                    'face_id': i,
                    'image': face_rgb,
                    'bbox': bbox
                })
            
            return detected_faces
            
        except Exception as e:
            print(f"[WARNING] Face detection failed: {e}")
            return []
    
    def predict_emotion(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Predict emotion from a face image.
        
        Parameters
        ----------
        face_image : np.ndarray
            Face image in RGB format.
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping emotion names to confidence scores.
        """
        if self.model is None:
            print("[WARNING] No model loaded, using random predictions")
            # Return random predictions if no model is loaded
            predictions = np.random.random(len(self.emotion_classes))
            predictions = predictions / predictions.sum()
            return {emotion: float(score) for emotion, score in zip(self.emotion_classes, predictions)}
        
        try:
            # Create dataset and dataloader for single image
            dataset = SingleImageDataset(face_image, transform=self.transform)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            # Get predictions
            features, _ = self.model.feature_extractor.extract_features(dataloader)
            probabilities = self.model.classifier.predict_proba(features)[0]
            
            # Create emotion-probability mapping
            emotion_scores = {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotion_classes, probabilities)
            }
            
            return emotion_scores
            
        except Exception as e:
            print(f"[ERROR] Emotion prediction failed: {e}")
            # Return uniform distribution as fallback
            uniform_prob = 1.0 / len(self.emotion_classes)
            return {emotion: uniform_prob for emotion in self.emotion_classes}
    
    def process_image(self, image: np.ndarray) -> List[Dict]:
        """
        Process an image to detect faces and predict emotions.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in RGB format.
            
        Returns
        -------
        List[Dict]
            List of results for each detected face containing:
            - face_id: int, unique identifier for the face
            - bbox: dict, bounding box coordinates {x, y, w, h}
            - emotions: dict, emotion predictions {emotion_name: confidence}
            - top_emotion: str, emotion with highest confidence
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        results = []
        for face_data in faces:
            # Predict emotion for each face
            emotions = self.predict_emotion(face_data['image'])
            
            # Find top emotion
            top_emotion = max(emotions, key=emotions.get)
            
            results.append({
                'face_id': face_data['face_id'],
                'bbox': face_data['bbox'],
                'emotions': emotions,
                'top_emotion': top_emotion
            })
        
        return results
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a video frame with face detection and emotion prediction.
        
        Parameters
        ----------
        frame : np.ndarray
            Video frame in BGR format (OpenCV format).
            
        Returns
        -------
        Tuple[np.ndarray, List[Dict]]
            Tuple containing:
            - Annotated frame with bounding boxes and emotion labels
            - List of detection results
        """
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.process_image(frame_rgb)
        
        # Annotate frame
        annotated_frame = frame.copy()
        for result in results:
            bbox = result['bbox']
            top_emotion = result['top_emotion']
            confidence = result['emotions'][top_emotion]
            
            # Draw bounding box
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw emotion label
            label = f"{top_emotion}: {confidence:.2f}"
            cv2.putText(
                annotated_frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        return annotated_frame, results


def get_available_models() -> List[str]:
    """
    Get list of available trained models.
    
    Returns
    -------
    List[str]
        List of available model directory names.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(models_dir):
        return []
    
    available_models = []
    for item in os.listdir(models_dir):
        model_path = os.path.join(models_dir, item)
        if os.path.isdir(model_path):
            # Check if it contains required model files
            if os.path.exists(os.path.join(model_path, 'lightgbm_model.txt')):
                available_models.append(item)
    
    return available_models