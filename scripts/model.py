#!/usr/bin/env python
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DeiTModel, DeiTForImageClassification
import timm
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class DeiTFeatureExtractor:
    """
    Feature extractor using DeiT (Data-efficient Image Transformer) models.
    
    This class wraps a pre-trained DeiT model to extract features from images,
    which can then be used for downstream tasks like emotion classification.
    Supports fine-tuning the model before feature extraction.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the pre-trained DeiT model. Default is 'facebook/deit-base-distilled-patch16-224'.
    feature_dim : int, optional
        Dimension of the extracted features. Default is 768.
    device : str, optional
        Device to run the model on ('cuda' or 'cpu'). Default is 'cuda' if available.
    freeze_ratio : float, optional
        Fraction of transformer layers to freeze during fine-tuning. Default is 0.8 (80%).
    """
    
    def __init__(
        self,
        model_name: str = 'facebook/deit-base-distilled-patch16-224',
        feature_dim: int = 768,
        device: str = None,
        freeze_ratio: float = 0.8
    ):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.freeze_ratio = freeze_ratio
        self.is_finetuned = False
        
        # Load both models - classification model for fine-tuning and base model for feature extraction
        print(f"[INFO] Loading DeiT model: {model_name} on {self.device}")
        self.classification_model = DeiTForImageClassification.from_pretrained(model_name)
        self.model = DeiTModel.from_pretrained(model_name)
        
        # Move models to device
        self.classification_model = self.classification_model.to(self.device)
        self.model = self.model.to(self.device)
        
        # Initially freeze all parameters in both models
        self._freeze_all_parameters()
        
    def _freeze_all_parameters(self):
        """Freeze all parameters in both models."""
        for param in self.classification_model.parameters():
            param.requires_grad = False
            
        for param in self.model.parameters():
            param.requires_grad = False
            
    def prepare_for_finetuning(self, num_classes: int):
        """
        Prepare the classification model for fine-tuning by unfreezing selected layers.
        
        Parameters
        ----------
        num_classes : int
            Number of target classes for the classification head.
        """
        # Reset the classification head for the new task
        self.classification_model.classifier = torch.nn.Linear(
            self.feature_dim, num_classes).to(self.device)
        
        # Unfreeze the classification head
        for param in self.classification_model.classifier.parameters():
            param.requires_grad = True
        
        # Calculate how many layers to unfreeze
        num_layers = len(self.classification_model.deit.encoder.layer)
        num_layers_to_unfreeze = int(num_layers * (1 - self.freeze_ratio))
        
        # Keep the embedding layer and first (num_layers - num_layers_to_unfreeze) layers frozen
        # and unfreeze the last num_layers_to_unfreeze layers
        for i in range(num_layers - num_layers_to_unfreeze, num_layers):
            for param in self.classification_model.deit.encoder.layer[i].parameters():
                param.requires_grad = True
                
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.classification_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.classification_model.parameters())
        print(f"[INFO] Fine-tuning {trainable_params:,} parameters out of {total_params:,} "
              f"({trainable_params/total_params:.2%})")
    
    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_classes: int = 8,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the DeiT model on emotion classification.
        
        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        val_loader : DataLoader, optional
            DataLoader for validation data. Default is None.
        num_classes : int, optional
            Number of emotion classes. Default is 8.
        epochs : int, optional
            Number of training epochs. Default is 5.
        learning_rate : float, optional
            Learning rate for fine-tuning. Default is 2e-5.
        weight_decay : float, optional
            Weight decay for optimizer. Default is 0.01.
            
        Returns
        -------
        Dict[str, List[float]]
            Dictionary with training history: {'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}.
        """
        # Prepare the model for fine-tuning
        self.prepare_for_finetuning(num_classes)
        
        # Set to training mode
        self.classification_model.train()
        
        # Define optimizer - only optimize parameters that require gradients
        optimizer = torch.optim.AdamW(
            [p for p in self.classification_model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(train_loader)
        )
        
        # Loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.classification_model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in progress_bar:
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.classification_model(images)
                loss = criterion(outputs.logits, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Update loss
                train_loss += loss.item() * images.size(0)
                progress_bar.set_postfix({'loss': loss.item()})
                
            # Calculate average training loss
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        # After fine-tuning, synchronize weights with the feature extraction model
        self._sync_models()
        self.is_finetuned = True
        
        return history
    
    def _validate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """
        Validate the model on the validation set.
        
        Parameters
        ----------
        val_loader : DataLoader
            DataLoader for validation data.
        criterion : torch.nn.Module
            Loss function.
            
        Returns
        -------
        Tuple[float, float]
            Validation loss and accuracy.
        """
        self.classification_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.classification_model(images)
                loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def _sync_models(self):
        """
        Synchronize the weights from the fine-tuned classification model to the feature extraction model.
        """
        # Copy the weights from classification model to feature extraction model
        self.model.load_state_dict(self.classification_model.deit.state_dict())
        self.model.eval()  # Set to evaluation mode for feature extraction
        
    def extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from images using the DeiT model.
        
        Parameters
        ----------
        dataloader : DataLoader
            PyTorch DataLoader containing the images.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (features, labels) as NumPy arrays.
        """
        # Make sure model is in evaluation mode
        self.model.eval()
        
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                # Move to device
                images = images.to(self.device)
                
                # Extract features from the CLS token output
                outputs = self.model(images)
                # Get pooler output (last hidden state of the [CLS] token)
                batch_features = outputs.pooler_output.cpu().numpy()
                
                # Collect features and labels
                features_list.append(batch_features)
                labels_list.append(labels.numpy())
        
        # Concatenate all batches
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        
        return features, labels


class LightGBMClassifier:
    """
    Wrapper for LightGBM classifier for emotion classification.
    
    Parameters
    ----------
    num_classes : int, optional
        Number of emotion classes. Default is 8.
    params : dict, optional
        LightGBM parameters. If None, default parameters are used.
    """
    
    def __init__(self, num_classes: int = 8, params: Optional[Dict] = None):
        self.num_classes = num_classes
        
        # Default parameters optimized for emotion classification
        self.params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'max_depth': -1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Update with user-provided parameters
        if params:
            self.params.update(params)
            
        self.model = None
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        num_boost_round: int = 100,
        early_stopping_rounds: int = 20
    ) -> Dict:
        """
        Train the LightGBM classifier.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        X_val : np.ndarray, optional
            Validation features. Default is None.
        y_val : np.ndarray, optional
            Validation labels. Default is None.
        num_boost_round : int, optional
            Number of boosting iterations. Default is 100.
        early_stopping_rounds : int, optional
            Activates early stopping. Default is 20.
            
        Returns
        -------
        Dict
            Training results.
        """
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val)
            
        # Train the model
        print(f"[INFO] Training LightGBM classifier with {X_train.shape[0]} samples")
        start_time = time.time()
        
        training_args = {
            'params': self.params,
            'train_set': train_data,
            'num_boost_round': num_boost_round,
        }
        
        if val_data:
            training_args.update({
                'valid_sets': [train_data, val_data],
                'valid_names': ['train', 'valid'],
                'callbacks': [lgb.early_stopping(early_stopping_rounds)],
            })
            
        self.model = lgb.train(**training_args)
        
        training_time = time.time() - start_time
        print(f"[INFO] Training completed in {training_time:.2f} seconds")
        
        # Return results
        results = {
            'training_time': training_time,
            'num_iterations': self.model.num_trees(),
        }
        
        if val_data:
            best_iter = self.model.best_iteration
            results.update({
                'best_iteration': best_iter,
                'best_score': self.model.best_score
            })
            
        return results
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters
        ----------
        X : np.ndarray
            Features to predict on.
            
        Returns
        -------
        np.ndarray
            Predicted class indices.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get predicted probabilities
        probs = self.model.predict(X)
        
        # Convert to class indices
        predictions = np.argmax(probs, axis=1)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Features to predict on.
            
        Returns
        -------
        np.ndarray
            Prediction probabilities for each class.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict(X)
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to a file.
        
        Parameters
        ----------
        path : str
            Path to save the model to.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        self.model.save_model(path)
        
    def load_model(self, path: str) -> None:
        """
        Load a trained model from a file.
        
        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        self.model = lgb.Booster(model_file=path)
        self.params['num_class'] = self.model.params['num_class']


class HybridEmotionClassifier:
    """
    Hybrid model for emotion classification combining DeiT feature extraction with LightGBM classification.
    
    This class integrates a DeiT feature extractor with a LightGBM classifier for efficient
    emotion recognition. It handles the end-to-end pipeline from input images to predicted emotions.
    
    Parameters
    ----------
    feature_extractor : DeiTFeatureExtractor or None, optional
        Pre-initialized feature extractor. If None, a new one is created.
    classifier : LightGBMClassifier or None, optional
        Pre-initialized classifier. If None, a new one is created.
    num_classes : int, optional
        Number of emotion classes. Default is 8.
    """
    
    def __init__(
        self,
        feature_extractor: Optional[DeiTFeatureExtractor] = None,
        classifier: Optional[LightGBMClassifier] = None,
        num_classes: int = 8
    ):
        # Initialize or use provided feature extractor
        if feature_extractor is None:
            self.feature_extractor = DeiTFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor
            
        # Initialize or use provided classifier
        if classifier is None:
            self.classifier = LightGBMClassifier(num_classes=num_classes)
        else:
            self.classifier = classifier
            
        self.num_classes = num_classes
        self.is_trained = False
        self.class_names = None
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        class_names: Optional[List[str]] = None,
        finetune_deit: bool = False,
        finetune_epochs: int = 5,
        finetune_lr: float = 2e-5,
        cache_features: bool = True,
        cache_dir: Optional[str] = None
    ) -> Dict:
        """
        Train the hybrid model.
        
        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        val_loader : DataLoader, optional
            DataLoader for validation data. Default is None.
        class_names : List[str], optional
            Names of emotion classes. Default is None.
        finetune_deit : bool, optional
            Whether to fine-tune the DeiT model before feature extraction. Default is False.
        finetune_epochs : int, optional
            Number of epochs for fine-tuning DeiT. Default is 5.
        finetune_lr : float, optional
            Learning rate for fine-tuning DeiT. Default is 2e-5.
        cache_features : bool, optional
            Whether to cache extracted features to disk. Default is True.
        cache_dir : str, optional
            Directory to save cached features. Default is None.
            
        Returns
        -------
        Dict
            Training results.
        """
        self.class_names = class_names
        
        # Step 1: Fine-tune DeiT if requested
        if finetune_deit:
            print(f"[INFO] Fine-tuning DeiT model for {finetune_epochs} epochs")
            finetune_results = self.feature_extractor.finetune(
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=self.num_classes,
                epochs=finetune_epochs,
                learning_rate=finetune_lr
            )
            finetune_prefix = "finetuned_"
        else:
            finetune_results = None
            finetune_prefix = ""
        
        # Function to load or extract features
        def get_features(loader, split_name):
            if cache_features and cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                feature_path = os.path.join(cache_dir, f"{finetune_prefix}{split_name}_features.npz")
                
                if os.path.exists(feature_path):
                    print(f"[INFO] Loading cached features from {feature_path}")
                    data = np.load(feature_path)
                    return data['features'], data['labels']
                    
            print(f"[INFO] Extracting features for {split_name} set")
            features, labels = self.feature_extractor.extract_features(loader)
            
            if cache_features and cache_dir:
                print(f"[INFO] Caching features to {feature_path}")
                np.savez(feature_path, features=features, labels=labels)
                
            return features, labels
            
        # Step 2: Extract features for training
        X_train, y_train = get_features(train_loader, "train")
        
        # Extract features for validation if provided
        X_val, y_val = None, None
        if val_loader:
            X_val, y_val = get_features(val_loader, "val")
        
        # Step 3: Train the LightGBM classifier
        lgb_results = self.classifier.train(
            X_train, y_train,
            X_val, y_val
        )
        
        self.is_trained = True
        
        # Combine results
        results = lgb_results
        if finetune_results:
            results['finetune_history'] = finetune_results
            
        return results
        
    def evaluate(
        self,
        test_loader: DataLoader,
        cache_features: bool = True,
        cache_dir: Optional[str] = None,
        plot_confusion_matrix: bool = True
    ) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Parameters
        ----------
        test_loader : DataLoader
            DataLoader for test data.
        cache_features : bool, optional
            Whether to use cached features if available. Default is True.
        cache_dir : str, optional
            Directory where cached features are stored. Default is None.
        plot_confusion_matrix : bool, optional
            Whether to plot and return confusion matrix. Default is True.
            
        Returns
        -------
        Dict
            Evaluation results including accuracy, classification report, and confusion matrix.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get prefix for cached features based on whether the model was fine-tuned
        finetune_prefix = "finetuned_" if self.feature_extractor.is_finetuned else ""
            
        # Load or extract features
        if cache_features and cache_dir and os.path.exists(os.path.join(cache_dir, f"{finetune_prefix}test_features.npz")):
            print(f"[INFO] Loading cached test features")
            data = np.load(os.path.join(cache_dir, f"{finetune_prefix}test_features.npz"))
            X_test, y_test = data['features'], data['labels']
        else:
            print(f"[INFO] Extracting test features")
            X_test, y_test = self.feature_extractor.extract_features(test_loader)
            if cache_features and cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez(os.path.join(cache_dir, f"{finetune_prefix}test_features.npz"), features=X_test, labels=y_test)
                
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get class names
        target_names = self.class_names if self.class_names else [f"Class {i}" for i in range(self.num_classes)]
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix if requested
        if plot_confusion_matrix:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, 
                        yticklabels=target_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.show()
        
        # Return results
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Make predictions on a batch of images.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader containing images to predict.
            
        Returns
        -------
        np.ndarray
            Predicted class indices.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        # Extract features
        features, _ = self.feature_extractor.extract_features(dataloader)
        
        # Make predictions
        predictions = self.classifier.predict(features)
        
        return predictions
    
    def save(self, directory: str) -> None:
        """
        Save the trained classifier model.
        
        Parameters
        ----------
        directory : str
            Directory to save the model to.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        os.makedirs(directory, exist_ok=True)
        
        # Save LightGBM model
        model_path = os.path.join(directory, "lightgbm_model.txt")
        self.classifier.save_model(model_path)
        
        # Save fine-tuned DeiT model if it was fine-tuned
        if self.feature_extractor.is_finetuned:
            deit_path = os.path.join(directory, "deit_model")
            os.makedirs(deit_path, exist_ok=True)
            self.feature_extractor.model.save_pretrained(deit_path)
            print(f"[INFO] Fine-tuned DeiT model saved to {deit_path}")
        
        # Save class names if available
        if self.class_names:
            with open(os.path.join(directory, "class_names.txt"), 'w') as f:
                for name in self.class_names:
                    f.write(f"{name}\n")
                    
        # Save metadata about the model
        with open(os.path.join(directory, "model_info.json"), 'w') as f:
            json.dump({
                "is_finetuned": self.feature_extractor.is_finetuned,
                "num_classes": self.num_classes,
                "model_name": self.feature_extractor.model_name,
                "feature_dim": self.feature_extractor.feature_dim
            }, f)
            
        print(f"[INFO] Model saved to {directory}")
        
    def load(self, directory: str) -> None:
        """
        Load a trained model.
        
        Parameters
        ----------
        directory : str
            Directory to load the model from.
        """
        # Load model info
        model_info_path = os.path.join(directory, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                
            # Initialize feature extractor with saved parameters if needed
            if hasattr(self.feature_extractor, 'model_name') and self.feature_extractor.model_name != model_info.get('model_name'):
                print(f"[WARNING] Current feature extractor model_name ({self.feature_extractor.model_name}) "
                      f"differs from saved model_name ({model_info.get('model_name')})")
        
        # Load LightGBM model
        model_path = os.path.join(directory, "lightgbm_model.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.classifier.load_model(model_path)
        
        # Load fine-tuned DeiT model if available
        deit_path = os.path.join(directory, "deit_model")
        if os.path.exists(deit_path):
            print(f"[INFO] Loading fine-tuned DeiT model from {deit_path}")
            self.feature_extractor.model = DeiTModel.from_pretrained(deit_path)
            self.feature_extractor.model = self.feature_extractor.model.to(self.feature_extractor.device)
            self.feature_extractor.model.eval()
            self.feature_extractor.is_finetuned = True
        
        # Load class names if available
        class_names_path = os.path.join(directory, "class_names.txt")
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f]
        
        self.num_classes = self.classifier.params['num_class']
        self.is_trained = True
        
        print(f"[INFO] Model loaded from {directory}")