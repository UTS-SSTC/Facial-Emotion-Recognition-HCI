#!/usr/bin/env python
from __future__ import annotations

import os
import json
import numpy as np
import time
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from transformers import DeiTModel
from transformers import AutoConfig, AutoModel 
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.backbone import VisionFeatureExtractor


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
            feature_extractor: Optional[VisionFeatureExtractor] = None,
            classifier: Optional[LightGBMClassifier] = None,
            num_classes: int = 8
    ):
        # Initialize or use provided feature extractor
        if feature_extractor is None:
            self.feature_extractor = VisionFeatureExtractor()
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
            finetune_backbone: bool = False,
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
        finetune_backbone : bool, optional
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
        if finetune_backbone:
            print(f"[INFO] Fine-tuning model for {finetune_epochs} epochs")
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
        if cache_features and cache_dir and os.path.exists(
                os.path.join(cache_dir, f"{finetune_prefix}test_features.npz")):
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
            backbone_path = os.path.join(directory, "deit_model")
            os.makedirs(backbone_path, exist_ok=True)
            self.feature_extractor.model.save_pretrained(backbone_path)
            print(f"[INFO] Fine-tuned model saved to {backbone_path}")

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
            if hasattr(self.feature_extractor, 'model_name') and self.feature_extractor.model_name != model_info.get(
                    'model_name'):
                print(f"[WARNING] Current feature extractor model_name ({self.feature_extractor.model_name}) "
                      f"differs from saved model_name ({model_info.get('model_name')})")

        # Load LightGBM model
        model_path = os.path.join(directory, "lightgbm_model.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.classifier.load_model(model_path)

        # Load fine-tuned DeiT model if available
        backbone_path = os.path.join(directory, "deit_model")      # 保持旧目录名也行
        if os.path.exists(backbone_path):
            cfg = AutoConfig.from_pretrained(backbone_path)
            print(f"[INFO] Loading fine-tuned {cfg.model_type.upper()} backbone "
                  f"from {backbone_path}")
    
            self.feature_extractor.model = AutoModel.from_pretrained(
                backbone_path,
                config=cfg,
                ignore_mismatched_sizes=True
            ).to(self.feature_extractor.device).eval()
    
            self.feature_extractor.is_finetuned = True

        # Load class names if available
        class_names_path = os.path.join(directory, "class_names.txt")
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f]

        self.num_classes = self.classifier.params['num_class']
        self.is_trained = True

        print(f"[INFO] Model loaded from {directory}")
        # """
        # Load a trained model.

        # Parameters
        # ----------
        # directory : str
        #     Directory to load the model from.
        # """
        # # Load model info
        # model_info_path = os.path.join(directory, "model_info.json")
        # if os.path.exists(model_info_path):
        #     with open(model_info_path, 'r') as f:
        #         model_info = json.load(f)

        #     # Initialize feature extractor with saved parameters if needed
        #     if hasattr(self.feature_extractor, 'model_name') and self.feature_extractor.model_name != model_info.get(
        #             'model_name'):
        #         print(f"[WARNING] Current feature extractor model_name ({self.feature_extractor.model_name}) "
        #               f"differs from saved model_name ({model_info.get('model_name')})")

        # # Load LightGBM model
        # model_path = os.path.join(directory, "lightgbm_model.txt")
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"Model file not found at {model_path}")

        # self.classifier.load_model(model_path)

        # # Load fine-tuned DeiT model if available
        # backbone_path = os.path.join(directory, "deit_model")
        # if os.path.exists(backbone_path):
        #     print(f"[INFO] Loading fine-tuned model from {backbone_path}")
        #     self.feature_extractor.model = DeiTModel.from_pretrained(backbone_path)
        #     self.feature_extractor.model = self.feature_extractor.model.to(self.feature_extractor.device)
        #     self.feature_extractor.model.eval()
        #     self.feature_extractor.is_finetuned = True

        # # Load class names if available
        # class_names_path = os.path.join(directory, "class_names.txt")
        # if os.path.exists(class_names_path):
        #     with open(class_names_path, 'r') as f:
        #         self.class_names = [line.strip() for line in f]

        # self.num_classes = self.classifier.params['num_class']
        # self.is_trained = True

        # print(f"[INFO] Model loaded from {directory}")