#!/usr/bin/env python
"""
Enhanced evaluation script for emotion detection models comparison.

This script performs comprehensive comparison of multiple trained models including:
1. ROC curves for each emotion class across all models
2. Performance metrics comparison (precision, recall, f1-score, specificity)
3. Generates comparison visualization charts
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'scripts' else current_dir
sys.path.insert(0, project_root)

# Import custom modules
try:
    import scripts.backbone as b
    import scripts.model as md
except ImportError:
    # Try alternative import paths
    try:
        sys.path.insert(0, os.path.join(project_root, 'scripts'))
        import backbone as b
        import model as md
    except ImportError:
        print("[ERROR] Cannot import required modules. Please ensure you're running from the project root directory.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Project root: {project_root}")
        print("Expected directory structure:")
        print("  Facial-Emotion-Recognition-HCI/")
        print("  ├── scripts/")
        print("  │   ├── backbone.py")
        print("  │   ├── model.py")
        print("  │   └── evaluate.py")
        print("  └── ...")
        sys.exit(1)

class AffectNetDataset(Dataset):
    """
    PyTorch Dataset for AffectNet emotion classification.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        DataFrame containing 'file_path' and 'expression' columns.
    transform : callable, optional
        Albumentations transform to apply to each image.

    Attributes
    ----------
    label_to_idx : dict
        Mapping from expression labels to integer indices.
    metadata_df : pd.DataFrame
        DataFrame with added 'label_idx' column for training.
    """

    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df
        self.transform = transform

        # Encode expressions to integer labels
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.metadata_df['expression'].unique()))}
        self.metadata_df['label_idx'] = self.metadata_df['expression'].map(self.label_to_idx)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx):
        """
        Load image and label at a given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Transformed image tensor.
        label : torch.Tensor
            Corresponding label tensor (long type).
        """
        row = self.metadata_df.iloc[idx]
        img_path = row['file_path']
        label = row['label_idx']

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Warning] Failed to load {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')  # fallback image

        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label, dtype=torch.long)


def create_affectnet_transforms(input_size=224):
    """
    Create Albumentations transforms for AffectNet dataset.

    Parameters
    ----------
    input_size : int, optional
        Target square image size. Default is 224.

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', and 'test' mapping to
        Albumentations transform pipelines.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    base = [
        A.Resize(input_size, input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]

    train_aug = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.ColorJitter(p=0.5)
    ]

    return {
        'train': A.Compose(train_aug + base),
        'val': A.Compose(base),
        'test': A.Compose(base)
    }


def load_affectnet_data(
    csv_path,
    train_ratio=0.7,
    val_ratio=0.15,
    batch_size=64,
    input_size=224,
    num_workers=4,
    shuffle=True,
    seed=42
):
    """
    Load AffectNet metadata CSV, split into train/val/test sets, and build DataLoaders.

    Parameters
    ----------
    csv_path : str
        Path to the metadata CSV file. Must include 'file_path' and 'expression' columns.
    train_ratio : float, optional
        Proportion of the dataset to use for training. Default is 0.7.
    val_ratio : float, optional
        Proportion of the dataset to use for validation. Default is 0.15.
    batch_size : int, optional
        Batch size for all splits. Default is 64.
    input_size : int, optional
        Image resize size. Default is 224.
    num_workers : int, optional
        Number of subprocesses for data loading. Default is 4.
    shuffle : bool, optional
        Whether to shuffle the dataset before splitting. Default is True.
    seed : int, optional
        Random seed for shuffling. Default is 42.

    Returns
    -------
    dict
        {
            'train_loader': DataLoader,
            'val_loader': DataLoader,
            'test_loader': DataLoader,
            'train_df': pd.DataFrame,
            'val_df': pd.DataFrame,
            'test_df': pd.DataFrame,
            'metadata_df': pd.DataFrame
        }
    """
    metadata_df = pd.read_csv(csv_path)
    if shuffle:
        metadata_df = metadata_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    total = len(metadata_df)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_df = metadata_df[:train_end]
    val_df = metadata_df[train_end:val_end]
    test_df = metadata_df[val_end:]

    print(f"[✓] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    transforms = create_affectnet_transforms(input_size)

    train_loader = DataLoader(
        AffectNetDataset(train_df, transform=transforms['train']),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
        prefetch_factor=2, persistent_workers=True
    )

    val_loader = DataLoader(
        AffectNetDataset(val_df, transform=transforms['val']),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    test_loader = DataLoader(
        AffectNetDataset(test_df, transform=transforms['test']),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'metadata_df': metadata_df
    }


class ModelComparator:
    """
    Comprehensive model comparison tool for emotion detection models.
    
    This class handles loading multiple models, evaluating them on test data,
    and generating detailed comparison visualizations.
    """
    
    def __init__(self, device=None):
        """Initialize the model comparator."""
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"[INFO] Using device: {self.device}")
        
        # Model configurations
        self.model_configs = [
            ("facebook/deit-small-distilled-patch16-224", "models/facebook_deit-small-distilled-patch16-224"),
            ("facebook/deit-base-distilled-patch16-224", "models/facebook_deit-base-distilled-patch16-224"), 
            ("microsoft/resnet-50", "models/microsoft_resnet-50"),
            ("microsoft/resnet-152", "models/microsoft_resnet-152"),
            ("timm/vit_small_patch16_224.augreg_in21k", "models/timm_vit_small_patch16_224.augreg_in21k"),
            ("google/vit-base-patch16-224", "models/google_vit-base-patch16-224")
        ]
        
        # Emotion class names
        self.class_names = [
            'anger', 'contempt', 'disgust', 'fear', 
            'happy', 'neutral', 'sad', 'surprise'
        ]
        
        self.models = {}
        self.results = {}
        
        # Create output directory
        os.makedirs('images', exist_ok=True)
    
    def load_models(self):
        """Load all models for comparison."""
        print("[INFO] Loading models for comparison...")
        
        for model_name, model_path in self.model_configs:
            try:
                print(f"[INFO] Loading {model_name}...")
                
                # Initialize backbone
                backbone = b.VisionFeatureExtractor(model_name=model_name, device=self.device)
                
                # Initialize hybrid model
                model = md.HybridEmotionClassifier(backbone)
                
                # Load trained weights
                model.load(model_path)
                
                # Store model with a short name
                short_name = model_name.split('/')[-1].replace('-', '_')
                self.models[short_name] = model
                
                print(f"[✓] Successfully loaded {short_name}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load {model_name}: {e}")
                continue
    
    def evaluate_model(self, model, test_loader):
        """
        Evaluate a single model and return detailed metrics.
        
        Parameters
        ----------
        model : HybridEmotionClassifier
            The model to evaluate
        test_loader : DataLoader
            Test data loader
            
        Returns
        -------
        dict
            Dictionary containing evaluation results
        """
        print(f"[INFO] Evaluating model...")
        
        # Extract features and labels
        X_test, y_test = model.feature_extractor.extract_features(test_loader)
        
        # Get predictions and probabilities
        y_pred = model.classifier.predict(X_test)
        y_prob = model.classifier.predict_proba(X_test)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=range(len(self.class_names))
        )
        
        # Calculate specificity for each class
        cm = confusion_matrix(y_test, y_pred, labels=range(len(self.class_names)))
        specificity = []
        for i in range(len(self.class_names)):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity.append(spec)
        
        # Prepare binary labels for ROC curves
        y_test_binary = label_binarize(y_test, classes=range(len(self.class_names)))
        
        # Calculate ROC curves for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(len(self.class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': np.array(specificity),
            'support': support,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def evaluate_all_models(self, test_loader):
        """Evaluate all loaded models."""
        print("[INFO] Evaluating all models...")
        
        for model_name, model in self.models.items():
            print(f"\n[INFO] Evaluating {model_name}...")
            results = self.evaluate_model(model, test_loader)
            self.results[model_name] = results
            
            print(f"[✓] {model_name} - Accuracy: {results['accuracy']:.4f}")
    
    def plot_roc_curves_comparison(self):
        """Plot ROC curves for all models across all classes."""
        print("[INFO] Generating ROC curves comparison...")
        
        # Create figure with 2x4 subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('ROC Curves Comparison Across Models and Emotion Classes', fontsize=16)
        
        # Color palette for models
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))
        
        for class_idx, class_name in enumerate(self.class_names):
            row = class_idx // 4
            col = class_idx % 4
            ax = axes[row, col]
            
            # Plot ROC curve for each model
            for (model_name, color) in zip(self.models.keys(), colors):
                if model_name in self.results:
                    results = self.results[model_name]
                    fpr = results['fpr'][class_idx]
                    tpr = results['tpr'][class_idx]
                    auc_score = results['roc_auc'][class_idx]
                    
                    ax.plot(fpr, tpr, color=color, lw=2, 
                           label=f'{model_name} (AUC={auc_score:.3f})')
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            
            # Customize subplot
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{class_name.title()} Class')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[✓] ROC curves saved to images/roc_curves_comparison.png")
    
    def plot_metrics_comparison(self):
        """Plot comparison of precision, recall, f1-score, and specificity."""
        print("[INFO] Generating metrics comparison plots...")
        
        metrics = ['precision', 'recall', 'f1_score', 'specificity']
        metric_names = ['Precision', 'Recall', 'F1-Score', 'Specificity']
        
        for metric, metric_name in zip(metrics, metric_names):
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            # Prepare data for plotting
            model_names = list(self.results.keys())
            x = np.arange(len(self.class_names))
            width = 0.12  # Width of bars
            
            # Plot bars for each model
            for i, model_name in enumerate(model_names):
                if model_name in self.results:
                    values = self.results[model_name][metric]
                    offset = (i - len(model_names)/2 + 0.5) * width
                    bars = ax.bar(x + offset, values, width, 
                                 label=model_name, alpha=0.8)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        if value > 0.01:  # Only show label if value is significant
                            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            # Customize plot
            ax.set_xlabel('Emotion Classes')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison Across Models and Emotion Classes')
            ax.set_xticks(x)
            ax.set_xticklabels([name.title() for name in self.class_names])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig(f'images/{metric}_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[✓] {metric_name} comparison saved to images/{metric}_comparison.png")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("[INFO] Generating summary report...")
        
        # Create summary dataframe
        summary_data = []
        
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Overall_Accuracy': results['accuracy'],
                'Avg_Precision': np.mean(results['precision']),
                'Avg_Recall': np.mean(results['recall']),
                'Avg_F1_Score': np.mean(results['f1_score']),
                'Avg_Specificity': np.mean(results['specificity'])
            }
            
            # Add per-class metrics
            for i, class_name in enumerate(self.class_names):
                row[f'{class_name.title()}_Precision'] = results['precision'][i]
                row[f'{class_name.title()}_Recall'] = results['recall'][i]
                row[f'{class_name.title()}_F1'] = results['f1_score'][i]
                row[f'{class_name.title()}_Specificity'] = results['specificity'][i]
                row[f'{class_name.title()}_AUC'] = results['roc_auc'][i]
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('images/model_comparison_summary.csv', index=False)
        
        # Create a heatmap of overall performance
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Prepare data for heatmap
        heatmap_data = summary_df[['Model', 'Overall_Accuracy', 'Avg_Precision', 
                                  'Avg_Recall', 'Avg_F1_Score', 'Avg_Specificity']].set_index('Model')
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.4f', 
                   cbar_kws={'label': 'Score'}, ax=ax)
        ax.set_title('Overall Performance Heatmap Across Models')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Models')
        
        plt.tight_layout()
        plt.savefig('images/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[✓] Summary report saved to images/model_comparison_summary.csv")
        print("[✓] Performance heatmap saved to images/performance_heatmap.png")
        
        # Print top performing models
        print("\n" + "="*60)
        print("TOP PERFORMING MODELS SUMMARY")
        print("="*60)
        
        best_accuracy = summary_df.loc[summary_df['Overall_Accuracy'].idxmax()]
        best_f1 = summary_df.loc[summary_df['Avg_F1_Score'].idxmax()]
        
        print(f"Best Overall Accuracy: {best_accuracy['Model']} ({best_accuracy['Overall_Accuracy']:.4f})")
        print(f"Best Average F1-Score: {best_f1['Model']} ({best_f1['Avg_F1_Score']:.4f})")
        
        print("\nDetailed Performance Summary:")
        print(summary_df[['Model', 'Overall_Accuracy', 'Avg_Precision', 
                         'Avg_Recall', 'Avg_F1_Score', 'Avg_Specificity']].round(4).to_string(index=False))


def main():
    """Main function to run the model comparison analysis."""
    print("="*60)
    print("EMOTION DETECTION MODELS COMPARISON ANALYSIS")
    print("="*60)
    
    # Configuration
    csv_path = "data/augmented/train.csv"  # Path to your test data CSV
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load test data
    print("[INFO] Loading test data...")
    data_loaders = load_affectnet_data(csv_path, batch_size=32)
    test_loader = data_loaders['test_loader']
    
    # Initialize comparator
    comparator = ModelComparator(device=device)
    
    # Load all models
    comparator.load_models()
    
    if not comparator.models:
        print("[ERROR] No models were successfully loaded. Exiting...")
        return
    
    # Evaluate all models
    comparator.evaluate_all_models(test_loader)
    
    # Generate comparison plots
    comparator.plot_roc_curves_comparison()
    comparator.plot_metrics_comparison()
    comparator.generate_summary_report()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("- images/roc_curves_comparison.png")
    print("- images/precision_comparison.png") 
    print("- images/recall_comparison.png")
    print("- images/f1_score_comparison.png")
    print("- images/specificity_comparison.png")
    print("- images/performance_heatmap.png")
    print("- images/model_comparison_summary.csv")


if __name__ == "__main__":
    main()