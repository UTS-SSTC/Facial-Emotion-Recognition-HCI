import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

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
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2, persistent_workers=True
    )

    val_loader = DataLoader(
        AffectNetDataset(val_df, transform=transforms['val']),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        AffectNetDataset(test_df, transform=transforms['test']),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
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