#!/usr/bin/env python
from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List
import albumentations as A
import cv2
import pandas as pd
from tqdm import tqdm

# Mapping of severity parameters for augmentation transforms
SEVERITY_PARAMS = {
    'light': {'p_base': 0.3, 'bc_lim': 0.1, 'hue_lim': 5,  'sat_lim': 10, 'val_lim': 10, 'blur_lim': 3},
    'medium':{'p_base': 0.5, 'bc_lim': 0.2, 'hue_lim': 10, 'sat_lim': 20, 'val_lim': 20, 'blur_lim': 5},
    'heavy': {'p_base': 0.7, 'bc_lim': 0.3, 'hue_lim': 15, 'sat_lim': 30, 'val_lim': 30, 'blur_lim': 7},
}

def create_augmentation_transforms(severity: str = 'medium') -> List[A.Compose]:
    """
    Create a list of Albumentations augmentation pipelines based on the given severity level.
    Compatible with Albumentations v2.0.6.

    Parameters
    ----------
    severity : {'light', 'medium', 'heavy'}, optional
        The intensity level of augmentation. Default is 'medium'.
        Controls the probability and strength of transformations.

    Returns
    -------
    List[albumentations.Compose]
        A list of composed Albumentations pipelines for image augmentation.
    """

    params = SEVERITY_PARAMS[severity]
    p = params['p_base']
    bc = params['bc_lim']
    hue = params['hue_lim']
    sat = params['sat_lim']
    val = params['val_lim']
    blur = params['blur_lim']

    pipelines: List[A.Compose] = [
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=bc, contrast_limit=bc, p=p),
            A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=sat, val_shift_limit=val, p=p),
        ]),
        A.Compose([
            A.GaussianBlur(blur_limit=blur, p=p),
        ]),
        A.Compose([
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=p),
            A.RandomBrightnessContrast(brightness_limit=bc, contrast_limit=bc, p=p),
        ]),
        A.Compose([
            A.ImageCompression(compression_type='jpeg', p=p),
        ]),
        A.Compose([
            A.HorizontalFlip(p=p),
            A.RandomRotate90(p=p),
        ]),
        A.Compose([
            A.CLAHE(clip_limit=4.0, p=p),
            A.CoarseDropout(
                p=p
            )
            ,
        ]),
        A.Compose([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=p),
            A.ChannelShuffle(p=p * 0.5),
        ]),
        A.Compose([A.MotionBlur(blur_limit=7, p=p)]),
        A.Compose([A.ElasticTransform(alpha=1, sigma=50, p=p)]),
        A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.3, p=p)]),
        A.Compose([
            A.RandomRain(
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                p=p
            )
        ]),
    ]

    return pipelines


def augment_split(
    split_dir: Path,
    dst_root: Path,
    target: int,
    severity: str,
    img_size: tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """
    Augment images in a single data split and generate an index DataFrame.

    Parameters
    ----------
    split_dir : Path
        Path to the processed split directory (e.g., processed/train).
    dst_root : Path
        Destination root for augmented images of this split.
    target : int
        Desired total number of samples per class (original + augmented).
    severity : {'light', 'medium', 'heavy'}
        Augmentation severity level.
    img_size : tuple of int, optional
        Output image size as (width, height), default (224, 224).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['file_name', 'file_path', 'expression']
        listing all images (original and augmented) in this split.
    """
    transforms = create_augmentation_transforms(severity)
    records: list[dict[str, str]] = []
    data_dir = dst_root.parent

    # Iterate through each expression class directory
    for expr_dir in sorted(split_dir.iterdir()):
        if not expr_dir.is_dir():
            continue
        expr = expr_dir.name
        dst_expr = dst_root / expr
        dst_expr.mkdir(parents=True, exist_ok=True)

        # Copy original images
        originals = list(expr_dir.glob('*.jpg'))
        for f in originals:
            dst = dst_expr / f.name
            dst.write_bytes(f.read_bytes())
            relative_path = dst.relative_to(data_dir).as_posix()
            records.append({
                'file_name': f.name,
                'file_path': f'./data/{relative_path}',  # ←★ 加前缀
                'expression': expr
            })

        # Determine how many augmented images are needed
        need = max(0, target - len(originals))
        for i in range(need):
            src = random.choice(originals)
            img = cv2.imread(str(src))
            transform = random.choice(transforms)
            aug_img = transform(image=img)['image']
            aug_img = cv2.resize(aug_img, img_size)
            dst_name = f'aug_{i:05d}_{expr}.jpg'
            dst_file = dst_expr / dst_name
            cv2.imwrite(str(dst_file), aug_img)
            relative_path = dst_file.relative_to(data_dir).as_posix()
            records.append({
                'file_name': dst_name,
                'file_path': f'./data/{relative_path}',  # ←★ 加前缀
                'expression': expr
            })

    return pd.DataFrame(records)


def main() -> None:
    """
    CLI entry point for dataset augmentation.

    Parses arguments and performs augmentation on the train split,
    copying val/test splits without augmentation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description='Augment processed AffectNet splits')
    parser.add_argument(
        '--processed_root', required=True,
        help='Path to processed dataset root (contains train/val/test)'
    )
    parser.add_argument(
        '--aug_root', default=None,
        help='Destination directory for augmented data'
    )
    parser.add_argument(
        '--target', type=int, default=8000,
        help='Target samples per class for training split'
    )
    parser.add_argument(
        '--severity', choices=list(SEVERITY_PARAMS), default='medium',
        help='Augmentation severity level'
    )
    args = parser.parse_args()

    src_root = Path(args.processed_root).resolve()
    dst_root = Path(args.aug_root).resolve() if args.aug_root else src_root.parent / 'augmented'
    dst_root.mkdir(parents=True, exist_ok=True)

    # Augment training split
    df_train = augment_split(src_root / 'train', dst_root / 'train', args.target, args.severity)
    df_train.to_csv(dst_root / 'train.csv', index=False)

    # Copy validation and test splits without augmentation
    for split in ('val', 'test'):
        records: list[dict[str, str]] = []
        for expr_dir in (src_root / split).iterdir():
            if not expr_dir.is_dir():
                continue
            dst_expr = dst_root / split / expr_dir.name
            dst_expr.mkdir(parents=True, exist_ok=True)
            for f in expr_dir.glob('*.jpg'):
                dst = dst_expr / f.name
                dst.write_bytes(f.read_bytes())
                relative_path = dst.relative_to(dst_root)
                records.append({'file_name': f.name, 'file_path': str(relative_path), 'expression': expr_dir.name})
        pd.DataFrame(records).to_csv(dst_root / f'{split}.csv', index=False)

def augment_affectnet(
    processed_root: str,
    aug_root: str,
    target: int = 8000,
    severity: str = "medium"
) -> dict[str, pd.DataFrame]:
    """
    Perform data augmentation on the training split of AffectNet.
    Validation and test splits are copied without augmentation.
    Outputs CSV metadata files and returns the metadata as dictionaries of DataFrames.

    Parameters
    ----------
    processed_root : str
        Path to the directory with preprocessed AffectNet dataset (must contain train/val/test folders).

    aug_root : str
        Destination path where augmented images and metadata will be saved.

    target : int, optional
        Target total number of images per class in the training split after augmentation. Default is 8000.

    severity : {'light', 'medium', 'heavy'}, optional
        Augmentation intensity level. Default is 'medium'.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing DataFrames for each split: {'train': ..., 'val': ..., 'test': ...}
        with columns ['file_name', 'file_path', 'expression'].
    """
    processed_root = Path(processed_root)
    aug_root = Path(aug_root)
    aug_root.mkdir(parents=True, exist_ok=True)
    data_dir = aug_root.parent
    # Augment training split
    df_train = augment_split(
        split_dir=processed_root / "train",
        dst_root=aug_root / "train",
        target=target,
        severity=severity
    )
    df_train.to_csv(aug_root / "train.csv", index=False)

    result = {"train": df_train}

    # Copy val/test splits without augmentation
    for split in ("val", "test"):
        records = []
        for expr_dir in (processed_root / split).iterdir():
            if not expr_dir.is_dir():
                continue
            dst_expr = aug_root / split / expr_dir.name
            dst_expr.mkdir(parents=True, exist_ok=True)
            for f in expr_dir.glob("*.jpg"):
                dst = dst_expr / f.name
                dst.write_bytes(f.read_bytes())
                rel_path = dst.relative_to(data_dir).as_posix()  # ←★ 改基准
                records.append({
                    "file_name": f.name,
                    "file_path": f"./data/{rel_path}",  # ←★ 加前缀
                    "expression": expr_dir.name
                })
        df_split = pd.DataFrame(records)
        df_split.to_csv(aug_root / f"{split}.csv", index=False)
        result[split] = df_split

    return result
