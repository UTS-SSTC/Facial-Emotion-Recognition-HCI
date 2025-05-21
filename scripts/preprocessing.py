#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Mapping from numeric labels to expression names
LABEL_MAP: Dict[int, str] = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
    7: "contempt",
}

SPLITS: Tuple[str, ...] = ("train", "val", "test")

###############################################################################
# Renaming utilities                                                          #
###############################################################################

def rename_images(
    root: Path,
    *,
    width: int = 5,
    dry_run: bool = True,
    out_csv: Optional[Path] = None,
) -> None:
    """
    Rename images to `<local_index>_<expression>.jpg` within each class folder.

    Parameters
    ----------
    root : Path
        AffectNet root directory containing `train`, `val`, and `test` subfolders.
    width : int, default=5
        Zero-padding width for the per-class running index.
    dry_run : bool, default=True
        If True, no files are renamed on disk; CSV mapping is still generated if requested.
    out_csv : Path or None, optional
        Path to output CSV file recording old→new filenames.
        If None, no CSV is written.

    Returns
    -------
    None
    """
    records: List[Dict[str, str | int]] = []

    for split in SPLITS:
        for label, expr_name in LABEL_MAP.items():
            src_dir = root / split / str(label)
            if not src_dir.exists():
                continue

            local_idx = 0
            for file in sorted(src_dir.iterdir()):
                if not file.is_file():
                    continue
                new_name = f"{local_idx:0{width}d}_{expr_name}.jpg"
                new_path = file.with_name(new_name)
                if not dry_run:
                    file.rename(new_path)

                data_dir = root.parent
                records.append({
                    "old_path": f"./data/{file.relative_to(data_dir).as_posix()}",
                    "new_path": f"./data/{new_path.relative_to(data_dir).as_posix()}",
                    "split": split,
                    "expression": expr_name,
                })
                local_idx += 1

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")


###############################################################################
# Copy & resize utilities                                                     #
###############################################################################

def process_split(
    split: str,
    *,
    src_root: Path,
    dst_root: Path,
    img_size: Tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """
    Copy and resize images for one split, returning an index DataFrame.

    Parameters
    ----------
    split : str
        Name of the split to process ('train', 'val', or 'test').
    src_root : Path
        Root directory containing the original split folders (named by label).
    dst_root : Path
        Base directory where processed images will be saved; within it will be
        subfolders for each split and expression.
    img_size : tuple of int, default=(224, 224)
        Target image size as (width, height).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['file_name', 'file_path', 'expression'], listing
        every image saved under `dst_root/split/...`.
    """
    records: List[Dict[str, str | int]] = []

    for label, expr_name in LABEL_MAP.items():
        src_dir = src_root / split / str(label)
        if not src_dir.exists():
            continue

        dst_dir = dst_root / split / expr_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        local_idx = 0
        for file in tqdm(sorted(src_dir.iterdir()), desc=f"{split}/{expr_name}"):
            if not file.is_file():
                continue

            dst_name = f"{local_idx:05d}_{expr_name}.jpg"
            dst_file = dst_dir / dst_name

            try:
                img = Image.open(file).convert("RGB")
                img.resize(img_size).save(dst_file)
            except Exception:
                # Skip any files that cannot be opened or saved
                continue

            data_dir = dst_root.parent
            relative_path = dst_file.relative_to(data_dir).as_posix()
            records.append({
                "file_name": dst_name,
                "file_path": f"./data/{relative_path}",
                "expression": expr_name,
            })

            local_idx += 1

    return pd.DataFrame(records)


###############################################################################
# Main entry                                                                  #
###############################################################################

def main() -> None:
    """
    CLI entry point for preprocessing AffectNet.

    Parses arguments to optionally rename images in-place, copy and resize
    them into a new directory structure, and export CSV indexes.
    """
    parser = argparse.ArgumentParser(description="Preprocess AffectNet images")
    parser.add_argument(
        "--affectnet_root",
        required=True,
        help="Path to the AffectNet root directory"
    )
    parser.add_argument(
        "--processed_root",
        default=None,
        help="Destination directory for processed data (default: sibling 'processed')"
    )
    parser.add_argument(
        "--img_size",
        nargs=2,
        type=int,
        default=[224, 224],
        metavar=("W", "H"),
        help="Resize images to this width and height"
    )
    parser.add_argument(
        "--rename",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to rename images in-place before processing"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=5,
        help="Zero-padding width for renaming"
    )
    parser.add_argument(
        "--dry_run",
        type=lambda x: x.lower() == "true",
        default=False,
        help="If true, simulate renaming without modifying files"
    )
    args = parser.parse_args()

    src_root = Path(args.affectnet_root).resolve()
    dst_root = (
        Path(args.processed_root).resolve()
        if args.processed_root
        else src_root.parent / "processed"
    )
    dst_root.mkdir(parents=True, exist_ok=True)

    # Rename images in-place if requested
    if args.rename:
        rename_images(
            root=src_root,
            width=args.width,
            dry_run=args.dry_run,
            out_csv=dst_root / "rename_map.csv",
        )
        if args.dry_run:
            return

    # Copy, resize, and index each split
    for split in SPLITS:
        df_split = process_split(
            split=split,
            src_root=src_root,
            dst_root=dst_root,
            img_size=tuple(args.img_size),
        )
        df_split.to_csv(dst_root / f"{split}.csv", index=False)

def process_affectnet(
    affectnet_root: str,
    processed_root: str,
    img_size=(224, 224),
    rename=True,
    rename_width=5,
    dry_run=False,
) -> dict[str, pd.DataFrame]:
    """
    Preprocess the AffectNet dataset by optionally renaming and resizing images.
    Generates structured CSV metadata for train, val, and test splits.

    Parameters
    ----------
    affectnet_root : str
        Path to the original AffectNet dataset containing 'train', 'val', and 'test' folders.

    processed_root : str
        Destination path where processed images and CSV metadata will be saved.

    img_size : tuple of int, optional
        Target image resolution as (width, height). Default is (224, 224).

    rename : bool, optional
        If True, images will be renamed to a consistent pattern per expression class. Default is True.

    rename_width : int, optional
        Zero-padding width for renamed images (e.g., 00001_happy.jpg). Default is 5.

    dry_run : bool, optional
        If True, only simulate renaming without making changes. Default is False.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing DataFrames for each split: {'train': ..., 'val': ..., 'test': ...}
        with columns ['file_name', 'file_path', 'expression'].
    """
    affectnet_root = Path(affectnet_root)
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    if rename:
        rename_images(
            root=affectnet_root,
            width=rename_width,
            dry_run=dry_run,
            out_csv=processed_root / "rename_map.csv"
        )
        if dry_run:
            return {}

    result = {}
    for split in ("train", "val", "test"):
        df_split = process_split(
            split=split,
            src_root=affectnet_root,
            dst_root=processed_root,
            img_size=img_size
        )
        df_split.to_csv(processed_root / f"{split}.csv", index=False)
        result[split] = df_split

    return result


