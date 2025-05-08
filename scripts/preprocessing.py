#!/usr/bin/env python
"""preprocess_affectnet.py
A simple preprocessing script for an *aligned* AffectNet dataset.
It can optionally rename files, resize/copy them to a clean directory
and write CSV index files for each split (train/val/test).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Mapping AffectNet numeric labels → English class names
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
    """Rename images to ``<local_index>_<expr_name>.jpg`` inside each class.

    Parameters
    ----------
    root : Path
        AffectNet root directory containing ``train``, ``val`` and ``test``.
    width : int, default 5
        Zero‑padding width for the per‑class running index.
    dry_run : bool, default True
        If *True* no files are modified; a rename map is still produced when
        *out_csv* is supplied.
    out_csv : Path | None
        File to store the rename map; ignored when *None*.
    """

    records: List[Dict[str, str | int]] = []

    for split in SPLITS:
        for label, expr_name in LABEL_MAP.items():
            src_dir = root / split / str(label)
            if not src_dir.exists():
                continue
            local_idx = 0  # restart counting for each expression class
            for file in sorted(src_dir.iterdir()):
                if not file.is_file():
                    continue
                new_name = f"{local_idx:0{width}d}_{expr_name}.jpg"
                new_path = file.with_name(new_name)
                if not dry_run:
                    file.rename(new_path)
                records.append({
                    "old_path": str(file),
                    "new_path": str(new_path),
                    "split": split,
                    "expression": expr_name,
                })
                local_idx += 1

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")


# Copy & CSV utilities
def process_split(
    split: str,
    *,
    src_root: Path,
    dst_root: Path,
    img_size: Tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """Copy one split to *dst_root* and return an index DataFrame."""

    records: List[Dict[str, str | int]] = []

    for label, expr_name in LABEL_MAP.items():
        src_dir = src_root / split / str(label)
        if not src_dir.exists():
            continue
        dst_dir = dst_root / split / expr_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Local index counter to keep destination file names unique per class
        local_idx = 0
        for file in tqdm(sorted(src_dir.iterdir()), desc=f"{split}/{expr_name}"):
            if not file.is_file():
                continue
            dst_name = f"{local_idx:05d}_{expr_name}.jpg"
            dst_file = dst_dir / dst_name
            try:
                img = Image.open(file).convert("RGB")
                img.resize(img_size).save(dst_file)
                records.append({
                    "file_name": dst_name,
                    "file_path": str(dst_file),
                    "expression": expr_name,
                })
                local_idx += 1
            except Exception:
                continue
    return pd.DataFrame(records)

# Main entry
def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--affectnet_root", required=True, help="Path to AffectNet root")
    parser.add_argument("--processed_root", default=None, help="Output directory for processed data")
    parser.add_argument("--img_size", nargs=2, type=int, default=[224, 224], metavar=("W", "H"))
    parser.add_argument("--rename", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--width", type=int, default=5, help="Zero‑padding width for renaming")
    parser.add_argument("--dry_run", type=lambda x: x.lower() == "true", default=False)
    args = parser.parse_args()

    src_root = Path(args.affectnet_root).resolve()
    dst_root = Path(args.processed_root).resolve() if args.processed_root else src_root.parent / "processed"
    dst_root.mkdir(parents=True, exist_ok=True)

    # Optional in‑place renaming
    if args.rename:
        rename_images(
            root=src_root,
            width=args.width,
            dry_run=args.dry_run,
            out_csv=dst_root / "rename_map.csv",
        )
        if args.dry_run:
            return

    # Copy into processed directory and export CSV files
    for split in SPLITS:
        df_split = process_split(
            split=split,
            src_root=src_root,
            dst_root=dst_root,
            img_size=tuple(args.img_size),
        )
        df_split.to_csv(dst_root / f"{split}.csv", index=False)


if __name__ == "__main__":
    main()
