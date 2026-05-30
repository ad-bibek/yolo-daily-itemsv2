"""
Split Dataset into Train / Val / Test
Usage:
    python scripts/02_split_dataset.py --verify
    python scripts/02_split_dataset.py --images dataset/raw_images --labels dataset/raw_labels
"""
import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(images_dir, labels_dir, output_dir,
                  train_ratio=0.7, val_ratio=0.2, seed=42):
    random.seed(seed)
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    image_files = sorted([
        f for f in images_path.glob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])

    paired = []
    for img in image_files:
        lbl = labels_path / (img.stem + ".txt")
        if lbl.exists():
            paired.append((img, lbl))

    if not paired:
        raise ValueError("No image-label pairs found.")

    random.shuffle(paired)
    n = len(paired)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": paired[:n_train],
        "val":   paired[n_train:n_train + n_val],
        "test":  paired[n_train + n_val:],
    }

    print(f"\n[INFO] Total: {n}  Train: {n_train}  Val: {n_val}  Test: {n - n_train - n_val}")

    for split_name, pairs in splits.items():
        img_out = output_path / "images" / split_name
        lbl_out = output_path / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_src, lbl_src in pairs:
            shutil.copy2(img_src, img_out / img_src.name)
            shutil.copy2(lbl_src, lbl_out / lbl_src.name)
        print(f"  [{split_name:5s}] {len(pairs)} samples")

    print("\n[DONE] Dataset split complete.")


def verify_dataset(dataset_dir):
    base = Path(dataset_dir)
    print("\n── Dataset verification ──────────────────────")
    for split in ["train", "val", "test"]:
        imgs = list((base / "images" / split).glob("*")) if (base / "images" / split).exists() else []
        lbls = list((base / "labels" / split).glob("*.txt")) if (base / "labels" / split).exists() else []
        print(f"  {split:5s}  images: {len(imgs):4d}  labels: {len(lbls):4d}")
    print("──────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",  default="dataset/raw_images")
    parser.add_argument("--labels",  default="dataset/raw_labels")
    parser.add_argument("--output",  default="dataset")
    parser.add_argument("--train",   type=float, default=0.7)
    parser.add_argument("--val",     type=float, default=0.2)
    parser.add_argument("--verify",  action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output)
    else:
        split_dataset(args.images, args.labels, args.output, args.train, args.val)
        verify_dataset(args.output)


if __name__ == "__main__":
    main()
