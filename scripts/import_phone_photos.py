"""
Phone Photo Importer — Daily Items v3
Usage:
    python scripts/import_phone_photos.py --input phone_dump/ --output dataset/raw_images --mode subfolders
"""
import shutil
import argparse
from pathlib import Path

CLASSES = [
    "pen", "shoes", "phone", "bottle", "cup",
    "glass", "eyewear", "wallet", "watch", "keys", "scissors",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}


def organize_from_subfolders(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {cls: 0 for cls in CLASSES}

    for cls in CLASSES:
        src_dir = input_dir / cls
        if not src_dir.exists():
            print(f"[WARN] No folder found for class '{cls}'")
            continue
        dest_dir = output_dir / cls
        dest_dir.mkdir(exist_ok=True)
        imgs = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        for i, img in enumerate(imgs):
            new_name = f"{cls}_{i+1:04d}{img.suffix.lower()}"
            shutil.copy2(img, dest_dir / new_name)
            counts[cls] += 1

    print("\n── Organized images ──────────────────────────")
    total = 0
    for cls, cnt in counts.items():
        status = "✓" if cnt >= 50 else "⚠ low" if cnt > 0 else "✗ missing"
        print(f"  {cls:<22} {cnt:>4} images  {status}")
        total += cnt
    print(f"  {'TOTAL':<22} {total:>4} images")
    print("──────────────────────────────────────────────")

    low = [cls for cls, cnt in counts.items() if cnt < 50]
    if low:
        print(f"\n[TIP] These classes need more images:")
        for cls in low:
            print(f"      - {cls}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="phone_dump")
    parser.add_argument("--output", default="dataset/raw_images")
    parser.add_argument("--mode",   default="subfolders")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    organize_from_subfolders(input_dir, output_dir)
    print(f"\n[NEXT] Upload dataset/raw_images to Roboflow and annotate.")


if __name__ == "__main__":
    main()
