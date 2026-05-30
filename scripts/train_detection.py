"""
Train Detection — Daily Items v3
Usage:
    python scripts/train_detection.py
"""
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")

    model.train(
        data="data_detection.yaml",
        epochs=100,
        batch=8,
        imgsz=640,
        device="0",
        project="runs/detect",
        name="items_v3",
        patience=30,
        workers=0,

        # Augmentation
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        erasing=0.4,

        # Optimizer
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        label_smoothing=0.1,

        save=True,
        save_period=20,
        plots=True,
        verbose=True,
        cache=True,
    )

    print("\n✓ Detection training complete.")
    print("  Best weights: runs/detect/items_v3/weights/best.pt")
