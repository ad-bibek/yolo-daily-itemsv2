"""
Evaluate Trained Models
Usage:
    python scripts/evaluate.py --model runs/detect/items_v3/weights/best.pt --task detect
    python scripts/evaluate.py --model runs/segment/items_v3/weights/best.pt --task segment
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def evaluate_model(model_path, data_yaml, task="detect",
                   split="test", conf=0.25, iou=0.50):
    model = YOLO(model_path)

    print(f"\n{'='*55}")
    print(f"  Model: {model_path}")
    print(f"  Task:  {task}  |  Split: {split}")
    print(f"{'='*55}\n")

    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=640,
        batch=8,
        device="0",
        conf=conf,
        iou=iou,
        plots=True,
        verbose=True,
        task=task,
        workers=0,
    )

    print(f"\n{'─'*55}")
    if task == "detect":
        print(f"  mAP50:      {metrics.box.map50:.4f}")
        print(f"  mAP50-95:   {metrics.box.map:.4f}")
        print(f"  Precision:  {metrics.box.mp:.4f}")
        print(f"  Recall:     {metrics.box.mr:.4f}")
    elif task == "segment":
        print(f"  Box mAP50:  {metrics.box.map50:.4f}")
        print(f"  Mask mAP50: {metrics.seg.map50:.4f}")
        print(f"  Precision:  {metrics.seg.mp:.4f}")
        print(f"  Recall:     {metrics.seg.mr:.4f}")
    print(f"{'─'*55}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--task",   default="detect", choices=["detect", "segment"])
    parser.add_argument("--split",  default="test",   choices=["val", "test"])
    parser.add_argument("--conf",   type=float, default=0.25)
    args = parser.parse_args()

    data_yaml = "data_detection.yaml" if args.task == "detect" else "data_segmentation.yaml"
    evaluate_model(args.model, data_yaml, args.task, args.split, args.conf)


if __name__ == "__main__":
    main()
