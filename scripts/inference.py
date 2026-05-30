"""
Run Inference — Detection & Segmentation
Usage:
    python scripts/inference.py --model runs/detect/items_v3/weights/best.pt --source photo.jpg
    python scripts/inference.py --model runs/segment/items_v3/weights/best.pt --source photo.jpg --task segment
    python scripts/inference.py --model runs/detect/items_v3/weights/best.pt --source webcam
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASS_COLORS = [
    (86, 180, 233), (230, 159, 0),  (0, 158, 115), (213, 94, 0),
    (0, 114, 178),  (204, 121, 167),(240, 228, 66), (80, 80, 80),
    (145, 30, 180), (70, 240, 240), (210, 245, 60),
]


def draw_detection(image, results, conf_thresh=0.25):
    img = image.copy()
    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf < conf_thresh:
                continue
            cls_id = int(box.cls)
            label  = f"{r.names[cls_id]} {conf:.2f}"
            color  = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_segmentation(image, results, conf_thresh=0.25, alpha=0.45):
    img     = image.copy()
    overlay = image.copy()

    for r in results:
        if r.masks is None:
            return draw_detection(image, results, conf_thresh)
        h, w = image.shape[:2]
        for i, mask in enumerate(r.masks.data.cpu().numpy()):
            if i >= len(r.boxes):
                break
            conf = float(r.boxes[i].conf)
            if conf < conf_thresh:
                continue
            cls_id = int(r.boxes[i].cls)
            color  = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[mask_r > 0.5] = color
            contours, _ = cv2.findContours(
                mask_r.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 2)
            x1, y1, x2, y2 = map(int, r.boxes[i].xyxy[0].tolist())
            label = f"{r.names[cls_id]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def run_on_image(model, path, task, conf, save, output_dir):
    image   = cv2.imread(path)
    if image is None:
        print(f"[ERROR] Cannot read: {path}")
        return
    results = model.predict(source=image, conf=conf, verbose=False)
    out     = draw_detection(image, results, conf) if task == "detect" else draw_segmentation(image, results, conf)
    cv2.imshow("YOLO Inference", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(output_dir) / Path(path).name
        cv2.imwrite(str(out_path), out)
        print(f"[INFO] Saved: {out_path}")


def run_on_video(model, source, task, conf, save, output_dir):
    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return
    writer = None
    if save and source != "webcam":
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(Path(output_dir) / "output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("[INFO] Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=conf, verbose=False)
        vis = draw_detection(frame, results, conf) if task == "detect" else draw_segmentation(frame, results, conf)
        cv2.imshow("YOLO Inference", vis)
        if writer:
            writer.write(vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--task",   default="detect", choices=["detect", "segment"])
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--save",   action="store_true")
    parser.add_argument("--output", default="runs/inference")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.source == "webcam":
        run_on_video(model, "webcam", args.task, args.conf, args.save, args.output)
    else:
        src = Path(args.source)
        if not src.exists():
            print(f"[ERROR] Source not found: {src}")
            return
        if src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            run_on_video(model, str(src), args.task, args.conf, args.save, args.output)
        elif src.is_dir():
            for img in src.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    run_on_image(model, str(img), args.task, args.conf, args.save, args.output)
        else:
            run_on_image(model, str(src), args.task, args.conf, args.save, args.output)


if __name__ == "__main__":
    main()
