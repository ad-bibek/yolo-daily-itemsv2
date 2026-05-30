"""
Streamlit App — YOLO Daily Items v3
Run: streamlit run app/streamlit_app.py
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import io
from pathlib import Path

st.set_page_config(page_title="YOLO Daily Items", page_icon="🎯", layout="wide")

CLASS_COLORS = [
    (86, 180, 233), (230, 159, 0),  (0, 158, 115), (213, 94, 0),
    (0, 114, 178),  (204, 121, 167),(240, 228, 66), (80, 80, 80),
    (145, 30, 180), (70, 240, 240), (210, 245, 60),
]

CLASSES = ["pen","shoes","phone","bottle","cup","glass",
           "eyewear","wallet","watch","keys","scissors"]

@st.cache_resource
def load_model(path):
    return YOLO(path)

def draw_detection(image, results, conf_thresh):
    img = image.copy()
    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf < conf_thresh:
                continue
            cls_id = int(box.cls)
            color  = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            label  = f"{r.names[cls_id]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(img, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return img

def draw_segmentation(image, results, conf_thresh, alpha=0.45):
    img = image.copy()
    overlay = image.copy()
    dets = []
    for r in results:
        if r.masks is None:
            return draw_detection(image, results, conf_thresh), []
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
            cv2.rectangle(img, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(img, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            dets.append({"class": r.names[cls_id], "confidence": round(conf, 4)})
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    return img, dets

def find_weights():
    all_w = list(Path("runs").rglob("best.pt")) if Path("runs").exists() else []
    det_w = [w for w in all_w if "detect" in str(w) and "seg" not in str(w)]
    seg_w = [w for w in all_w if "seg" in str(w)]
    return det_w, seg_w

def img_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# UI
st.title("🎯 YOLO Daily Items Detector")
st.markdown("Upload a photo to detect everyday objects.")

with st.sidebar:
    st.header("⚙️ Settings")
    task = st.radio("Task", ["Detection", "Segmentation"])
    det_w, seg_w = find_weights()
    weights = det_w if task == "Detection" else seg_w
    weight_names = [str(w) for w in weights]

    if weight_names:
        selected = st.selectbox("Model", weight_names)
    else:
        st.error("No weights found. Train first.")
        selected = None

    conf  = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
    alpha = st.slider("Mask opacity", 0.1, 0.9, 0.45, 0.05) if task == "Segmentation" else 0.45

    st.divider()
    st.markdown("**Classes:**")
    for cls in CLASSES:
        st.markdown(f"• {cls}")

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","bmp","webp"])

if uploaded and selected:
    pil_img = ImageOps.exif_transpose(Image.open(uploaded).convert("RGB"))
    np_img  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)

    with st.spinner("Running inference..."):
        model   = load_model(selected)
        results = model.predict(source=np_img, conf=conf, verbose=False)
        if task == "Detection":
            out_np = draw_detection(np_img, results, conf)
            dets = [{"class": r.names[int(b.cls)], "confidence": round(float(b.conf),4)}
                    for r in results for b in r.boxes if float(b.conf) >= conf]
        else:
            out_np, dets = draw_segmentation(np_img, results, conf, alpha)

    out_pil = Image.fromarray(cv2.cvtColor(out_np, cv2.COLOR_BGR2RGB))
    with col2:
        st.subheader(f"Result — {task}")
        st.image(out_pil, use_container_width=True)

    st.download_button("⬇ Download result", data=img_to_bytes(out_pil),
                       file_name=f"result_{uploaded.name}", mime="image/png")

    if dets:
        st.subheader(f"Detected: {len(dets)} object(s)")
        from collections import Counter
        counts = Counter(d["class"] for d in dets)
        cols = st.columns(min(len(counts), 5))
        for i, (cls, cnt) in enumerate(counts.items()):
            cols[i % len(cols)].metric(cls, cnt)
    else:
        st.warning("No objects detected. Try lowering confidence threshold.")
elif not selected:
    st.info("Train a model first, then restart the app.")
else:
    st.info("Upload an image to get started.")
