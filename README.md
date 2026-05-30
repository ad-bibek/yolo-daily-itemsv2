# YOLO Daily Items — Detection \& Segmentation v3

Custom YOLOv8 project for detecting and segmenting 11 everyday household items.

## Classes

|ID|Class|
|-|-|
|0|pen|
|1|shoes|
|2|phone|
|3|bottle|
|4|cup|
|5|glass|
|6|eyewear|
|7|wallet|
|8|watch|
|9|keys|
|10|scissors|

## Project Structure

```
yolo\_project/
├── dataset/
│   ├── images/train, val, test
│   └── labels/train, val, test
├── scripts/
│   ├── import\_phone\_photos.py
│   ├── 02\_split\_dataset.py
│   ├── train\_detection.py
│   ├── train\_segmentation.py
│   ├── evaluate.py
│   └── inference.py
├── app/
│   └── streamlit\_app.py
├── data\_detection.yaml
├── data\_segmentation.yaml
└── requirements.txt
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Import photos
python scripts/import\_phone\_photos.py --input phone\_dump/ --output dataset/raw\_images --mode subfolders

# Verify dataset
python scripts/02\_split\_dataset.py --verify

# Train
python scripts/train\_detection.py
python scripts/train\_segmentation.py

# Evaluate
python scripts/evaluate.py --model runs/detect/items\_v3/weights/best.pt --task detect

# Inference
python scripts/inference.py --model runs/detect/items\_v3/weights/best.pt --source photo.jpg

# Streamlit app
streamlit run app/streamlit\_app.py
```

## Results

|Model|mAP50|
|-|-|
|Detection (YOLOv8s)|0.869|
|Segmentation (YOLOv8s-seg)|0.873|



