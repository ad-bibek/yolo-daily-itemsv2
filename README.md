# YOLO Daily Items — Detection & Segmentation v3

Custom YOLOv8 project for detecting and segmenting 11 everyday household items.

## Classes
| ID | Class |
|---|---|
| 0 | pen |
| 1 | shoes |
| 2 | phone |
| 3 | bottle |
| 4 | cup |
| 5 | glass |
| 6 | eyewear |
| 7 | wallet |
| 8 | watch |
| 9 | keys |
| 10 | scissors |

## Project Structure
```
yolo_project/
├── dataset/
│   ├── images/train, val, test
│   └── labels/train, val, test
├── scripts/
│   ├── import_phone_photos.py
│   ├── 02_split_dataset.py
│   ├── train_detection.py
│   ├── train_segmentation.py
│   ├── evaluate.py
│   └── inference.py
├── app/
│   └── streamlit_app.py
├── data_detection.yaml
├── data_segmentation.yaml
└── requirements.txt
```

## Quick Start
```bash
# Install
pip install -r requirements.txt

# Import photos
python scripts/import_phone_photos.py --input phone_dump/ --output dataset/raw_images --mode subfolders

# Verify dataset
python scripts/02_split_dataset.py --verify

# Train
python scripts/train_detection.py
python scripts/train_segmentation.py

# Evaluate
python scripts/evaluate.py --model runs/detect/items_v3/weights/best.pt --task detect

# Inference
python scripts/inference.py --model runs/detect/items_v3/weights/best.pt --source photo.jpg

# Streamlit app
streamlit run app/streamlit_app.py
```

## Results
| Model | mAP50 |
|---|---|
| Detection (YOLOv8s) | TBD |
| Segmentation (YOLOv8s-seg) | TBD |

"Dataset available on Roboflow: https://app.roboflow.com/bibeks-workspace-zf5a0/yolo-daily-items-v2/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true"
