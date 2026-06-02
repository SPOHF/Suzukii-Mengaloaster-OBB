# Suzuki Insect Detection (YOLO26 + Roboflow)

This repository contains a deep learning-based object detection system for insect monitoring using Ultralytics YOLO (2026) and datasets exported from Roboflow.

The model is designed for detecting insects in agricultural and experimental environments and can be used for research, monitoring, and automated image analysis.

---

## Project Structure

```
Suzukii-Mengaloaster-OBB/
├── train.py               # Training script
├── predict.py     # Inference / prediction script
├── train/                 # Training dataset (ignored in git)
├── valid/                 # Validation dataset (ignored in git)
├── test/                  # Test dataset (ignored in git)
├── runs/                  # YOLO outputs (ignored in git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model

The model is based on Ultralytics YOLO (2026) and uses pretrained weights:

- Base model: `yolo26[n,s,m,l,xl].pt`
- Framework: Ultralytics YOLO
- Backend: PyTorch
- Hardware support: Apple Silicon (MPS)

---

## Dataset

The dataset is not included in this repository. It must be downloaded separately from Roboflow in YOLO format.

https://app.roboflow.com/spohf-insect-counting/drosophila-mengaloaster-obb

**Steps:**

1. Create a Roboflow account
2. Export dataset in YOLO format
3. Download the dataset
4. Place it in the project root:

```
train/
valid/
test/
data.yaml
```

The `data.yaml` file defines class names and dataset paths.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/suzuki-insect-detection.git
cd suzuki-insect-detection
```

### 2. Create a virtual environment

```bash
python3.14 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Run training using:

```bash
python train.py
```

Default configuration:

- Image size: `640`
- Epochs: `100`
- Batch size: `16`
- Device: `mps` (Apple Silicon GPU if available)

---

## Inference

Run prediction on a single image:

```bash
python predict.py
```

To modify the input image, edit the `source` parameter inside the script:

```python
source="path/to/image.jpg"
```

### Display results instead of saving

```python
results = model.predict(
    source="image.jpg",
    conf=0.25,
    save=False,
    show=True
)
```

---

## Notes

- Dataset must be downloaded from Roboflow separately.
- Model outputs are stored in the `runs/` directory and are not version controlled.
- The system is optimized for Apple Silicon (MPS acceleration).

---
