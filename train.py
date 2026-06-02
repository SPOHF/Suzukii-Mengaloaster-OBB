import torch
from ultralytics import YOLO

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()

model = YOLO("yolo26n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=device,
    patience=20
)