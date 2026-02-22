"""
CivicLens - EfficientNetV2 Inference
Replaces YOLO inference with the higher-accuracy classification model.
Drop best_roadscan.pt in the backend/ folder.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image
import os

MODEL_PATH = "best_roadscan.pt"
IMAGE_SIZE = 224

_model = None
_classes = None

def get_model():
    global _model, _classes
    if _model is not None:
        return _model, _classes

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'. Drop best_roadscan.pt in backend/")

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    _classes = ckpt["classes"]
    num_classes = len(_classes)

    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _model = model
    return _model, _classes


SEVERITY_MAP = {
    "sinkhole":           "High",
    "pothole":            "High",
    "water_leakage":      "High",
    "garbage_overflow":   "Medium",
    "broken_streetlight": "Medium",
    "broken_sidewalk":    "Low",
}

DEPARTMENT_MAP = {
    "pothole":            "Roads & Infrastructure Department",
    "sinkhole":           "Roads & Infrastructure Department",
    "water_leakage":      "Water & Sewage Department",
    "garbage_overflow":   "Sanitation & Waste Management",
    "broken_streetlight": "Electrical & Street Lighting Department",
    "broken_sidewalk":    "Public Works Department",
}

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_inference(image_path: str) -> dict | None:
    model, classes = get_model()

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, idx = probs.max(0)

    confidence = float(confidence)
    class_name = classes[int(idx)]

    # Reject if model is not confident enough
    if confidence < 0.4:
        return None

    return {
        "class": class_name,
        "confidence": round(confidence, 3),
        "severity": SEVERITY_MAP.get(class_name, "Medium"),
        "department": DEPARTMENT_MAP.get(class_name, "Municipal Corporation"),
        "bbox": None,  # classification model has no bounding box
        "annotated_url": image_path,
    }