"""
RoadScan AI - EfficientNetV2 Inference
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

MODEL_PATH = "final_model.pt"
IMAGE_SIZE = 224

_model   = None
_classes = None

def get_model():
    global _model, _classes
    if _model is not None:
        return _model, _classes

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'.")

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    _classes    = ckpt["classes"]
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
    "potholes":                "High",
    "cracked_pavement":        "Medium",
    "road_debris_obstruction": "High",
    "broken_road_signs":       "Medium",
    "faded_lane_markings":     "Low",
    "normal_road":             "None",
}

DEPARTMENT_MAP = {
    "potholes":                "Roads & Infrastructure Department",
    "cracked_pavement":        "Roads & Infrastructure Department",
    "road_debris_obstruction": "Roads & Infrastructure Department",
    "broken_road_signs":       "Traffic & Signage Department",
    "faded_lane_markings":     "Roads & Infrastructure Department",
    "normal_road":             None,
}

DISPLAY_NAME = {
    "potholes":                "Pothole",
    "cracked_pavement":        "Cracked Pavement",
    "road_debris_obstruction": "Road Debris / Obstruction",
    "broken_road_signs":       "Broken / Damaged Road Sign",
    "faded_lane_markings":     "Faded Lane Markings",
    "normal_road":             "No Issue Detected",
}

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

NO_ISSUE_THRESHOLD = 0.60   # below this → no issue filed
REJECT_THRESHOLD   = 0.85   # below this → uncertain, flag for manual review

def run_inference(image_path: str) -> dict | None:
    model, classes = get_model()

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits          = model(tensor)
        probs           = torch.softmax(logits, dim=1)[0]
        confidence, idx = probs.max(0)

    confidence = float(confidence)
    class_name = classes[int(idx)]

    # Gate 1 — not confident enough to flag anything
    if confidence < NO_ISSUE_THRESHOLD:
        return {
            "class":        "normal_road",
            "display_name": "No Issue Detected",
            "confidence":   round(confidence, 3),
            "severity":     "None",
            "department":   None,
            "flagged":      False,
            "note":         "Confidence below threshold — no issue filed.",
            "bbox":         None,
            "annotated_url": image_path,
        }

    # Gate 2 — model sees something but isn't sure
    if confidence < REJECT_THRESHOLD:
        return {
            "class":        "unknown",
            "display_name": "Uncertain — Needs Review",
            "confidence":   round(confidence, 3),
            "severity":     "Unknown",
            "department":   "Municipal Corporation",
            "flagged":      True,
            "note":         "Low confidence — flagged for manual review.",
            "bbox":         None,
            "annotated_url": image_path,
        }

    # Gate 3 — normal road predicted with high confidence
    if class_name == "normal_road":
        return {
            "class":        "normal_road",
            "display_name": "No Issue Detected",
            "confidence":   round(confidence, 3),
            "severity":     "None",
            "department":   None,
            "flagged":      False,
            "note":         "Road appears normal.",
            "bbox":         None,
            "annotated_url": image_path,
        }

    # High-confidence damage prediction
    return {
        "class":        class_name,
        "display_name": DISPLAY_NAME.get(class_name, class_name),
        "confidence":   round(confidence, 3),
        "severity":     SEVERITY_MAP.get(class_name, "Medium"),
        "department":   DEPARTMENT_MAP.get(class_name, "Municipal Corporation"),
        "flagged":      True,
        "note":         None,
        "bbox":         None,
        "annotated_url": image_path,
    }