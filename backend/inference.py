"""
CivicLens - YOLO Inference
"""

from ultralytics import YOLO
import torch
import cv2
import os

# Fix for PyTorch 2.6 weights_only change
import torch.serialization
torch.serialization.add_safe_globals([])

MODEL_PATH = "best.pt"
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'. Drop best.pt in backend/")
        _model = YOLO(MODEL_PATH)
    return _model


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


def run_inference(image_path: str) -> dict | None:
    model = get_model()
    results = model(image_path, conf=0.25)

    best = None
    best_conf = 0.0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                best = {
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "severity": SEVERITY_MAP.get(cls_name, "Medium"),
                    "department": DEPARTMENT_MAP.get(cls_name, "Municipal Corporation"),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }

    if best:
        annotated = results[0].plot()
        annotated_path = image_path.replace(".", "_annotated.")
        cv2.imwrite(annotated_path, annotated)
        best["annotated_url"] = annotated_path

    return best