"""
CivicLens - Roboflow Universe Dataset Downloader
Downloads pre-labeled datasets for all 6 classes.
"""

from roboflow import Roboflow
import os

API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=API_KEY)

DATASETS = [
    ("civiclens-r76kq",        "pothole-vhmow-tyhuu",          1, "pothole"),
    ("sewage-3rtqt",           "water-leakage-agrsb",          1, "water_leakage"),
    ("disabled-obstacles",     "broken-indo-sidewalk-4llpq",   2, "broken_sidewalk"),
    ("mariswary-deepak-4ajr0", "garbage-can-overflow",         4, "garbage_overflow"),
    ("sinkholedatasetfyp",     "sinkhole-detection-xwwkt",     3, "sinkhole"),
    ("civiclens-r76kq",        "damaged-traffic-sign-85gfu",   1, "broken_streetlight"),
]

print("=" * 55)
print("  CivicLens — Downloading All Datasets")
print("=" * 55)

for workspace, project, version, class_name in DATASETS:
    print(f"\n📦 Downloading: {class_name}")
    try:
        dataset = rf.workspace(workspace).project(project).version(version).download(
            "yolov8",
            location=f"../dataset/downloads/{class_name}"
        )
        print(f"  ✅ Done")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

print("\n✅ All downloads done!")
print("   Next: python merge_datasets.py")