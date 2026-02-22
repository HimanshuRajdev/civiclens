"""
CivicLens - Dataset Merger (fixed)
Looks for downloaded folders in dataset/downloads/
"""

import os
import shutil
import yaml
from pathlib import Path

DOWNLOADS_DIR = "../dataset/downloads"
OUTPUT_DIR    = "../dataset/final"

CLASSES = [
    "pothole",
    "garbage_overflow",
    "broken_streetlight",
    "water_leakage",
    "broken_sidewalk",
    "sinkhole",
]

def merge():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    print("=" * 55)
    print("  CivicLens — Dataset Merger")
    print("=" * 55)

    counts = {"train": 0, "val": 0}

    for class_name in os.listdir(DOWNLOADS_DIR):
        class_path = os.path.join(DOWNLOADS_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"\n📂 Merging: {class_name}")

        for split in ["train", "valid", "val", "test"]:
            split_path = os.path.join(class_path, split)
            if not os.path.isdir(split_path):
                continue

            out_split = "val" if split in ["valid", "val", "test"] else "train"
            img_dir = os.path.join(split_path, "images")
            lbl_dir = os.path.join(split_path, "labels")

            if not os.path.isdir(img_dir):
                continue

            copied = 0
            for fname in os.listdir(img_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                stem = Path(fname).stem
                lbl_file = os.path.join(lbl_dir, f"{stem}.txt")
                if not os.path.exists(lbl_file):
                    continue

                shutil.copy2(
                    os.path.join(img_dir, fname),
                    os.path.join(OUTPUT_DIR, out_split, "images", f"{class_name}_{fname}")
                )
                shutil.copy2(
                    lbl_file,
                    os.path.join(OUTPUT_DIR, out_split, "labels", f"{class_name}_{stem}.txt")
                )
                copied += 1

            counts[out_split] += copied
            print(f"  ✅ {out_split}: +{copied} images")

    yaml_content = {
        "path": str(Path(OUTPUT_DIR).resolve()),
        "train": "train/images",
        "val":   "val/images",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("\n" + "=" * 55)
    print(f"  Train images: {counts['train']}")
    print(f"  Val images:   {counts['val']}")
    print(f"  dataset.yaml written ✅")
    print("=" * 55)
    print("\n  ✅ Dataset ready! Next: python train.py")

if __name__ == "__main__":
    merge()