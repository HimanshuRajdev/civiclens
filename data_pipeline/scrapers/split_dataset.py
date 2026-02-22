"""
CivicLens - Dataset Splitter
Splits raw scraped images into train/val sets (80/20).
Run this AFTER scraping and BEFORE labeling.
"""

import os
import shutil
import random
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

RAW_DIR    = "../dataset/images/raw"
TRAIN_DIR  = "../dataset/images/train"
VAL_DIR    = "../dataset/images/val"
SPLIT_RATIO = 0.8   # 80% train, 20% val

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ─── SPLIT ────────────────────────────────────────────────────────────────────

def split_class(class_name: str):
    src = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(src):
        print(f"  ⚠️  No folder found for '{class_name}', skipping.")
        return 0, 0

    images = [
        f for f in os.listdir(src)
        if Path(f).suffix.lower() in VALID_EXTS
    ]
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR,   class_name), exist_ok=True)

    for fname in train_imgs:
        shutil.copy2(
            os.path.join(src, fname),
            os.path.join(TRAIN_DIR, class_name, fname)
        )
    for fname in val_imgs:
        shutil.copy2(
            os.path.join(src, fname),
            os.path.join(VAL_DIR, class_name, fname)
        )

    return len(train_imgs), len(val_imgs)


def main():
    random.seed(42)

    raw_path = Path(RAW_DIR)
    if not raw_path.exists():
        print("❌ Raw dataset folder not found. Run scrape.py first.")
        return

    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    if not classes:
        print("❌ No class folders found in raw/. Run scrape.py first.")
        return

    print("=" * 55)
    print("  CivicLens Dataset Splitter (80/20 Train/Val)")
    print("=" * 55)

    total_train = total_val = 0
    for cls in sorted(classes):
        t, v = split_class(cls)
        print(f"  {cls:<30} train={t:>3}  val={v:>3}")
        total_train += t
        total_val   += v

    print("=" * 55)
    print(f"  {'TOTAL':<30} train={total_train:>3}  val={total_val:>3}")
    print(f"\n  ✅ Done! Images split into:")
    print(f"     {os.path.abspath(TRAIN_DIR)}")
    print(f"     {os.path.abspath(VAL_DIR)}")
    print(f"\n  Next step: open LabelImg and start annotating!")
    print(f"  See: ../docs/LABELING_GUIDE.md")


if __name__ == "__main__":
    main()