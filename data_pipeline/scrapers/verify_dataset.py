"""
CivicLens - Dataset Verifier
Run after scraping to check for:
  - Corrupted/unreadable images
  - Class imbalance
  - Duplicate images (via hash)
  - Minimum size violations
Prints a full health report.
"""

import os
import hashlib
from pathlib import Path
from PIL import Image
from collections import defaultdict

RAW_DIR   = "../dataset/images/raw"
MIN_WIDTH  = 200
MIN_HEIGHT = 200
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def file_hash(path: str) -> str:
    """MD5 hash of file content for duplicate detection."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def verify_class(class_dir: str, class_name: str):
    issues = []
    hashes = {}
    counts = {"ok": 0, "corrupt": 0, "too_small": 0, "duplicate": 0}

    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        if Path(fname).suffix.lower() not in VALID_EXTS:
            continue

        # Check for duplicates
        h = file_hash(fpath)
        if h in hashes:
            issues.append(f"  DUPLICATE: {fname} == {hashes[h]}")
            os.remove(fpath)        # auto-remove duplicate
            counts["duplicate"] += 1
            continue
        hashes[h] = fname

        # Check readability + size
        try:
            img = Image.open(fpath)
            w, h = img.size
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                issues.append(f"  TOO SMALL: {fname} ({w}x{h})")
                os.remove(fpath)
                counts["too_small"] += 1
            else:
                counts["ok"] += 1
        except Exception as e:
            issues.append(f"  CORRUPT: {fname} — {e}")
            os.remove(fpath)
            counts["corrupt"] += 1

    return counts, issues


def main():
    print("=" * 60)
    print("  CivicLens Dataset Health Check")
    print("=" * 60)

    if not os.path.isdir(RAW_DIR):
        print("❌ No raw dataset folder. Run scrape.py first.")
        return

    classes = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ])

    all_counts = defaultdict(int)
    class_totals = {}

    for cls in classes:
        cls_dir = os.path.join(RAW_DIR, cls)
        print(f"\n📦 {cls}")
        counts, issues = verify_class(cls_dir, cls)

        print(f"   ✅ OK: {counts['ok']}   "
              f"🔁 Dupes removed: {counts['duplicate']}   "
              f"📏 Too small: {counts['too_small']}   "
              f"❌ Corrupt: {counts['corrupt']}")

        if issues:
            for i in issues[:5]:   # show max 5 issues per class
                print(f"   {i}")
            if len(issues) > 5:
                print(f"   ... and {len(issues)-5} more")

        class_totals[cls] = counts["ok"]
        for k, v in counts.items():
            all_counts[k] += v

    # Class balance warning
    print("\n" + "=" * 60)
    print("  Class Balance")
    print("=" * 60)
    if class_totals:
        max_count = max(class_totals.values())
        min_count = min(class_totals.values())
        for cls, cnt in sorted(class_totals.items(), key=lambda x: -x[1]):
            bar = "█" * (cnt * 30 // max(max_count, 1))
            print(f"  {cls:<30} {bar} {cnt}")
        if max_count > 0 and min_count / max_count < 0.5:
            print(f"\n  ⚠️  WARNING: Class imbalance detected!")
            print(f"     Scrape more images for underrepresented classes.")

    print("\n" + "=" * 60)
    print(f"  TOTAL OK:          {all_counts['ok']}")
    print(f"  Duplicates removed:{all_counts['duplicate']}")
    print(f"  Too small removed: {all_counts['too_small']}")
    print(f"  Corrupt removed:   {all_counts['corrupt']}")
    print("=" * 60)
    print("\n  ✅ Verification complete. Run split_dataset.py next.")


if __name__ == "__main__":
    main()