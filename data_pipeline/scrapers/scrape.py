"""
CivicLens - Image Scraper
Uses icrawler to scrape images for each civic issue class.
Multiple search queries per class to maximize diversity.
"""

import os
import time
import shutil
import random
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATASET_DIR = "../dataset/images/raw"
IMAGES_PER_QUERY = 80       # images per search query
MAX_PER_CLASS = 400         # hard cap per class

# ─── SEARCH QUERIES PER CLASS ─────────────────────────────────────────────────
# Multiple queries per class = more diversity = better model generalization

CLASSES = {
    "pothole": [
        "pothole road damage",
        "road pothole street",
        "large pothole asphalt",
        "pothole aerial view",
        "damaged road surface crack",
        "road crater damage",
    ],
    "garbage_overflow": [
        "overflowing garbage bin street",
        "trash overflow public bin",
        "garbage dump sidewalk",
        "waste overflow urban",
        "litter street garbage pile",
        "public dustbin overflowing",
    ],
    "broken_streetlight": [
        "broken streetlight pole",
        "damaged street lamp",
        "broken traffic light pole",
        "fallen street light",
        "dark broken lamp post",
        "streetlight vandalized",
    ],
    "water_leakage": [
        "water pipe leakage street",
        "burst water pipe road",
        "road flooding water leak",
        "water main break street",
        "pipe leak puddle urban",
        "water gushing road pipe burst",
    ],
    "broken_sidewalk": [
        "broken sidewalk pavement",
        "cracked footpath urban",
        "damaged pavement tiles",
        "uneven broken walkway",
        "crumbling sidewalk concrete",
        "raised cracked pavement",
    ],
    "sinkhole": [
        "urban sinkhole road",
        "sinkhole street collapse",
        "road collapse sinkhole",
        "ground collapse urban",
        "sinkhole pavement city",
        "road cave in sinkhole",
    ],
}

# ─── SCRAPER ──────────────────────────────────────────────────────────────────

def scrape_class(class_name: str, queries: list[str]):
    class_dir = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    total = 0
    for i, query in enumerate(queries):
        if total >= MAX_PER_CLASS:
            print(f"  ✅ Reached max ({MAX_PER_CLASS}) for '{class_name}', stopping.")
            break

        remaining = MAX_PER_CLASS - total
        num = min(IMAGES_PER_QUERY, remaining)

        print(f"  🔍 Query {i+1}/{len(queries)}: '{query}' → fetching {num} images")

        # Temp folder per query to avoid naming conflicts
        temp_dir = os.path.join(class_dir, f"_temp_{i}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": temp_dir},
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4,
            )
            crawler.crawl(
                keyword=query,
                max_num=num,
                min_size=(200, 200),       # filter tiny/irrelevant images
                file_idx_offset=total,
            )
        except Exception as e:
            print(f"  ⚠️  Bing failed for '{query}': {e}. Trying Google...")
            try:
                crawler = GoogleImageCrawler(
                    storage={"root_dir": temp_dir},
                    feeder_threads=1,
                    parser_threads=1,
                    downloader_threads=2,
                )
                crawler.crawl(keyword=query, max_num=num, file_idx_offset=total)
            except Exception as e2:
                print(f"  ❌ Google also failed: {e2}. Skipping query.")
                continue

        # Move files from temp → class root with proper naming
        scraped = 0
        for fname in os.listdir(temp_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                ext = os.path.splitext(fname)[1]
                new_name = f"{class_name}_{total + scraped:04d}{ext}"
                shutil.move(
                    os.path.join(temp_dir, fname),
                    os.path.join(class_dir, new_name),
                )
                scraped += 1
        shutil.rmtree(temp_dir, ignore_errors=True)

        total += scraped
        print(f"     ✔ Got {scraped} images (total: {total})")
        time.sleep(random.uniform(1.5, 3.0))   # polite delay between queries

    print(f"  🎉 '{class_name}' done — {total} images total\n")
    return total


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("=" * 55)
    print("  CivicLens Image Scraper")
    print("=" * 55)

    summary = {}
    for class_name, queries in CLASSES.items():
        print(f"\n📦 Scraping class: '{class_name}'")
        count = scrape_class(class_name, queries)
        summary[class_name] = count

    print("\n" + "=" * 55)
    print("  SCRAPING COMPLETE — Summary")
    print("=" * 55)
    total_all = 0
    for cls, cnt in summary.items():
        print(f"  {cls:<30} {cnt:>4} images")
        total_all += cnt
    print(f"  {'TOTAL':<30} {total_all:>4} images")
    print(f"\n  Images saved to: {os.path.abspath(DATASET_DIR)}")
    print("  Next step: run  python split_dataset.py")


if __name__ == "__main__":
    main()