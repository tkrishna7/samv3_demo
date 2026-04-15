"""
Download SAM2 model weights from Meta's servers.
Run this once before starting the app:
    uv run python download_weights.py
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

WEIGHTS_DIR = Path("weights")

MODELS = {
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "filename": "sam2_hiera_tiny.pt",
        "size_mb": 38,
    },
    "small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "filename": "sam2_hiera_small.pt",
        "size_mb": 46,
    },
    "base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "filename": "sam2_hiera_base_plus.pt",
        "size_mb": 80,
    },
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "filename": "sam2_hiera_large.pt",
        "size_mb": 224,
    },
}


def download_file(url: str, dest: Path) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)

    # Default: download the small model (good speed/accuracy balance)
    model_key = "small"
    if len(sys.argv) > 1 and sys.argv[1] in MODELS:
        model_key = sys.argv[1]

    model = MODELS[model_key]
    dest = WEIGHTS_DIR / model["filename"]

    if dest.exists():
        print(f"[✓] {dest.name} already exists, skipping download.")
        return

    print(f"Downloading SAM2 {model_key} model (~{model['size_mb']}MB)...")
    print(f"  Source : {model['url']}")
    print(f"  Dest   : {dest}")
    download_file(model["url"], dest)
    print(f"\n[✓] Saved to {dest}")
    print("\nAvailable models: tiny, small, base_plus, large")
    print("Usage: uv run python download_weights.py [tiny|small|base_plus|large]")


if __name__ == "__main__":
    main()
