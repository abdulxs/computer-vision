#!/usr/bin/env python3
"""
Simple computer vision utilities
"""

from pathlib import Path
from typing import List, Tuple
import json


def list_images(directory: str = ".") -> List[Path]:
    """List all images in a directory"""
    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    dir_path = Path(directory)
    return [f for f in dir_path.iterdir() if f.suffix.lower() in image_exts]


def get_image_info(image_path: str) -> dict:
    """Get basic image info"""
    path = Path(image_path)
    if not path.exists():
        return {"error": "File not found"}
    
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "extension": path.suffix.lower()
    }


def batch_info(directory: str = ".") -> List[dict]:
    """Get info for all images in directory"""
    images = list_images(directory)
    return [get_image_info(str(img)) for img in images]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(batch_info(sys.argv[1]), indent=2))
    else:
        print(json.dumps(batch_info("."), indent=2))
