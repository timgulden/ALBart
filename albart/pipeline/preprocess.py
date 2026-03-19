"""Downsample album art to 32x32 PNG."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from albart.utils import DATA_DIR

logger = logging.getLogger(__name__)

ART_32_DIR = DATA_DIR / "art_32"
TARGET_SIZE = (32, 32)


def ensure_dirs() -> None:
    ART_32_DIR.mkdir(parents=True, exist_ok=True)


def downsample_art(track_id: str, source_path: Path) -> Path | None:
    """
    Downsample the source image to 32x32 using Lanczos resampling.
    source_path may be absolute or relative to DATA_DIR.
    Returns relative path from data/ or None on failure.
    """
    if not source_path.is_absolute():
        source_path = DATA_DIR / source_path

    dest = ART_32_DIR / f"{track_id}.png"
    if dest.exists():
        logger.debug("32x32 art already exists: %s", dest.name)
        return dest.relative_to(DATA_DIR)

    try:
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(dest, format="PNG")
        logger.debug("Downsampled art: %s", dest.name)
        return dest.relative_to(DATA_DIR)
    except Exception as e:
        logger.error("Failed to downsample art for %s: %s", track_id, e)
        return None


def load_art_32(art_path_32: str | Path) -> "np.ndarray":
    """Load a 32x32 PNG and return a (32, 32, 3) uint8 numpy array."""
    import numpy as np

    path = DATA_DIR / art_path_32 if not Path(art_path_32).is_absolute() else Path(art_path_32)
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)
