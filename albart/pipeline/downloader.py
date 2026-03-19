"""Download preview MP3s and album art images."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from albart.utils import DATA_DIR

logger = logging.getLogger(__name__)

PREVIEWS_DIR = DATA_DIR / "previews"
ART_ORIGINAL_DIR = DATA_DIR / "art_original"

TIMEOUT_SECONDS = 30


def ensure_dirs() -> None:
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    ART_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)


def download_preview(track_id: str, url: str) -> Path | None:
    """
    Download audio preview for track_id. Extension is inferred from the URL
    (.mp3 for Spotify, .m4a for iTunes). Returns relative path from data/
    or None on failure.
    """
    ext = ".m4a" if "apple.com" in url or ".m4a" in url else ".mp3"
    dest = PREVIEWS_DIR / f"{track_id}{ext}"
    if dest.exists():
        logger.debug("Preview already exists: %s", dest.name)
        return dest.relative_to(DATA_DIR)

    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.debug("Downloaded preview: %s", dest.name)
        return dest.relative_to(DATA_DIR)
    except requests.RequestException as e:
        logger.error("Failed to download preview for %s: %s", track_id, e)
        return None


def download_art(track_id: str, url: str) -> Path | None:
    """
    Download original album art for track_id. Returns relative path from data/
    or None on failure.
    """
    dest = ART_ORIGINAL_DIR / f"{track_id}.jpg"
    if dest.exists():
        logger.debug("Art already exists: %s", dest.name)
        return dest.relative_to(DATA_DIR)

    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.debug("Downloaded art: %s", dest.name)
        return dest.relative_to(DATA_DIR)
    except requests.RequestException as e:
        logger.error("Failed to download art for %s: %s", track_id, e)
        return None
