"""Spotify API client: authenticate and pull top tracks."""

import logging
import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth

logger = logging.getLogger(__name__)

SCOPE = "user-top-read"
PAGE_SIZE = 50


def get_client() -> spotipy.Spotify:
    """Return an authenticated Spotify client using env var credentials."""
    auth = SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ.get(
            "SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback"
        ),
        scope=SCOPE,
    )
    return spotipy.Spotify(auth_manager=auth)


def fetch_top_tracks(sp: spotipy.Spotify) -> list[dict]:
    """
    Fetch all available top tracks for long_term time range.
    Returns a list of normalized track dicts.
    """
    tracks = []
    offset = 0

    while True:
        logger.debug("Fetching top tracks: offset=%d", offset)
        result = sp.current_user_top_tracks(
            limit=PAGE_SIZE,
            offset=offset,
            time_range="long_term",
        )
        items = result.get("items", [])
        if not items:
            break

        for item in items:
            track = _normalize_track(item)
            if track is not None:
                tracks.append(track)

        offset += len(items)
        if result.get("next") is None:
            break

    logger.info("Fetched %d top tracks from Spotify", len(tracks))
    return tracks


def _normalize_track(item: dict) -> dict | None:
    """Extract relevant fields from a Spotify track object."""
    try:
        track_id = item["id"]
        title = item["name"]
        artist = ", ".join(a["name"] for a in item["artists"])
        album = item["album"]["name"]
        preview_url = item.get("preview_url")  # may be None

        # Pick the art URL closest to 300px wide
        images = item["album"].get("images", [])
        art_url = _pick_art_url(images, target=300)
        if not art_url:
            logger.warning("No album art for track %s (%s)", track_id, title)
            return None

        return {
            "track_id": track_id,
            "title": title,
            "artist": artist,
            "album": album,
            "preview_url": preview_url,
            "preview_path": None,
            "art_url": art_url,
            "art_path_original": None,
            "art_path_32": None,
            "embedding_status": "no_preview" if preview_url is None else "pending",
        }
    except (KeyError, TypeError) as e:
        logger.error("Failed to normalize track: %s", e)
        return None


def _pick_art_url(images: list[dict], target: int) -> str | None:
    """Return the image URL whose width is closest to target pixels."""
    if not images:
        return None
    return min(images, key=lambda img: abs(img.get("width", 0) - target))["url"]
