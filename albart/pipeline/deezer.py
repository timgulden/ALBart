"""Deezer Search API lookup for 30-second preview URLs."""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.deezer.com/search"
TIMEOUT_SECONDS = 10
REQUEST_DELAY_SECONDS = 0.15
MAX_RETRIES = 4


def lookup_preview_url(title: str, artist: str) -> str | None:
    """
    Search Deezer for a matching track and return its 30-second preview URL,
    or None if not found. Retries with exponential backoff on rate limit errors.
    """
    primary_artist = artist.split(",")[0].strip()
    query = f"{primary_artist} {title}"

    params = {
        "q": query,
        "limit": 5,
    }

    delay = REQUEST_DELAY_SECONDS
    data = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(SEARCH_URL, params=params, timeout=TIMEOUT_SECONDS)
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                logger.debug("Rate limited by Deezer, waiting %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            time.sleep(delay)
            break
        except requests.HTTPError:
            logger.warning("Deezer lookup failed for '%s': %s", query, resp.status_code)
            time.sleep(delay)
            return None
        except Exception as e:
            logger.warning("Deezer lookup error for '%s': %s", query, e)
            time.sleep(delay)
            return None
    else:
        logger.warning("Deezer lookup gave up after %d attempts for '%s'", MAX_RETRIES, query)
        return None

    if data is None:
        return None

    results = data.get("data", [])
    if not results:
        logger.debug("No Deezer results for '%s'", query)
        return None

    # Prefer artist name match, then fall back to first result with a preview
    for result in results:
        artist_name = result.get("artist", {}).get("name", "").lower()
        if primary_artist.lower() in artist_name or artist_name in primary_artist.lower():
            url = result.get("preview")
            if url:
                return url

    for result in results:
        url = result.get("preview")
        if url:
            return url

    return None
