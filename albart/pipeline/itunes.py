"""iTunes Search API lookup for 30-second preview URLs."""

from __future__ import annotations

import logging
import time
import urllib.parse

import requests

logger = logging.getLogger(__name__)

SEARCH_URL = "https://itunes.apple.com/search"
TIMEOUT_SECONDS = 10
REQUEST_DELAY_SECONDS = 0.2   # baseline delay between requests
MAX_RETRIES = 4


def lookup_preview_url(title: str, artist: str) -> str | None:
    """
    Search iTunes for a matching track and return its 30-second preview URL,
    or None if not found. Retries with exponential backoff on rate limit errors.
    """
    # Use the first listed artist if there are multiple
    primary_artist = artist.split(",")[0].strip()
    query = f"{primary_artist} {title}"

    params = {
        "term": query,
        "media": "music",
        "entity": "song",
        "limit": 5,
    }

    delay = REQUEST_DELAY_SECONDS
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(SEARCH_URL, params=params, timeout=TIMEOUT_SECONDS)
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                logger.debug("Rate limited by iTunes, waiting %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            time.sleep(delay)
            break
        except requests.HTTPError:
            logger.warning("iTunes lookup failed for '%s': %s", query, resp.status_code)
            time.sleep(delay)
            return None
        except Exception as e:
            logger.warning("iTunes lookup failed for '%s': %s", query, e)
            time.sleep(delay)
            return None
    else:
        logger.warning("iTunes lookup gave up after %d attempts for '%s'", MAX_RETRIES, query)
        return None

    results = data.get("results", [])
    if not results:
        logger.debug("No iTunes results for '%s'", query)
        return None

    # Pick the best match: prefer exact artist name match, then take first result
    for result in results:
        artist_name = result.get("artistName", "").lower()
        if primary_artist.lower() in artist_name or artist_name in primary_artist.lower():
            url = result.get("previewUrl")
            if url:
                return url

    # Fall back to first result with a preview URL
    for result in results:
        url = result.get("previewUrl")
        if url:
            return url

    return None
