"""Unified weighted track selection with artist penalty.

This is the ONE place candidate selection logic lives.  Previously
duplicated across ``_pick_normal_hop``, ``_pick_5d_neighbor``,
``_find_nearest_512``, and ``_pick_long_hop`` in the old DJ class.

All functions are pure — no I/O, no side effects.
"""

from __future__ import annotations

import numpy as np


def select_from_candidates(
    candidates: list[tuple[str, float]],
    *,
    recent_artists: tuple[str, ...],
    artist_penalty: float,
    allow_same_artist: bool,
    song_k: int,
    track_artist_map: dict[str, str],
    rng: np.random.Generator,
) -> str | None:
    """Pick one track from distance-sorted candidates.

    Args:
        candidates: ``[(track_id, distance), ...]`` sorted by distance.
        recent_artists: lowercased artist names from last ~3 tracks.
        artist_penalty: weight multiplier for same-artist candidates.
            0.01 for orbit (strong penalty), 0.1 for normal play.
        allow_same_artist: if True, skip artist penalty entirely.
        song_k: max candidates to consider (controls exploration width).
        track_artist_map: ``{track_id: lowercase_artist}`` for penalty lookup.
        rng: numpy random generator for weighted choice.

    Returns:
        Selected track_id, or None if candidates is empty.
    """
    if not candidates:
        return None

    # Trim to song_k candidates
    pool = candidates[:song_k]

    # Deterministic when K=1
    if song_k <= 1 or len(pool) == 1:
        return pool[0][0]

    tids = [c[0] for c in pool]
    dists = np.array([c[1] for c in pool], dtype=np.float64)

    # Inverse-cube-distance weighting — strongly favors nearest tracks
    # while keeping the full pool for occasional variety
    weights = 1.0 / np.maximum(dists, 1e-8) ** 3

    # Artist penalty
    if not allow_same_artist:
        recent_set = set(recent_artists)
        for i, tid in enumerate(tids):
            artist = track_artist_map.get(tid, "")
            if artist in recent_set:
                weights[i] *= artist_penalty

    total = weights.sum()
    if total <= 0:
        return pool[0][0]
    weights /= total

    chosen = rng.choice(len(tids), p=weights)
    return tids[chosen]


def get_recent_artists(
    history: tuple[str, ...],
    track_artist_map: dict[str, str],
    n: int = 3,
) -> tuple[str, ...]:
    """Extract lowercased artist names from the last *n* played tracks."""
    artists: list[str] = []
    for tid in history[-n:]:
        artist = track_artist_map.get(tid, "")
        if artist:
            artists.append(artist)
    return tuple(artists)
