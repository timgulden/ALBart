"""Unified weighted track selection with artist penalty.

This is the ONE place candidate selection logic lives.  Previously
duplicated across ``_pick_normal_hop``, ``_pick_25d_neighbor``,
``_find_nearest_512``, and ``_pick_long_hop`` in the old DJ class.

All functions are pure — no I/O, no side effects.
"""

from __future__ import annotations

import numpy as np

# Maximum exponent for inverse-distance weighting.
# At temperature=0 → exponent=MAX_EXPONENT (very peaked).
# At temperature=1 → exponent=0 (uniform).
MAX_EXPONENT = 6.0


def select_from_candidates(
    candidates: list[tuple[str, float]],
    *,
    recent_artists: tuple[str, ...],
    artist_penalty: float,
    allow_same_artist: bool,
    temperature: float,
    track_artist_map: dict[str, str],
    rng: np.random.Generator,
) -> str | None:
    """Pick one track from distance-sorted candidates.

    Args:
        candidates: ``[(track_id, distance), ...]`` sorted by distance.
        recent_artists: lowercased artist names from last ~3 tracks.
        artist_penalty: weight multiplier for same-artist candidates.
        allow_same_artist: if True, skip artist penalty entirely.
        temperature: 0.0 (always nearest) to 1.0 (uniform random).
        track_artist_map: ``{track_id: lowercase_artist}`` for penalty lookup.
        rng: numpy random generator for weighted choice.

    Returns:
        Selected track_id, or None if candidates is empty.
    """
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][0]

    tids = [c[0] for c in candidates]
    dists = np.array([c[1] for c in candidates], dtype=np.float64)

    # Inverse-distance weighting with temperature-controlled exponent.
    # temp=0 → exponent=6 (strongly peaked), temp=1 → exponent=0 (uniform).
    exponent = MAX_EXPONENT * (1.0 - max(0.0, min(1.0, temperature)))

    if exponent < 1e-10:
        weights = np.ones_like(dists)
    else:
        weights = 1.0 / np.maximum(dists, 1e-8) ** exponent

    # Artist penalty
    if not allow_same_artist:
        recent_set = set(recent_artists)
        for i, tid in enumerate(tids):
            artist = track_artist_map.get(tid, "")
            if artist in recent_set:
                weights[i] *= artist_penalty

    total = weights.sum()
    if total <= 0:
        return candidates[0][0]
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
