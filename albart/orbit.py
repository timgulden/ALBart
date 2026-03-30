"""Orbit — guided navigation through embedding space via track anchors.

The user picks anchor tracks that define a cyclic journey.  The DJ
alternates between two phases:

  DWELL  — play ~30 min of music near the current waypoint using normal
           512D neighbor hops (stays in genre).
  TRANSIT — move from current waypoint to the next in ~10 steps through
            raw 512D embedding space.  Each step covers 1/N of the remaining
            distance (N counts down from transit_steps).  This finds tracks
            that bridge genres via idiosyncratic audio similarities.

Usage (from the DJ):
    orbit.pick(current_emb, dj)  # returns (tid, hop_type)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

TRANSIT_STEPS = 10
DWELL_DURATION = 30.0 * 60.0  # 30 minutes in seconds


@dataclass
class OrbitAnchor:
    """A single waypoint in the orbit."""
    description: str
    track_id: str               # Spotify track ID of the anchor track
    embedding_512: np.ndarray   # 512D L2-normalized audio embedding
    position_5d: np.ndarray     # position in UMAP 5D space (for viewer)


class Orbit:
    """Two-phase orbit: dwell at waypoints, transit between them in 512D."""

    def __init__(
        self,
        anchors: list[OrbitAnchor],
        *,
        allow_same_artist: bool = False,
    ) -> None:
        if not anchors:
            raise ValueError("Orbit requires at least one anchor")
        self.anchors = anchors
        self.current_index = 0       # which anchor we're at or heading toward
        self.allow_same_artist = allow_same_artist

        # Phase state
        self.phase: str = "dwell"    # "dwell" or "transit"
        self._dwell_start: float = time.monotonic()
        self._transit_remaining: int = 0
        self._arrived: bool = False  # True after completing a transit to this anchor

        logger.info("Orbit created: %d anchors", len(anchors))
        for i, a in enumerate(anchors):
            logger.info("  [%d] %s  5d=%s", i, a.description,
                        np.array2string(a.position_5d, precision=2))

    @property
    def target(self) -> OrbitAnchor:
        """The anchor we're dwelling at (dwell) or heading toward (transit)."""
        return self.anchors[self.current_index]

    @property
    def next_anchor(self) -> OrbitAnchor:
        """The next anchor after the current one."""
        idx = (self.current_index + 1) % len(self.anchors)
        return self.anchors[idx]

    def start_dwell(self) -> None:
        """Enter dwell phase at the current anchor."""
        self.phase = "dwell"
        self._dwell_start = time.monotonic()
        self._arrived = True
        logger.info("Orbit DWELL at [%d]: %s",
                     self.current_index, self.target.description)

    def start_transit(self) -> None:
        """Enter transit phase toward the next anchor."""
        self.phase = "transit"
        self._transit_remaining = TRANSIT_STEPS
        self._arrived = False
        self.current_index = (self.current_index + 1) % len(self.anchors)
        logger.info(
            "Orbit TRANSIT → [%d]: %s  (%d steps)",
            self.current_index, self.target.description, TRANSIT_STEPS,
        )

    def dwell_elapsed(self) -> float:
        """Seconds spent in current dwell phase."""
        return time.monotonic() - self._dwell_start

    def should_leave_dwell(self) -> bool:
        """True if dwell time has expired."""
        return self.phase == "dwell" and self.dwell_elapsed() >= DWELL_DURATION

    def transit_step(self, current_emb: np.ndarray) -> np.ndarray:
        """Compute the 512D target for the next transit step.

        Moves 1/N of the remaining distance toward the target anchor,
        where N = transit_remaining.  Returns the target 512D point.
        """
        target_emb = self.target.embedding_512
        remaining = max(1, self._transit_remaining)
        fraction = 1.0 / remaining

        # Step toward target
        direction = target_emb.astype(np.float64) - current_emb.astype(np.float64)
        target_point = current_emb.astype(np.float64) + direction * fraction
        # Re-normalize (CLAP embeddings are L2-normalized)
        norm = np.linalg.norm(target_point)
        if norm > 1e-8:
            target_point /= norm

        self._transit_remaining -= 1
        dist = float(np.linalg.norm(direction))
        logger.info(
            "Orbit transit step %d/%d  frac=1/%d  dist_remaining=%.4f",
            TRANSIT_STEPS - self._transit_remaining, TRANSIT_STEPS,
            remaining, dist,
        )

        return target_point.astype(np.float32)

    def transit_done(self) -> bool:
        """True if all transit steps are used up."""
        return self._transit_remaining <= 0

    def get_progress(self) -> dict:
        """Return visualization state for the orbit viewer."""
        prev_idx = (self.current_index - 1) % len(self.anchors)

        if self.phase == "dwell":
            # Show the incoming segment as complete only if we actually
            # arrived here via transit (not on initial startup)
            progress = 1.0 if self._arrived else 0.0
        else:
            # During transit, progress = steps completed / total
            steps_done = TRANSIT_STEPS - self._transit_remaining
            progress = steps_done / TRANSIT_STEPS

        return {
            "phase": self.phase,
            "current_index": self.current_index,
            "prev_index": prev_idx,
            "segment_progress": progress,
            "dwell_elapsed": self.dwell_elapsed() if self.phase == "dwell" else 0,
            "dwell_duration": DWELL_DURATION,
            "transit_remaining": self._transit_remaining if self.phase == "transit" else 0,
            "transit_total": TRANSIT_STEPS,
        }


def _normalize_search(s: str) -> str:
    """Normalize for fuzzy matching."""
    import re
    s = s.lower()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r'[.\-_/\\]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _find_track(
    query: str,
    id_list: list[str],
    db: dict,
    used_indices: set[int],
    id_to_idx: dict[str, int],
) -> int | None:
    """Find a track by 'Artist — Title' or partial match.

    Returns the index into id_list, or None if not found.
    """
    q = _normalize_search(query)

    # Try splitting on common separators: " — ", " -- ", " - ", ": "
    parts = None
    for sep in [" \u2014 ", " -- ", " - ", ": "]:
        if sep in query:
            parts = query.split(sep, 1)
            break

    best_idx = None
    best_score = 0

    for tid in id_list:
        idx = id_to_idx.get(tid)
        if idx is None or idx in used_indices:
            continue
        row = db.get(tid)
        if not row:
            continue
        title = _normalize_search(row["title"] or "")
        artist = _normalize_search(row["artist"] or "")
        combined = f"{artist} {title}"

        score = 0
        if parts:
            q_a = _normalize_search(parts[0])
            q_t = _normalize_search(parts[1])
            if q_a in artist and q_t in title:
                score = 100  # exact: artist + title both match
            elif q_a in artist and title in q_t:
                score = 90   # artist matches, title is subset of query
            elif q_t in artist and q_a in title:
                score = 85   # swapped order
            elif q_a in title and q_t in artist:
                score = 85   # swapped order
            elif q_t in title or title in q_t:
                score = 50   # title match (either direction)
            elif q_a in artist or artist in q_a:
                score = 30   # artist match (either direction)
        # No separator — search both fields
        if score == 0 and (q in combined or combined in q):
            score = 20

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def build_orbit(
    descriptions: list[str],
    all_embeddings_512: np.ndarray,
    all_umap_5d: np.ndarray,
    id_list: list[str],
    db: dict,
    *,
    allow_same_artist: bool = False,
) -> Orbit:
    """Create an Orbit from track descriptions (typically 'Artist — Title').

    Matches each description against the library by metadata.  Uses the
    matched track's actual 5D UMAP position and 512D embedding as the anchor.
    This guarantees anchors land on real, well-separated points.
    """
    id_to_idx = {tid: i for i, tid in enumerate(id_list)}
    used_indices: set[int] = set()
    anchors = []

    for i, desc in enumerate(descriptions):
        idx = _find_track(desc, id_list, db, used_indices, id_to_idx)
        if idx is None:
            logger.warning("  Anchor %d: no match for '%s' — skipping", i, desc)
            continue

        used_indices.add(idx)
        tid = id_list[idx]
        row = db.get(tid, {})
        name = f"{row['title'] or '?'} — {row['artist'] or '?'}"

        # Warn if the match looks wrong (query artist doesn't appear in match)
        match_quality = "OK"
        # Try to extract artist from the description
        desc_artist = None
        for sep in [" \u2014 ", " -- ", " - ", ": "]:
            if sep in desc:
                desc_artist = desc.split(sep, 1)[0]
                break
        if desc_artist:
            q_artist = _normalize_search(desc_artist)
            match_artist = _normalize_search(row['artist'] or '')
            if q_artist not in match_artist and match_artist not in q_artist:
                match_quality = "WEAK (artist mismatch)"
                logger.warning(
                    "  Anchor %d: weak match — query '%s' matched '%s' "
                    "(artist '%s' not found in '%s')",
                    i, desc[:50], name, desc_artist, row['artist'],
                )

        anchors.append(OrbitAnchor(
            description=desc,
            track_id=tid,
            embedding_512=all_embeddings_512[idx].copy(),
            position_5d=all_umap_5d[idx].copy(),
        ))
        logger.info("  Anchor %d: '%s' → %s  [%s]  5d=%s",
                     i, desc[:50], name, match_quality,
                     np.array2string(all_umap_5d[idx], precision=2))

    if not anchors:
        raise ValueError("No anchor tracks matched in the library")

    return Orbit(anchors, allow_same_artist=allow_same_artist)
