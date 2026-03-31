"""Pure orbit state machine.

Every function takes time as a parameter (no ``time.monotonic()`` calls).
Returns new ``OrbitState`` copies — never mutates.

Two phases cycle through anchor tracks:
  DWELL  — ~30 min near the anchor in 5D UMAP space.
  TRANSIT — ~10 steps through 512D space toward the next anchor.
"""

from __future__ import annotations

import re

from albart.core.state import OrbitAnchorState, OrbitPhase, OrbitState


# Defaults (previously hard-coded in orbit.py)
DWELL_DURATION = 30.0 * 60.0    # seconds
TRANSIT_STEPS_MAX = 10
TRANSIT_STEPS_PER_UNIT = 3       # steps per unit of 25D distance
ARRIVAL_THRESHOLD = 0.15         # fraction of initial distance


def compute_transit_steps(distance: float) -> int:
    """Determine how many transit steps based on 25D distance.

    Short distances (same cluster) get 1 step; large genre crossings
    get up to TRANSIT_STEPS_MAX.
    """
    return max(1, min(TRANSIT_STEPS_MAX, round(distance * TRANSIT_STEPS_PER_UNIT)))


# ---------------------------------------------------------------------------
# Phase queries
# ---------------------------------------------------------------------------

def dwell_elapsed(orbit: OrbitState, current_time: float) -> float:
    """Seconds spent in the current dwell phase."""
    return current_time - orbit.dwell_start_mono


def should_leave_dwell(orbit: OrbitState, current_time: float) -> bool:
    """True if dwell time has expired."""
    return (
        orbit.phase == OrbitPhase.DWELL
        and dwell_elapsed(orbit, current_time) >= DWELL_DURATION
    )


def transit_done(orbit: OrbitState) -> bool:
    """True if all transit steps have been used."""
    return orbit.transit_remaining <= 0


def check_arrival(orbit: OrbitState, distance_to_target: float) -> bool:
    """True if within the arrival threshold of the target anchor."""
    if orbit.phase != OrbitPhase.TRANSIT:
        return False
    threshold = orbit.transit_initial_dist * ARRIVAL_THRESHOLD
    return distance_to_target <= threshold


# ---------------------------------------------------------------------------
# Phase transitions (return new OrbitState)
# ---------------------------------------------------------------------------

def start_dwell(orbit: OrbitState, current_time: float) -> OrbitState:
    """Enter dwell phase at the current anchor.

    Marks the incoming segment as completed.
    """
    prev_idx = (orbit.current_index - 1) % len(orbit.anchors)
    return orbit.model_copy(update={
        "phase": OrbitPhase.DWELL,
        "dwell_start_mono": current_time,
        "arrived": True,
        "completed_segments": orbit.completed_segments | {prev_idx},
    })


def start_transit(
    orbit: OrbitState,
    initial_distance: float,
    start_tid: str | None = None,
) -> OrbitState:
    """Enter transit phase toward the next anchor.

    Args:
        initial_distance: L2 distance from the current position to the
            next anchor in 512D space.  Used for early-arrival detection.
        start_tid: track playing when transit begins (fixed origin for
            linear stepping).
    """
    next_index = (orbit.current_index + 1) % len(orbit.anchors)
    steps = compute_transit_steps(initial_distance)
    return orbit.model_copy(update={
        "phase": OrbitPhase.TRANSIT,
        "current_index": next_index,
        "transit_remaining": steps,
        "transit_total": steps,
        "arrived": False,
        "transit_initial_dist": initial_distance,
        "transit_start_tid": start_tid,
    })


def advance_transit_step(orbit: OrbitState) -> tuple[OrbitState, float]:
    """Decrement transit steps and return the linear interpolation fraction.

    Uses linear stepping from a fixed origin: step N targets
    ``N / TRANSIT_STEPS`` of the way from transit start to anchor.
    This avoids the Zeno drift problem where picked tracks pull the
    trajectory off-axis.

    Returns:
        ``(new_orbit, fraction)`` — fraction is the proportion of the
        full start→anchor distance to target (0.0 to 1.0).
    """
    total = max(1, orbit.transit_total)
    steps_used = total - orbit.transit_remaining + 1
    fraction = steps_used / total
    new_orbit = orbit.model_copy(update={
        "transit_remaining": orbit.transit_remaining - 1,
    })
    return new_orbit, fraction


def record_played(orbit: OrbitState, track_id: str) -> OrbitState:
    """Record that a track was played (for progress tracking)."""
    return orbit.model_copy(update={"last_played_tid": track_id})


# ---------------------------------------------------------------------------
# Progress (for UI visualization)
# ---------------------------------------------------------------------------

def get_progress(
    orbit: OrbitState,
    current_time: float,
    segment_progress: float = 0.0,
) -> dict:
    """Return visualization state for the orbit viewer.

    Args:
        segment_progress: 0.0–1.0 ratio of transit progress.  Computed
            by the engine from 512D distances (requires embeddings).
    """
    prev_idx = (orbit.current_index - 1) % len(orbit.anchors)

    if orbit.phase == OrbitPhase.DWELL:
        progress = 1.0 if orbit.arrived else 0.0
    else:
        progress = segment_progress

    return {
        "phase": orbit.phase.value,
        "current_index": orbit.current_index,
        "prev_index": prev_idx,
        "segment_progress": progress,
        "completed_segments": sorted(orbit.completed_segments),
        "dwell_elapsed": dwell_elapsed(orbit, current_time) if orbit.phase == OrbitPhase.DWELL else 0,
        "dwell_duration": DWELL_DURATION,
        "transit_remaining": orbit.transit_remaining if orbit.phase == OrbitPhase.TRANSIT else 0,
        "transit_total": orbit.transit_total,
    }


# ---------------------------------------------------------------------------
# Track matching (pure string logic, used by build_orbit)
# ---------------------------------------------------------------------------

def normalize_search(s: str) -> str:
    """Normalize for fuzzy matching."""
    s = s.lower()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r'[.\-_/\\]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def find_track(
    query: str,
    tracks: list[dict],
    used_ids: set[str],
) -> str | None:
    """Find a track by 'Artist — Title' or partial match.

    Args:
        query: e.g. ``"Massive Attack — Teardrop"``
        tracks: list of dicts with ``track_id``, ``title``, ``artist``.
        used_ids: already-assigned track IDs to skip.

    Returns:
        Matched ``track_id``, or None.
    """
    q = normalize_search(query)

    # Try splitting on common separators
    parts = None
    for sep in [" \u2014 ", " -- ", " - ", ": "]:
        if sep in query:
            parts = query.split(sep, 1)
            break

    best_tid = None
    best_score = 0

    for row in tracks:
        tid = row["track_id"]
        if tid in used_ids:
            continue
        title = normalize_search(row.get("title") or "")
        artist = normalize_search(row.get("artist") or "")
        combined = f"{artist} {title}"

        score = 0
        if parts:
            q_a = normalize_search(parts[0])
            q_t = normalize_search(parts[1])
            if q_a in artist and q_t in title:
                score = 100
            elif q_a in artist and title in q_t:
                score = 90
            elif q_t in artist and q_a in title:
                score = 85
            elif q_a in title and q_t in artist:
                score = 85
            elif q_t in title or title in q_t:
                score = 50
            elif q_a in artist or artist in q_a:
                score = 30

        if score == 0 and (q in combined or combined in q):
            score = 20

        if score > best_score:
            best_score = score
            best_tid = tid

    return best_tid
