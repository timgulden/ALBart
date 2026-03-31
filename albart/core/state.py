"""Immutable state models for the DJ engine.

All models are frozen Pydantic types.  Logic functions receive state and
return new copies via ``model_copy(update={...})``.  No mutation, no I/O.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Track reference (lightweight — no embedding data)
# ---------------------------------------------------------------------------

class TrackRef(BaseModel, frozen=True):
    """Minimal track metadata returned by database queries."""
    track_id: str
    title: str
    artist: str
    album: str


# ---------------------------------------------------------------------------
# Playback snapshot (from Spotify polling)
# ---------------------------------------------------------------------------

class PlaybackSnapshot(BaseModel, frozen=True):
    """Cached Spotify playback state.

    Written by the effects layer after each poll; read by pure logic to
    decide when to pick the next track.
    """
    progress_ms: int = 0
    duration_ms: int = 0
    snapshot_time: float = 0.0   # monotonic seconds when last polled
    is_playing: bool = False
    volume: int = -1             # -1 = unknown
    current_track_id: str | None = None


# ---------------------------------------------------------------------------
# Orbit state
# ---------------------------------------------------------------------------

class OrbitPhase(str, Enum):
    DWELL = "dwell"
    TRANSIT = "transit"


class OrbitAnchorState(BaseModel, frozen=True):
    """One waypoint in the orbit.  Embedding data lives in the database —
    only IDs and descriptions travel in state."""
    description: str
    track_id: str


class OrbitState(BaseModel, frozen=True):
    """Complete orbit navigation state.  Pure logic updates this via
    ``model_copy``; the engine dispatches resulting commands."""
    anchors: tuple[OrbitAnchorState, ...] = ()
    current_index: int = 0
    allow_same_artist: bool = False

    phase: OrbitPhase = OrbitPhase.DWELL
    dwell_start_mono: float = 0.0
    transit_remaining: int = 0
    transit_total: int = 10            # total steps for this transit (dynamic)
    transit_initial_dist: float = 1.0
    transit_start_tid: str | None = None   # track playing when transit began
    arrived: bool = False
    completed_segments: frozenset[int] = frozenset()

    # Track ID of the last track played during this orbit (for transit
    # progress computation — the actual embedding is fetched by the engine).
    last_played_tid: str | None = None


# ---------------------------------------------------------------------------
# Mood state
# ---------------------------------------------------------------------------

class MoodState(BaseModel, frozen=True):
    """Mood descriptor state.

    The mood *mask* (a numpy bool array over all tracks) is derived data
    computed by the effects layer and cached in the engine — it is NOT
    stored here, keeping state serialisable and small.
    """
    mood_text: str | None = None
    descriptors: tuple[str, ...] = ()
    threshold: float = 0.35


# ---------------------------------------------------------------------------
# DJ state — the single source of truth
# ---------------------------------------------------------------------------

class DJState(BaseModel, frozen=True):
    """Complete DJ state.  Immutable — logic returns new copies.

    Contains everything the pure navigation logic needs to decide what
    happens next.  Does NOT contain numpy arrays (embeddings are
    reference data held by the database / engine cache).
    """

    # ── Navigation parameters ────────────────────────────────────────
    song_k: int = 10
    hop_multiplier: float = 5.0
    hop_interval_seconds: float = 1800.0   # 30 min default
    mode: str = "exact"                    # "exact" or "listen"

    # ── Track history ────────────────────────────────────────────────
    history: tuple[str, ...] = ()          # ordered track IDs
    played: frozenset[str] = frozenset()
    # Maps track_id → set label for tracks that start a new set.
    # Labels: "NEW SET", "ORBIT DWELL", "ORBIT TRANSIT"
    set_starts: tuple[tuple[str, str], ...] = ()  # ((track_id, label), ...)

    # ── Timing ───────────────────────────────────────────────────────
    last_hop_time: float = 0.0             # monotonic seconds

    # ── Pending-pick state machine ───────────────────────────────────
    next_pick: str | None = None           # picked but not yet played
    monitored_track: str | None = None     # Spotify ID we're watching
    monitor_start: float = 0.0             # when monitoring began
    pending_hop_type: str | None = None    # "NEW SET", "ORBIT DWELL", "ORBIT TRANSIT", or None
    orbit_picked: bool = False

    # ── Sub-states ───────────────────────────────────────────────────
    orbit: OrbitState | None = None
    mood: MoodState = Field(default_factory=MoodState)
    playback: PlaybackSnapshot = Field(default_factory=PlaybackSnapshot)

    # ── Flags ────────────────────────────────────────────────────────
    live_emb_available: bool = False
    stop_requested: bool = False

    # ── Library size (for display / exhaustion check) ────────────────
    total_tracks: int = 0
