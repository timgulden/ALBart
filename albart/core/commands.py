"""Command types emitted by pure logic, executed by the effects layer.

Commands are frozen Pydantic models — just data describing a desired side
effect.  The engine's ``execute_commands`` dispatcher routes each type to
the appropriate effect function.
"""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field

from albart.core.state import DJState


# ---------------------------------------------------------------------------
# Command types
# ---------------------------------------------------------------------------

class PlayTrackCommand(BaseModel, frozen=True):
    """Tell Spotify to start playing a track."""
    track_id: str
    via_queue: bool = False


class ResumePlaybackCommand(BaseModel, frozen=True):
    """Resume paused Spotify playback."""
    pass


class BroadcastToMapCommand(BaseModel, frozen=True):
    """Send this track's embedding to the map display via UDP."""
    track_id: str


class FindNeighborsCommand(BaseModel, frozen=True):
    """Ask the database for nearest-neighbor candidates.

    The engine executes this, then calls ``navigation.on_neighbors_found``
    with the results.
    """
    target_track_id: str | None = None
    space: str = "512d"                              # "512d" or "25d"
    k: int = 100
    exclude_played: frozenset[str] = frozenset()
    mood_mask_active: bool = False
    recent_artists: tuple[str, ...] = ()
    artist_penalty: float = 0.1
    allow_same_artist: bool = False
    hop_type: str = "normal"                         # "normal", "LONG", "orbit_dwell", "orbit_transit"
    # For orbit transit / long hop: pre-computed 512D target key
    use_target_embedding: str | None = None          # engine cache key


class ComputeTransitTargetCommand(BaseModel, frozen=True):
    """Interpolate toward the orbit anchor in 512D space.

    The engine computes the target embedding and stores it in cache
    under ``result_key``, then issues a FindNeighborsCommand with
    ``use_target_embedding=result_key``.
    """
    current_track_id: str
    target_anchor_track_id: str
    transit_remaining: int
    result_key: str = "transit_target"


class ComputeLongHopTargetCommand(BaseModel, frozen=True):
    """Extrapolate the trajectory for a long hop.

    Engine computes the target, stores under ``result_key``, then issues
    a FindNeighborsCommand.
    """
    prev_track_id: str
    current_track_id: str
    hop_multiplier: float
    result_key: str = "long_hop_target"


class ComputeMoodMaskCommand(BaseModel, frozen=True):
    """Recompute the mood mask from descriptors.

    The engine calls the text embedder + database to produce a boolean
    mask and caches it.
    """
    descriptors: tuple[str, ...] = ()
    threshold: float = 0.35


class UpdateOrbitPositionCommand(BaseModel, frozen=True):
    """Tell the engine to record the track's 512D embedding for orbit
    progress tracking."""
    track_id: str


class IngestTrackCommand(BaseModel, frozen=True):
    """Trigger background ingestion of an unknown Spotify track.

    The engine dispatches this to a background thread that downloads
    the preview, computes CLAP embedding + 25D UMAP projection, and
    stores everything in PostgreSQL.
    """
    track_id: str


# ---------------------------------------------------------------------------
# Union of all command types
# ---------------------------------------------------------------------------

Command = Union[
    PlayTrackCommand,
    ResumePlaybackCommand,
    BroadcastToMapCommand,
    FindNeighborsCommand,
    ComputeTransitTargetCommand,
    ComputeLongHopTargetCommand,
    ComputeMoodMaskCommand,
    UpdateOrbitPositionCommand,
    IngestTrackCommand,
]


# ---------------------------------------------------------------------------
# LogicResult — the return type of every pure logic function
# ---------------------------------------------------------------------------

class LogicResult(BaseModel):
    """Returned by pure logic functions: new state + commands to execute."""
    state: DJState
    commands: list[Command] = Field(default_factory=list)
    log_message: str | None = None
