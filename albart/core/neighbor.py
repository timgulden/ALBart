"""Build neighbor-query commands from current state.

Pure helper that populates ``FindNeighborsCommand`` with the right
exclusion sets, artist penalty, and mood flag based on DJ state.
"""

from __future__ import annotations

from albart.core.commands import FindNeighborsCommand
from albart.core.sampling import get_recent_artists
from albart.core.state import DJState


def build_neighbor_query(
    state: DJState,
    *,
    target_track_id: str | None = None,
    use_target_embedding: str | None = None,
    space: str = "512d",
    k: int | None = None,
    hop_type: str = "normal",
) -> FindNeighborsCommand:
    """Construct a ``FindNeighborsCommand`` from current DJ state.

    Automatically populates:
    - ``exclude_played`` from ``state.played``
    - ``recent_artists`` from ``state.history``
    - ``artist_penalty``: 0.01 for orbit modes, 0.1 for normal
    - ``mood_mask_active``: True when mood descriptors are set
    - ``allow_same_artist``: from orbit state
    - ``k``: defaults to ``state.song_k``

    Args:
        state: current DJ state.
        target_track_id: track whose embedding to query around.
        use_target_embedding: engine cache key for a pre-computed target.
        space: ``"512d"`` or ``"5d"``.
        k: override for candidate count (defaults to ``state.song_k``).
        hop_type: ``"normal"``, ``"LONG"``, ``"orbit_dwell"``,
            ``"orbit_transit"``.
    """
    in_orbit = state.orbit is not None
    allow_same = in_orbit and state.orbit.allow_same_artist

    # Orbit modes get stronger artist penalty
    if in_orbit and not allow_same:
        penalty = 0.01
    else:
        penalty = 0.1

    return FindNeighborsCommand(
        target_track_id=target_track_id,
        use_target_embedding=use_target_embedding,
        space=space,
        k=k if k is not None else state.song_k,
        exclude_played=state.played,
        mood_mask_active=bool(state.mood.descriptors),
        recent_artists=get_recent_artists(state.history, {}),
        artist_penalty=penalty,
        allow_same_artist=allow_same,
        hop_type=hop_type,
    )
