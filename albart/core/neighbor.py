"""Build neighbor-query commands from current state.

Pure helper that populates ``FindNeighborsCommand`` with the right
exclusion sets, artist penalty, and mood flag based on DJ state.
"""

from __future__ import annotations

from albart.core.commands import FindNeighborsCommand
from albart.core.sampling import get_recent_artists
from albart.core.state import DJState

# Always fetch 100 candidates; temperature controls the selection
# distribution over these candidates.
NEIGHBOR_K = 100


def build_neighbor_query(
    state: DJState,
    *,
    target_track_id: str | None = None,
    use_target_embedding: str | None = None,
    space: str = "512d",
    k: int = NEIGHBOR_K,
    hop_type: str = "normal",
) -> FindNeighborsCommand:
    """Construct a ``FindNeighborsCommand`` from current DJ state.

    Automatically populates:
    - ``exclude_played`` from ``state.played``
    - ``recent_artists`` from ``state.history``
    - ``artist_penalty``: 0.1 for normal
    - ``mood_mask_active``: True when mood descriptors are set
    - ``allow_same_artist``: from orbit state
    - ``k``: defaults to 100 (temperature controls selection width)

    Args:
        state: current DJ state.
        target_track_id: track whose embedding to query around.
        use_target_embedding: engine cache key for a pre-computed target.
        space: ``"512d"`` or ``"25d"``.
        k: candidate count (defaults to 100).
        hop_type: ``"normal"``, ``"LONG"``, ``"orbit_dwell"``,
            ``"orbit_transit"``.
    """
    in_orbit = state.orbit is not None
    allow_same = in_orbit and state.orbit.allow_same_artist

    # Same artist penalty: light enough to allow repeats when
    # the neighborhood is dominated by one artist (e.g. orbit
    # dwell on Ray Charles), but enough to add variety
    penalty = 0.1

    return FindNeighborsCommand(
        target_track_id=target_track_id,
        use_target_embedding=use_target_embedding,
        space=space,
        k=k,
        exclude_played=state.played,
        mood_mask_active=bool(state.mood.descriptors),
        recent_artists=get_recent_artists(state.history, {}),
        artist_penalty=penalty,
        allow_same_artist=allow_same,
        hop_type=hop_type,
    )
