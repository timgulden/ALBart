"""Pure navigation logic — the heart of the DJ.

Every function takes immutable state and returns ``LogicResult(state, commands)``.
No I/O, no side effects, no ``time.monotonic()`` calls — current time is a
parameter.

This module replaces the body of ``DJ.run()`` and the four picking methods.
"""

from __future__ import annotations

import numpy as np

from albart.core.commands import (
    BroadcastToMapCommand,
    Command,
    ComputeLongHopTargetCommand,
    ComputeTransitTargetCommand,
    FindNeighborsCommand,
    LogicResult,
    PlayTrackCommand,
    ResumePlaybackCommand,
    UpdateOrbitPositionCommand,
)
from albart.core.neighbor import build_neighbor_query
from albart.core.orbit_logic import (
    advance_transit_step,
    record_played,
    should_leave_dwell,
    start_dwell,
    start_transit,
    transit_done,
)
from albart.core.sampling import get_recent_artists, select_from_candidates
from albart.core.state import DJState, OrbitPhase, PlaybackSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remaining_ms(playback: PlaybackSnapshot, now: float) -> int:
    """Estimate remaining playback time from cached snapshot."""
    if playback.duration_ms <= 0:
        return 0
    elapsed = now - playback.snapshot_time
    estimated_progress = playback.progress_ms + int(elapsed * 1000)
    return max(0, playback.duration_ms - estimated_progress)


def _noop(state: DJState) -> LogicResult:
    """No action this tick."""
    return LogicResult(state=state)


# ---------------------------------------------------------------------------
# on_poll_tick — called every ~5 seconds by the engine
# ---------------------------------------------------------------------------

def on_poll_tick(
    state: DJState,
    playback: PlaybackSnapshot,
    now: float,
    *,
    override_tid: str | None = None,
    has_live_emb: bool = False,
) -> LogicResult:
    """Main decision function.  Replaces the body of ``DJ.run()``'s while loop.

    Args:
        state: current immutable DJ state.
        playback: latest Spotify playback snapshot (from effects poll).
        now: current monotonic time.
        override_tid: track ID from map click override (if any).
        has_live_emb: whether a live audio embedding is available.

    Returns:
        LogicResult with updated state and commands to execute.
    """
    state = state.model_copy(update={
        "playback": playback,
        "live_emb_available": has_live_emb,
    })

    if state.stop_requested:
        return _noop(state)

    if not state.history:
        return _noop(state)

    # ── Handle override ──────────────────────────────────────────────
    if override_tid is not None:
        return on_override(state, override_tid, now)

    # ── Broadcast current track to map ───────────────────────────────
    commands: list[Command] = []
    last_tid = state.history[-1]
    commands.append(BroadcastToMapCommand(track_id=last_tid))

    # ── Check: Spotify paused or stopped → try to resume ─────────────
    current = playback.current_track_id
    if current is None:
        since_last = now - playback.snapshot_time
        if not playback.is_playing and since_last > 15:
            return LogicResult(
                state=state,
                commands=[ResumePlaybackCommand()],
            )
        return LogicResult(state=state, commands=commands)

    # ── Manual track change detection ────────────────────────────────
    if current != last_tid and state.next_pick is None:
        # User changed the track in Spotify — follow along.
        # Allow re-playing tracks already in history (e.g. user switches
        # back to a just-ingested track).
        state = state.model_copy(update={
            "played": state.played | {current},
            "history": (*state.history, current),
            "last_hop_time": now,
        })
        return LogicResult(
            state=state,
            commands=[BroadcastToMapCommand(track_id=current)],
            log_message=f"Manual change: {current}",
        )

    # ── Spotify Control mode: just follow, don't pick ──────────────
    if not state.dj_active:
        return LogicResult(state=state, commands=commands)

    # ── Pending pick: play when song ends ────────────────────────────
    remaining = _remaining_ms(playback, now)

    if state.next_pick is not None:
        track_changed = (
            state.monitored_track is not None
            and current != state.monitored_track
        )
        elapsed_monitoring = now - state.monitor_start
        track_restarted = (
            current == state.monitored_track
            and elapsed_monitoring > 10
            and playback.progress_ms < 3000
        )
        if track_changed or track_restarted or remaining <= 2000:
            pick = state.next_pick
            state = state.model_copy(update={
                "next_pick": None,
                "monitored_track": None,
            })
            return LogicResult(
                state=state,
                commands=[PlayTrackCommand(track_id=pick)] + commands,
            )
        return LogicResult(state=state, commands=commands)

    # ── Decide whether it's time to pick ─────────────────────────────
    if state.orbit is not None:
        # Orbit mode: pick eagerly once track starts playing
        if remaining == 0:
            return LogicResult(state=state, commands=commands)  # wait for track to start
        # Fall through to pick
    else:
        # Normal mode: only pick near end of song
        if remaining == 0 or remaining > 8000:
            return LogicResult(state=state, commands=commands)

    # ── Pick next track ──────────────────────────────────────────────
    return _initiate_pick(state, now, commands)


def _initiate_pick(
    state: DJState,
    now: float,
    pending_commands: list[Command],
) -> LogicResult:
    """Decide which pick strategy to use and issue the right command.

    The actual neighbor search happens asynchronously — the engine will
    call ``on_neighbors_found`` with the results.
    """
    last_tid = state.history[-1]
    time_since_hop = now - state.last_hop_time

    commands: list[Command] = list(pending_commands)

    # ── Orbit mode ───────────────────────────────────────────────────
    if state.orbit is not None:
        return _initiate_orbit_pick(state, now, commands)

    # ── Long hop (set change) ────────────────────────────────────────
    if time_since_hop >= state.hop_interval_seconds and len(state.history) >= 2:
        state = state.model_copy(update={
            "last_hop_time": now,
            "pending_hop_type": "NEW SET",
        })
        prev_tid = state.history[-2]
        commands.append(ComputeLongHopTargetCommand(
            prev_track_id=prev_tid,
            current_track_id=last_tid,
            hop_multiplier=state.hop_multiplier,
        ))
        return LogicResult(state=state, commands=commands)

    # ── Normal hop ───────────────────────────────────────────────────
    query = build_neighbor_query(
        state,
        target_track_id=last_tid,
        space="25d",
        hop_type="normal",
    )
    commands.append(query)
    return LogicResult(state=state, commands=commands)


def _initiate_orbit_pick(
    state: DJState,
    now: float,
    pending_commands: list[Command],
) -> LogicResult:
    """Orbit-mode picking: dwell or transit."""
    orbit = state.orbit
    assert orbit is not None
    last_tid = state.history[-1]
    commands: list[Command] = list(pending_commands)

    # ── Check transit arrival → dwell transition ──────────────────────
    # The engine marks arrived=True when the last transit track starts.
    # We delay the actual phase change to here — when the arriving track
    # is ending and we're about to pick the first dwell track.
    if orbit.phase == OrbitPhase.TRANSIT and orbit.arrived:
        orbit = start_dwell(orbit, now)
        state = state.model_copy(update={
            "orbit": orbit,
            "pending_hop_type": "ORBIT DWELL",
        })

    # ── Check dwell → transit transition ─────────────────────────────
    if orbit.phase == OrbitPhase.DWELL and should_leave_dwell(orbit, now):
        # Compute initial distance for the transit (engine handles this
        # via the ComputeTransitTargetCommand).  For now, start transit
        # with a placeholder distance — engine will compute the real one.
        target_anchor_tid = orbit.anchors[
            (orbit.current_index + 1) % len(orbit.anchors)
        ].track_id
        # Issue transit target computation — engine will start_transit
        # on orbit state once it has the initial distance.
        orbit = start_transit(orbit, initial_distance=1.0, start_tid=last_tid)
        state = state.model_copy(update={
            "orbit": orbit,
            "pending_hop_type": "ORBIT TRANSIT",
        })

    # ── DWELL: 5D neighbor search around anchor ──────────────────────
    if orbit.phase == OrbitPhase.DWELL:
        anchor_tid = orbit.anchors[orbit.current_index].track_id
        query = build_neighbor_query(
            state,
            target_track_id=anchor_tid,
            space="25d",
            hop_type="orbit_dwell",
        )
        commands.append(query)
        return LogicResult(state=state, commands=commands)

    # ── TRANSIT: step through 512D toward next anchor ────────────────
    orbit, fraction = advance_transit_step(orbit)
    state = state.model_copy(update={"orbit": orbit})

    target_anchor_tid = orbit.anchors[orbit.current_index].track_id
    commands.append(ComputeTransitTargetCommand(
        current_track_id=last_tid,
        target_anchor_track_id=target_anchor_tid,
        transit_remaining=orbit.transit_remaining + 1,  # before decrement
    ))
    return LogicResult(state=state, commands=commands)


# ---------------------------------------------------------------------------
# on_neighbors_found — called by the engine after a FindNeighborsCommand
# ---------------------------------------------------------------------------

def on_neighbors_found(
    state: DJState,
    candidates: list[tuple[str, float]],
    track_artists: dict[str, str],
    hop_type: str,
    now: float,
    rng: np.random.Generator,
) -> LogicResult:
    """Select from candidates and set up the pending pick.

    Args:
        candidates: ``[(track_id, distance), ...]`` from the database.
        track_artists: ``{track_id: lowercase_artist}`` for the candidates.
        hop_type: ``"normal"``, ``"LONG"``, ``"orbit_dwell"``, ``"orbit_transit"``.
        now: current monotonic time.
        rng: numpy random generator.
    """
    in_orbit = state.orbit is not None
    allow_same = in_orbit and state.orbit.allow_same_artist
    penalty = 0.01 if in_orbit and not allow_same else 0.1

    recent = get_recent_artists(state.history, track_artists)

    next_tid = select_from_candidates(
        candidates,
        recent_artists=recent,
        artist_penalty=penalty,
        allow_same_artist=allow_same,
        song_k=state.song_k,
        track_artist_map=track_artists,
        rng=rng,
    )

    if next_tid is None:
        if not in_orbit:
            # Reset played set to avoid exhaustion
            state = state.model_copy(update={"played": frozenset()})
            return LogicResult(
                state=state,
                log_message="No candidates — reset played set",
            )
        return LogicResult(
            state=state,
            log_message="Orbit: no candidates found",
        )

    # Preserve pending_hop_type if already set (e.g. by orbit transition or
    # engine.new_set).  Only set from hop_type if not already labeled.
    pending = state.pending_hop_type
    if pending is None and hop_type == "NEW SET":
        pending = "NEW SET"

    state = state.model_copy(update={
        "next_pick": next_tid,
        "monitored_track": state.playback.current_track_id,
        "monitor_start": now,
        "pending_hop_type": pending,
        "orbit_picked": in_orbit,
    })

    return LogicResult(state=state)


# ---------------------------------------------------------------------------
# on_track_played — called after PlayTrackCommand succeeds
# ---------------------------------------------------------------------------

def on_track_played(
    state: DJState,
    track_id: str,
    now: float,
) -> LogicResult:
    """Update state after a track starts playing."""
    # Guard: don't double-add if already the last entry (race between
    # server-thread skip/new_set and engine loop)
    if state.history and state.history[-1] == track_id:
        return LogicResult(state=state)

    hop_label = state.pending_hop_type  # "NEW SET", "ORBIT DWELL", "ORBIT TRANSIT", or None
    if hop_label:
        new_set_starts = (*state.set_starts, (track_id, hop_label))
    else:
        new_set_starts = state.set_starts

    commands: list[Command] = [
        BroadcastToMapCommand(track_id=track_id),
    ]

    # Update orbit position for progress tracking
    new_orbit = state.orbit
    if new_orbit is not None and state.orbit_picked:
        new_orbit = record_played(new_orbit, track_id)
        commands.append(UpdateOrbitPositionCommand(track_id=track_id))

    # Reset playback cache to avoid stale remaining-ms triggering next pick
    fresh_playback = state.playback.model_copy(update={
        "progress_ms": 0,
        "duration_ms": 300_000,   # assume ~5 min until next poll
        "snapshot_time": now,
    })

    state = state.model_copy(update={
        "played": state.played | {track_id},
        "history": (*state.history, track_id),
        "pending_hop_type": None,
        "orbit_picked": False,
        "set_starts": new_set_starts,
        "orbit": new_orbit,
        "playback": fresh_playback,
        "next_pick": None,
        "monitored_track": None,
    })

    return LogicResult(
        state=state,
        commands=commands,
        log_message=hop_label,
    )


# ---------------------------------------------------------------------------
# on_override — handle map click or manual override
# ---------------------------------------------------------------------------

def on_override(
    state: DJState,
    track_id: str,
    now: float,
) -> LogicResult:
    """Handle an explicit track override (map click or API call)."""
    state = state.model_copy(update={
        "next_pick": None,
        "monitored_track": None,
        "last_hop_time": now,
    })
    return LogicResult(
        state=state,
        commands=[PlayTrackCommand(track_id=track_id)],
        log_message=f"Override: {track_id}",
    )


# ---------------------------------------------------------------------------
# on_orbit_transit_arrival — called by engine when transit target is reached
# ---------------------------------------------------------------------------

def on_orbit_transit_arrival(
    state: DJState,
    now: float,
) -> DJState:
    """Transition orbit to dwell after transit arrival."""
    if state.orbit is None:
        return state
    new_orbit = start_dwell(state.orbit, now)
    return state.model_copy(update={
        "orbit": new_orbit,
        "pending_hop_type": "ORBIT DWELL",
    })
