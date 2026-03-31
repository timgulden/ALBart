"""Tests for core/navigation.py — main DJ decision functions."""

from __future__ import annotations

import numpy as np
import pytest

from albart.core.commands import (
    BroadcastToMapCommand,
    ComputeLongHopTargetCommand,
    FindNeighborsCommand,
    PlayTrackCommand,
    ResumePlaybackCommand,
    UpdateOrbitPositionCommand,
)
from albart.core.navigation import (
    on_neighbors_found,
    on_orbit_transit_arrival,
    on_override,
    on_poll_tick,
    on_track_played,
)
from albart.core.state import (
    DJState,
    OrbitAnchorState,
    OrbitPhase,
    OrbitState,
    PlaybackSnapshot,
)


@pytest.fixture
def playing_state():
    """State with one track in history and active playback."""
    return DJState(
        history=("track_1",),
        played=frozenset({"track_1"}),
        total_tracks=100,
        playback=PlaybackSnapshot(
            progress_ms=180_000,
            duration_ms=200_000,
            snapshot_time=1000.0,
            is_playing=True,
            current_track_id="track_1",
        ),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestOnPollTick:
    def test_noop_when_no_history(self):
        state = DJState()
        pb = PlaybackSnapshot()
        result = on_poll_tick(state, pb, now=0.0)
        assert result.commands == []

    def test_noop_when_stop_requested(self, playing_state):
        state = playing_state.model_copy(update={"stop_requested": True})
        result = on_poll_tick(state, state.playback, now=1000.0)
        assert result.commands == []

    def test_resume_when_paused_long(self, playing_state):
        """Should resume playback if paused for > 15 seconds."""
        pb = PlaybackSnapshot(
            is_playing=False,
            current_track_id=None,
            snapshot_time=980.0,  # 20s ago
        )
        result = on_poll_tick(playing_state, pb, now=1000.0)
        assert any(isinstance(c, ResumePlaybackCommand) for c in result.commands)

    def test_no_resume_when_briefly_paused(self, playing_state):
        """Brief pause during track transition — don't resume."""
        pb = PlaybackSnapshot(
            is_playing=False,
            current_track_id=None,
            snapshot_time=995.0,  # only 5s ago
        )
        result = on_poll_tick(playing_state, pb, now=1000.0)
        assert not any(isinstance(c, ResumePlaybackCommand) for c in result.commands)

    def test_manual_change_detected(self, playing_state):
        """When Spotify track changes and we have no pending pick."""
        pb = playing_state.playback.model_copy(update={
            "current_track_id": "new_track",
        })
        result = on_poll_tick(playing_state, pb, now=1001.0)
        assert "new_track" in result.state.history
        assert "new_track" in result.state.played

    def test_pending_pick_plays_at_song_end(self, playing_state):
        """When a pick is pending and song is about to end."""
        state = playing_state.model_copy(update={
            "next_pick": "track_2",
            "monitored_track": "track_1",
            "monitor_start": 990.0,
        })
        # Song about to end: 1s remaining
        pb = state.playback.model_copy(update={
            "progress_ms": 199_000,
            "duration_ms": 200_000,
            "snapshot_time": 1000.0,
        })
        result = on_poll_tick(state, pb, now=1000.0)
        play_cmds = [c for c in result.commands if isinstance(c, PlayTrackCommand)]
        assert len(play_cmds) == 1
        assert play_cmds[0].track_id == "track_2"
        assert result.state.next_pick is None

    def test_normal_pick_near_song_end(self, playing_state):
        """Should initiate pick when remaining < 8s and no pending pick."""
        pb = playing_state.playback.model_copy(update={
            "progress_ms": 195_000,
            "duration_ms": 200_000,
            "snapshot_time": 1000.0,
        })
        result = on_poll_tick(playing_state, pb, now=1000.0)
        find_cmds = [c for c in result.commands if isinstance(c, FindNeighborsCommand)]
        assert len(find_cmds) == 1
        assert find_cmds[0].hop_type == "normal"

    def test_too_early_to_pick(self, playing_state):
        """Should not pick when most of song remaining."""
        pb = playing_state.playback.model_copy(update={
            "progress_ms": 10_000,
            "duration_ms": 200_000,
            "snapshot_time": 1000.0,
        })
        result = on_poll_tick(playing_state, pb, now=1000.0)
        find_cmds = [c for c in result.commands if isinstance(c, FindNeighborsCommand)]
        assert len(find_cmds) == 0

    def test_long_hop_after_interval(self, playing_state):
        """Should trigger long hop when hop interval has elapsed."""
        state = playing_state.model_copy(update={
            "history": ("track_0", "track_1"),
            "played": frozenset({"track_0", "track_1"}),
            "last_hop_time": 0.0,  # 1800s ago
            "hop_interval_seconds": 1800.0,
        })
        pb = state.playback.model_copy(update={
            "progress_ms": 195_000,
            "duration_ms": 200_000,
            "snapshot_time": 1800.0,
        })
        result = on_poll_tick(state, pb, now=1800.0)
        long_cmds = [c for c in result.commands if isinstance(c, ComputeLongHopTargetCommand)]
        assert len(long_cmds) == 1

    def test_override_takes_priority(self, playing_state):
        result = on_poll_tick(
            playing_state, playing_state.playback,
            now=1000.0, override_tid="override_track",
        )
        play_cmds = [c for c in result.commands if isinstance(c, PlayTrackCommand)]
        assert len(play_cmds) == 1
        assert play_cmds[0].track_id == "override_track"


class TestOnNeighborsFound:
    def test_picks_from_candidates(self, playing_state, rng):
        candidates = [("t2", 0.1), ("t3", 0.2), ("t4", 0.3)]
        artists = {"t2": "a", "t3": "b", "t4": "c"}
        result = on_neighbors_found(
            playing_state, candidates, artists, "normal", 1000.0, rng,
        )
        assert result.state.next_pick in {"t2", "t3", "t4"}
        assert result.state.monitored_track is not None

    def test_no_candidates_resets_played(self, playing_state, rng):
        result = on_neighbors_found(
            playing_state, [], {}, "normal", 1000.0, rng,
        )
        assert result.state.played == frozenset()
        assert result.state.next_pick is None

    def test_orbit_no_candidates_doesnt_reset(self, rng):
        state = DJState(
            orbit=OrbitState(
                anchors=(OrbitAnchorState(description="A", track_id="t1"),),
            ),
            history=("t1",),
            played=frozenset({"t1"}),
        )
        result = on_neighbors_found(state, [], {}, "orbit_dwell", 1000.0, rng)
        assert result.state.played == frozenset({"t1"})  # not cleared


class TestOnTrackPlayed:
    def test_updates_history(self, playing_state):
        result = on_track_played(playing_state, "track_2", now=1001.0)
        assert result.state.history[-1] == "track_2"
        assert "track_2" in result.state.played
        assert result.state.next_pick is None

    def test_long_hop_marks_set_start(self, playing_state):
        state = playing_state.model_copy(update={"pending_hop_type": "NEW SET"})
        result = on_track_played(state, "track_2", now=1001.0)
        labels = dict(result.state.set_starts)
        assert labels["track_2"] == "NEW SET"
        assert result.state.pending_hop_type is None

    def test_orbit_position_update(self):
        state = DJState(
            orbit=OrbitState(
                anchors=(OrbitAnchorState(description="A", track_id="t1"),),
            ),
            orbit_picked=True,
            history=("t1",),
        )
        result = on_track_played(state, "t2", now=100.0)
        assert result.state.orbit.last_played_tid == "t2"
        update_cmds = [c for c in result.commands
                       if isinstance(c, UpdateOrbitPositionCommand)]
        assert len(update_cmds) == 1

    def test_broadcasts_to_map(self, playing_state):
        result = on_track_played(playing_state, "track_2", now=1001.0)
        bcast = [c for c in result.commands if isinstance(c, BroadcastToMapCommand)]
        assert len(bcast) == 1
        assert bcast[0].track_id == "track_2"

    def test_resets_playback_cache(self, playing_state):
        result = on_track_played(playing_state, "track_2", now=1001.0)
        assert result.state.playback.progress_ms == 0
        assert result.state.playback.duration_ms == 300_000


class TestOnOverride:
    def test_override(self, playing_state):
        result = on_override(playing_state, "override_t", now=1002.0)
        assert result.state.last_hop_time == 1002.0
        assert result.state.next_pick is None
        play_cmds = [c for c in result.commands if isinstance(c, PlayTrackCommand)]
        assert play_cmds[0].track_id == "override_t"


class TestOnOrbitTransitArrival:
    def test_transitions_to_dwell(self):
        state = DJState(
            orbit=OrbitState(
                anchors=(
                    OrbitAnchorState(description="A", track_id="t1"),
                    OrbitAnchorState(description="B", track_id="t2"),
                ),
                current_index=1,
                phase=OrbitPhase.TRANSIT,
            ),
        )
        new_state = on_orbit_transit_arrival(state, now=5000.0)
        assert new_state.orbit.phase == OrbitPhase.DWELL
        assert new_state.pending_hop_type == "ORBIT DWELL"

    def test_no_orbit_noop(self):
        state = DJState()
        new_state = on_orbit_transit_arrival(state, now=5000.0)
        assert new_state.orbit is None
