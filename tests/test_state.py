"""Tests for core/state.py — immutability and model_copy correctness."""

from __future__ import annotations

import pytest

from albart.core.state import (
    DJState,
    MoodState,
    OrbitAnchorState,
    OrbitPhase,
    OrbitState,
    PlaybackSnapshot,
    TrackRef,
)


class TestDJStateImmutability:
    def test_frozen_rejects_mutation(self):
        s = DJState()
        with pytest.raises(Exception):
            s.song_k = 99

    def test_model_copy_preserves_original(self):
        s = DJState(song_k=10, total_tracks=5000)
        s2 = s.model_copy(update={"song_k": 20})
        assert s.song_k == 10
        assert s2.song_k == 20

    def test_history_is_tuple(self):
        s = DJState(history=("a", "b", "c"))
        assert isinstance(s.history, tuple)
        assert len(s.history) == 3

    def test_played_is_frozenset(self):
        s = DJState(played=frozenset({"a", "b"}))
        assert isinstance(s.played, frozenset)

    def test_append_to_history_via_copy(self):
        s = DJState(history=("a",))
        s2 = s.model_copy(update={"history": (*s.history, "b")})
        assert s.history == ("a",)
        assert s2.history == ("a", "b")

    def test_add_to_played_via_copy(self):
        s = DJState(played=frozenset({"a"}))
        s2 = s.model_copy(update={"played": s.played | {"b"}})
        assert "b" not in s.played
        assert "b" in s2.played

    def test_defaults(self):
        s = DJState()
        assert s.song_k == 10
        assert s.hop_multiplier == 5.0
        assert s.mode == "exact"
        assert s.history == ()
        assert s.played == frozenset()
        assert s.orbit is None
        assert s.stop_requested is False


class TestOrbitState:
    def test_frozen(self):
        o = OrbitState()
        with pytest.raises(Exception):
            o.current_index = 5

    def test_phase_enum(self):
        o = OrbitState(phase=OrbitPhase.TRANSIT)
        assert o.phase == OrbitPhase.TRANSIT
        assert o.phase.value == "transit"

    def test_completed_segments_immutable(self):
        o = OrbitState(completed_segments=frozenset({0, 1}))
        o2 = o.model_copy(update={
            "completed_segments": o.completed_segments | {2},
        })
        assert 2 not in o.completed_segments
        assert 2 in o2.completed_segments


class TestPlaybackSnapshot:
    def test_defaults(self):
        p = PlaybackSnapshot()
        assert p.progress_ms == 0
        assert p.duration_ms == 0
        assert p.is_playing is False
        assert p.volume == -1
        assert p.current_track_id is None

    def test_frozen(self):
        p = PlaybackSnapshot(progress_ms=1000)
        with pytest.raises(Exception):
            p.progress_ms = 2000


class TestMoodState:
    def test_defaults(self):
        m = MoodState()
        assert m.mood_text is None
        assert m.descriptors == ()
        assert m.threshold == 0.35


class TestTrackRef:
    def test_creation(self):
        t = TrackRef(track_id="abc", title="Song", artist="Band", album="LP")
        assert t.track_id == "abc"
