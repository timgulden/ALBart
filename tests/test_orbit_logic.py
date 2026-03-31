"""Tests for core/orbit_logic.py — pure orbit state machine."""

from __future__ import annotations

import pytest

from albart.core.orbit_logic import (
    ARRIVAL_THRESHOLD,
    DWELL_DURATION,
    TRANSIT_STEPS_MAX,
    advance_transit_step,
    check_arrival,
    dwell_elapsed,
    find_track,
    get_progress,
    normalize_search,
    record_played,
    should_leave_dwell,
    start_dwell,
    start_transit,
    transit_done,
)
from albart.core.state import OrbitAnchorState, OrbitPhase, OrbitState


@pytest.fixture
def dwell_orbit():
    """Orbit in dwell phase at anchor 0."""
    return OrbitState(
        anchors=(
            OrbitAnchorState(description="Anchor A", track_id="t1"),
            OrbitAnchorState(description="Anchor B", track_id="t2"),
            OrbitAnchorState(description="Anchor C", track_id="t3"),
        ),
        current_index=0,
        phase=OrbitPhase.DWELL,
        dwell_start_mono=1000.0,
    )


@pytest.fixture
def transit_orbit():
    """Orbit in transit phase heading to anchor 1."""
    return OrbitState(
        anchors=(
            OrbitAnchorState(description="Anchor A", track_id="t1"),
            OrbitAnchorState(description="Anchor B", track_id="t2"),
            OrbitAnchorState(description="Anchor C", track_id="t3"),
        ),
        current_index=1,
        phase=OrbitPhase.TRANSIT,
        transit_remaining=TRANSIT_STEPS_MAX,
        transit_total=TRANSIT_STEPS_MAX,
        transit_initial_dist=2.0,
    )


class TestDwellPhase:
    def test_dwell_elapsed(self, dwell_orbit):
        assert dwell_elapsed(dwell_orbit, 1500.0) == 500.0

    def test_should_leave_dwell_not_yet(self, dwell_orbit):
        assert not should_leave_dwell(dwell_orbit, 1000.0 + DWELL_DURATION - 1)

    def test_should_leave_dwell_expired(self, dwell_orbit):
        assert should_leave_dwell(dwell_orbit, 1000.0 + DWELL_DURATION + 1)

    def test_should_leave_dwell_wrong_phase(self, transit_orbit):
        """Transit phase should never trigger dwell leave."""
        assert not should_leave_dwell(transit_orbit, 999999.0)


class TestTransitPhase:
    def test_transit_done_initially_false(self, transit_orbit):
        assert not transit_done(transit_orbit)

    def test_transit_done_at_zero(self, transit_orbit):
        o = transit_orbit.model_copy(update={"transit_remaining": 0})
        assert transit_done(o)

    def test_advance_step_decrements(self, transit_orbit):
        new_o, frac = advance_transit_step(transit_orbit)
        assert new_o.transit_remaining == TRANSIT_STEPS_MAX - 1
        assert frac == pytest.approx(1.0 / TRANSIT_STEPS_MAX)
        # Original unchanged
        assert transit_orbit.transit_remaining == TRANSIT_STEPS_MAX

    def test_advance_step_fraction_increases(self, transit_orbit):
        """As steps count down, fraction gets larger (covers more distance)."""
        o = transit_orbit
        fractions = []
        for _ in range(TRANSIT_STEPS_MAX):
            o, frac = advance_transit_step(o)
            fractions.append(frac)
        assert fractions[-1] > fractions[0]

    def test_check_arrival_close(self, transit_orbit):
        """Within 15% of initial distance should trigger arrival."""
        close_dist = transit_orbit.transit_initial_dist * ARRIVAL_THRESHOLD * 0.5
        assert check_arrival(transit_orbit, close_dist)

    def test_check_arrival_far(self, transit_orbit):
        """Far from target should not trigger arrival."""
        far_dist = transit_orbit.transit_initial_dist * 0.5
        assert not check_arrival(transit_orbit, far_dist)

    def test_check_arrival_wrong_phase(self, dwell_orbit):
        assert not check_arrival(dwell_orbit, 0.0)


class TestPhaseTransitions:
    def test_start_dwell(self, transit_orbit):
        o = start_dwell(transit_orbit, current_time=5000.0)
        assert o.phase == OrbitPhase.DWELL
        assert o.dwell_start_mono == 5000.0
        assert o.arrived is True
        # Previous segment (index 0) should be marked completed
        prev_idx = (transit_orbit.current_index - 1) % len(transit_orbit.anchors)
        assert prev_idx in o.completed_segments

    def test_start_transit(self, dwell_orbit):
        o = start_transit(dwell_orbit, initial_distance=3.5)
        assert o.phase == OrbitPhase.TRANSIT
        assert o.current_index == 1  # next anchor
        assert o.transit_remaining == TRANSIT_STEPS_MAX
        assert o.arrived is False
        assert o.transit_initial_dist == 3.5

    def test_start_transit_wraps_around(self):
        """Last anchor should wrap to first."""
        o = OrbitState(
            anchors=(
                OrbitAnchorState(description="A", track_id="t1"),
                OrbitAnchorState(description="B", track_id="t2"),
            ),
            current_index=1,
            phase=OrbitPhase.DWELL,
        )
        o2 = start_transit(o, initial_distance=1.0)
        assert o2.current_index == 0  # wrapped

    def test_record_played(self, dwell_orbit):
        o = record_played(dwell_orbit, "track_99")
        assert o.last_played_tid == "track_99"
        assert dwell_orbit.last_played_tid is None  # original unchanged


class TestProgress:
    def test_dwell_progress_arrived(self, dwell_orbit):
        o = dwell_orbit.model_copy(update={"arrived": True})
        p = get_progress(o, current_time=1500.0)
        assert p["phase"] == "dwell"
        assert p["segment_progress"] == 1.0
        assert p["dwell_elapsed"] == 500.0

    def test_transit_progress(self, transit_orbit):
        p = get_progress(transit_orbit, current_time=0.0, segment_progress=0.6)
        assert p["phase"] == "transit"
        assert p["segment_progress"] == 0.6
        assert p["transit_remaining"] == TRANSIT_STEPS_MAX


class TestFuzzyMatching:
    def test_normalize_search(self):
        # em dash → hyphen → space, then whitespace collapsed
        assert normalize_search("Massive Attack — Teardrop") == "massive attack teardrop"
        assert normalize_search("hello-world") == "hello world"

    def test_find_track_exact(self):
        tracks = [
            {"track_id": "t1", "title": "Teardrop", "artist": "Massive Attack"},
            {"track_id": "t2", "title": "Angel", "artist": "Massive Attack"},
        ]
        result = find_track("Massive Attack — Teardrop", tracks, set())
        assert result == "t1"

    def test_find_track_skips_used(self):
        tracks = [
            {"track_id": "t1", "title": "Teardrop", "artist": "Massive Attack"},
        ]
        result = find_track("Massive Attack — Teardrop", tracks, {"t1"})
        assert result is None

    def test_find_track_partial(self):
        tracks = [
            {"track_id": "t1", "title": "Teardrop", "artist": "Massive Attack"},
        ]
        result = find_track("Teardrop", tracks, set())
        assert result == "t1"

    def test_find_track_no_match(self):
        tracks = [
            {"track_id": "t1", "title": "Completely Different", "artist": "Unknown"},
        ]
        result = find_track("Massive Attack — Teardrop", tracks, set())
        assert result is None
