"""Tests for core/mood.py — descriptor parsing and mood state updates."""

from __future__ import annotations

from albart.core.commands import ComputeMoodMaskCommand
from albart.core.mood import (
    clear_mood,
    parse_mood_descriptors,
    should_recompute_mask,
    update_mood,
)
from albart.core.state import DJState, MoodState


class TestParseMoodDescriptors:
    def test_positive_only(self):
        pos, neg = parse_mood_descriptors([
            "dark ambient",
            "trip hop",
            "downtempo",
        ])
        assert pos == ["dark ambient", "trip hop", "downtempo"]
        assert neg == []

    def test_negative_prefix(self):
        pos, neg = parse_mood_descriptors([
            "jazz fusion",
            "NOT: country pop",
            "NOT: bubblegum pop",
            "smooth jazz",
        ])
        assert pos == ["jazz fusion", "smooth jazz"]
        assert neg == ["country pop", "bubblegum pop"]

    def test_case_insensitive_not(self):
        pos, neg = parse_mood_descriptors(["not: happy hardcore"])
        assert pos == []
        assert neg == ["happy hardcore"]

    def test_empty_lines_skipped(self):
        pos, neg = parse_mood_descriptors(["a", "", "  ", "b"])
        assert pos == ["a", "b"]
        assert neg == []


class TestShouldRecomputeMask:
    def test_same_mood_no_recompute(self):
        m = MoodState(descriptors=("a", "b"), threshold=0.35)
        assert not should_recompute_mask(m, m)

    def test_descriptors_changed(self):
        old = MoodState(descriptors=("a",), threshold=0.35)
        new = MoodState(descriptors=("a", "b"), threshold=0.35)
        assert should_recompute_mask(old, new)

    def test_threshold_changed(self):
        old = MoodState(descriptors=("a",), threshold=0.35)
        new = MoodState(descriptors=("a",), threshold=0.50)
        assert should_recompute_mask(old, new)


class TestUpdateMood:
    def test_update_descriptors(self):
        state = DJState()
        result = update_mood(state, descriptors=["dark", "ambient"])
        assert result.state.mood.descriptors == ("dark", "ambient")
        # Should issue recompute command
        assert len(result.commands) == 1
        assert isinstance(result.commands[0], ComputeMoodMaskCommand)

    def test_update_threshold_only(self):
        state = DJState(mood=MoodState(descriptors=("x",), threshold=0.3))
        result = update_mood(state, threshold=0.5)
        assert result.state.mood.threshold == 0.5
        assert result.state.mood.descriptors == ("x",)
        assert len(result.commands) == 1

    def test_no_change_no_command(self):
        m = MoodState(descriptors=("a",), threshold=0.35)
        state = DJState(mood=m)
        result = update_mood(state, mood_text="some text")
        # Only mood_text changed, not descriptors or threshold
        assert result.state.mood.mood_text == "some text"
        assert len(result.commands) == 0


class TestClearMood:
    def test_clear(self):
        state = DJState(mood=MoodState(
            descriptors=("a", "b"),
            threshold=0.5,
            mood_text="dark",
        ))
        result = clear_mood(state)
        assert result.state.mood.descriptors == ()
        assert result.state.mood.mood_text is None
        # Should issue a recompute (engine clears the mask)
        assert len(result.commands) == 1
