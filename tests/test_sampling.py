"""Tests for core/sampling.py — unified weighted selection."""

from __future__ import annotations

import numpy as np
import pytest

from albart.core.sampling import get_recent_artists, select_from_candidates


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def candidates():
    """5 candidates with increasing distance."""
    return [
        ("track_a", 0.1),
        ("track_b", 0.2),
        ("track_c", 0.3),
        ("track_d", 0.5),
        ("track_e", 1.0),
    ]


@pytest.fixture
def artist_map():
    return {
        "track_a": "artist_x",
        "track_b": "artist_y",
        "track_c": "artist_x",
        "track_d": "artist_z",
        "track_e": "artist_w",
    }


class TestSelectFromCandidates:
    def test_empty_returns_none(self, rng):
        assert select_from_candidates(
            [],
            recent_artists=(),
            artist_penalty=0.1,
            allow_same_artist=False,
            song_k=10,
            track_artist_map={},
            rng=rng,
        ) is None

    def test_single_candidate(self, rng):
        result = select_from_candidates(
            [("only_track", 0.5)],
            recent_artists=(),
            artist_penalty=0.1,
            allow_same_artist=False,
            song_k=10,
            track_artist_map={},
            rng=rng,
        )
        assert result == "only_track"

    def test_song_k_1_picks_nearest(self, candidates, rng, artist_map):
        result = select_from_candidates(
            candidates,
            recent_artists=(),
            artist_penalty=0.1,
            allow_same_artist=False,
            song_k=1,
            track_artist_map=artist_map,
            rng=rng,
        )
        assert result == "track_a"

    def test_song_k_trims_pool(self, candidates, rng, artist_map):
        """With k=2, only the first 2 candidates should be pickable."""
        picks = set()
        for seed in range(200):
            r = np.random.default_rng(seed)
            pick = select_from_candidates(
                candidates,
                recent_artists=(),
                artist_penalty=0.1,
                allow_same_artist=False,
                song_k=2,
                track_artist_map=artist_map,
                rng=r,
            )
            picks.add(pick)
        assert picks == {"track_a", "track_b"}

    def test_artist_penalty_reduces_weight(self, candidates, artist_map):
        """Same-artist tracks should be picked much less often."""
        # artist_x = track_a and track_c
        picks = {"track_a": 0, "track_b": 0, "track_c": 0}
        for seed in range(1000):
            r = np.random.default_rng(seed)
            pick = select_from_candidates(
                candidates[:3],
                recent_artists=("artist_x",),
                artist_penalty=0.01,
                allow_same_artist=False,
                song_k=3,
                track_artist_map=artist_map,
                rng=r,
            )
            if pick in picks:
                picks[pick] += 1

        # track_b should dominate since artist_x is penalized
        assert picks["track_b"] > picks["track_a"] * 5

    def test_allow_same_artist_skips_penalty(self, candidates, artist_map):
        """With allow_same_artist=True, no penalty should be applied."""
        picks = {"track_a": 0, "track_b": 0}
        for seed in range(1000):
            r = np.random.default_rng(seed)
            pick = select_from_candidates(
                candidates[:2],
                recent_artists=("artist_x",),
                artist_penalty=0.01,
                allow_same_artist=True,
                song_k=2,
                track_artist_map=artist_map,
                rng=r,
            )
            if pick in picks:
                picks[pick] += 1

        # track_a is closer so should still be picked more, but not
        # overwhelmingly more than if penalty were applied
        ratio = picks["track_a"] / max(picks["track_b"], 1)
        assert ratio < 10  # without penalty, ratio is moderate

    def test_inverse_distance_weighting(self, rng, artist_map):
        """Closer candidates should be picked more often."""
        cands = [("near", 0.01), ("far", 1.0)]
        picks = {"near": 0, "far": 0}
        for seed in range(1000):
            r = np.random.default_rng(seed)
            pick = select_from_candidates(
                cands,
                recent_artists=(),
                artist_penalty=0.1,
                allow_same_artist=False,
                song_k=2,
                track_artist_map=artist_map,
                rng=r,
            )
            picks[pick] += 1
        assert picks["near"] > picks["far"] * 10


class TestGetRecentArtists:
    def test_basic(self):
        artists = get_recent_artists(
            ("t1", "t2", "t3", "t4"),
            {"t1": "a", "t2": "b", "t3": "c", "t4": "d"},
            n=3,
        )
        assert set(artists) == {"b", "c", "d"}

    def test_empty_history(self):
        assert get_recent_artists((), {}, n=3) == ()

    def test_missing_artists(self):
        artists = get_recent_artists(("t1", "t2"), {"t1": "a"}, n=3)
        assert artists == ("a",)
