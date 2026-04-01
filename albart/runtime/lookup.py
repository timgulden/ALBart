"""Nearest-neighbor lookup, Vose alias table, dwell and brightness.

Uses PostgreSQL + pgvector (via DatabaseClient) for all neighbor queries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from albart.effects.database import DatabaseClient

logger = logging.getLogger(__name__)


class AliasTable:
    """
    Vose's Alias Method for O(1) weighted sampling.

    Attributes:
        track_ids:   parallel array of track ID strings
        weights:     normalized probability weights (sum to 1)
        dwell_times: absolute dwell seconds per track
        brightness:  scalar [0,1] based on raw nearest-neighbor distance
    """

    def __init__(
        self,
        track_ids: list[str],
        weights: np.ndarray,
        d_min_raw: float,
        dwell_k: float,
        dwell_floor: float,
        brightness_k: float,
        brightness_floor: float,
        brightness_power: float,
        min_dwell: float,
        max_dwell: float,
    ) -> None:
        self.track_ids = track_ids
        self.d_min_raw = d_min_raw
        n = len(track_ids)

        # Normalize weights
        w = np.asarray(weights, dtype=np.float64)
        self.weights = w / w.sum()

        # Dwell time from raw d_min (confidence-keyed)
        d_eff_dwell = max(0.0, d_min_raw - dwell_floor)
        d_min_dwell = float(np.clip(
            np.exp(-dwell_k * d_eff_dwell) * max_dwell, min_dwell, max_dwell
        ))
        self.dwell_times = np.full(n, d_min_dwell)

        # Brightness from raw d_min
        d_eff = max(0.0, d_min_raw - brightness_floor)
        base = float(np.clip(np.exp(-brightness_k * d_eff), 0.0, 1.0))
        self.brightness = base ** brightness_power

        # Build Vose alias table
        prob = self.weights * n
        self._prob  = np.ones(n, dtype=np.float64)
        self._alias = np.zeros(n, dtype=np.int64)

        small, large = [], []
        for i, p in enumerate(prob):
            (small if p < 1.0 else large).append(i)

        while small and large:
            s = small.pop()
            l = large.pop()
            self._prob[s] = prob[s]
            self._alias[s] = l
            prob[l] = prob[l] + prob[s] - 1.0
            (small if prob[l] < 1.0 else large).append(l)

        for i in small + large:
            self._prob[i] = 1.0

    def sample(self) -> tuple[str, float]:
        """Draw one sample in O(1). Returns (track_id, dwell_seconds)."""
        n = len(self.track_ids)
        i = np.random.randint(0, n)
        idx = i if np.random.random() < self._prob[i] else self._alias[i]
        return self.track_ids[idx], float(self.dwell_times[idx])


class DatabaseTrackLookup:
    """pgvector-backed nearest-neighbor lookup → AliasTable."""

    def __init__(self, db: DatabaseClient) -> None:
        self._db = db
        total = db.get_total_tracks()
        logger.info("DatabaseTrackLookup ready: %d tracks with embeddings", total)

    def query(
        self,
        embedding: np.ndarray,
        sampling_top_n: int = 20,
        sampling_rank_decay: float = 0.85,
        dwell_k: float = 25.0,
        dwell_floor: float = 0.07,
        brightness_k: float = 25.0,
        brightness_floor: float = 0.07,
        brightness_power: float = 1.5,
        min_dwell: float = 1.0,
        max_dwell: float = 10.0,
    ) -> AliasTable:
        """Query pgvector for nearest neighbors, return AliasTable."""
        candidates = self._db.find_neighbors_512d(
            embedding, k=sampling_top_n,
        )

        if not candidates:
            logger.warning("No neighbors found — returning empty AliasTable")
            return AliasTable(
                track_ids=[], weights=np.array([]),
                d_min_raw=1.0,
                dwell_k=dwell_k, dwell_floor=dwell_floor,
                brightness_k=brightness_k, brightness_floor=brightness_floor,
                brightness_power=brightness_power,
                min_dwell=min_dwell, max_dwell=max_dwell,
            )

        top_ids = [c[0] for c in candidates]
        d_min_raw = candidates[0][1]

        weights = np.array(
            [sampling_rank_decay ** i for i in range(len(top_ids))],
            dtype=np.float64,
        )

        logger.debug(
            "Query: top1=%s  d_min_raw=%.4f",
            top_ids[0] if top_ids else "—", d_min_raw,
        )

        return AliasTable(
            track_ids=top_ids,
            weights=weights,
            d_min_raw=d_min_raw,
            dwell_k=dwell_k,
            dwell_floor=dwell_floor,
            brightness_k=brightness_k,
            brightness_floor=brightness_floor,
            brightness_power=brightness_power,
            min_dwell=min_dwell,
            max_dwell=max_dwell,
        )
