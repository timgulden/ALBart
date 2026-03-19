"""FAISS nearest-neighbor lookup, Vose alias table, dwell and brightness."""

from __future__ import annotations

import logging

import numpy as np

from albart.pipeline.embedder import load_index

logger = logging.getLogger(__name__)


class AliasTable:
    """
    Vose's Alias Method for O(1) weighted sampling.

    Attributes:
        track_ids:   parallel array of track ID strings
        weights:     normalized probability weights (sum to 1)
        dwell_times: absolute dwell seconds per track (unclamped raw softmax * max_dwell)
        brightness:  scalar [0,1] based on nearest-neighbor distance
    """

    def __init__(
        self,
        track_ids: list[str],
        distances: np.ndarray,
        softmax_k: float,
        dwell_k: float,
        brightness_k: float,
        min_dwell: float,
        max_dwell: float,
    ) -> None:
        self.track_ids = track_ids
        n = len(track_ids)

        # --- Sampling weights (normalized) ---
        raw_weights = np.exp(-softmax_k * distances)
        self.weights = raw_weights / raw_weights.sum()

        # --- Dwell times (absolute, not normalized) ---
        raw_dwell = np.exp(-dwell_k * distances)
        self.dwell_times = np.clip(raw_dwell * max_dwell, min_dwell, max_dwell)

        # --- Brightness from nearest neighbor distance ---
        d_min = float(distances[0])  # distances are sorted ascending from FAISS
        self.brightness = float(np.clip(np.exp(-brightness_k * d_min), 0.0, 1.0))

        # --- Build Vose alias table ---
        prob = self.weights * n
        self._prob = np.ones(n, dtype=np.float64)
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

        # Numerical clean-up
        for i in small + large:
            self._prob[i] = 1.0

    def sample(self) -> tuple[str, float]:
        """
        Draw one sample in O(1).
        Returns (track_id, dwell_seconds).
        """
        n = len(self.track_ids)
        i = np.random.randint(0, n)
        if np.random.random() < self._prob[i]:
            idx = i
        else:
            idx = self._alias[i]
        return self.track_ids[idx], float(self.dwell_times[idx])


class TrackLookup:
    """Wraps the FAISS index and builds AliasTable instances from embeddings."""

    def __init__(self) -> None:
        self.index, self._track_ids = load_index()
        self._track_id_list = [str(t) for t in self._track_ids]
        logger.info("TrackLookup ready: %d tracks", self.index.ntotal)

    def query(
        self,
        embedding: np.ndarray,
        softmax_k: float = 5.0,
        dwell_k: float = 3.0,
        brightness_k: float = 2.0,
        min_dwell: float = 1.0,
        max_dwell: float = 10.0,
    ) -> AliasTable:
        """
        Query all tracks, return a fully built AliasTable.
        Distances are sorted ascending (nearest first) by FAISS.
        """
        n = self.index.ntotal
        query = embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, n)
        distances = distances[0]
        indices = indices[0]

        # Filter FAISS sentinel values
        valid = indices >= 0
        distances = distances[valid]
        indices = indices[valid]

        track_ids = [self._track_id_list[i] for i in indices]

        logger.debug(
            "Query: n=%d  d_min=%.3f  d_max=%.3f",
            len(track_ids), distances[0], distances[-1],
        )

        return AliasTable(
            track_ids=track_ids,
            distances=distances,
            softmax_k=softmax_k,
            dwell_k=dwell_k,
            brightness_k=brightness_k,
            min_dwell=min_dwell,
            max_dwell=max_dwell,
        )
