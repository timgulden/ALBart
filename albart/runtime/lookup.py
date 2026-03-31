"""FAISS nearest-neighbor lookup, Vose alias table, dwell and brightness.

Single-index strategy: queries the norm FAISS index (RMS-normalized to 0.12)
and ranks tracks by L2 distance.  Sampling weights are geometric over rank.
Brightness and dwell are calibrated from the nearest-neighbor distance.
"""

from __future__ import annotations

import logging

import numpy as np

# Lazy import — the listener runtime still uses FAISS for the LED display.
# This will be migrated to DatabaseClient in a future pass.
def _get_faiss_paths():
    from albart.utils import DATA_DIR
    return DATA_DIR / "faiss_norm.index", DATA_DIR / "faiss_norm_ids.npy"

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


class TrackLookup:
    """Single raw FAISS index lookup → AliasTable."""

    def __init__(self) -> None:
        import faiss
        idx_path, ids_path = _get_faiss_paths()
        self._index = faiss.read_index(str(idx_path))
        ids = np.load(str(ids_path), allow_pickle=True)
        self._id_list = [str(t) for t in ids]
        logger.info("TrackLookup ready: %d tracks", self._index.ntotal)

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
        """Query raw FAISS index, return AliasTable sorted by L2 distance."""
        q = embedding.reshape(1, -1).astype(np.float32)
        dists, idxs = self._index.search(q, sampling_top_n)

        top_ids = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx >= 0:
                top_ids.append(self._id_list[idx])

        d_min_raw = float(dists[0][0]) if len(dists[0]) > 0 else 1.0

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
