"""FAISS nearest-neighbor lookup, Vose alias table, dwell and brightness.

Dual-index rank fusion strategy
--------------------------------
The system maintains two FAISS indices built from the same set of track previews:

  raw  — previews embedded with no RMS normalization.
         Matches recordings at a similar level to the original preview.
         Best for: most tracks at natural listening volume (rrh_movo: 97% top-10).

  norm — previews embedded after RMS-normalizing to ``norm_target`` (default 0.12).
         Blurs the mel-spectrogram, more robust to room acoustics for recordings
         captured at high or inconsistent volume.
         Best for: BTS Dynamite and similar high-energy loud tracks (73% top-10).

At query time the recording is embedded BOTH ways and each embedding queries
its matching index.  Results are merged with Reciprocal Rank Fusion (RRF):
  score(track) = 1/(k + raw_rank) + 1/(k + norm_rank)
where k=60 (standard constant).  Higher score = better combined rank.

Sampling weights are geometric over final RRF rank (rank 1 = weight 1.0,
rank 2 = decay^1, …) — independent of distance scale.  Brightness and dwell
use the raw index's nearest-neighbor distance, which is calibrated to existing
config values.
"""

from __future__ import annotations

import logging

import numpy as np

from albart.pipeline.embedder import (
    FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH,
    FAISS_NORM_INDEX_PATH, FAISS_NORM_IDS_PATH,
    load_index,
)

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


def _query_index(
    index,
    track_id_list: list[str],
    embeddings: list[np.ndarray],
    top_k: int,
) -> dict[str, float]:
    """Query a FAISS index; return min-distance per track across all embeddings."""
    track_min: dict[str, float] = {}
    for emb in embeddings:
        q = emb.reshape(1, -1).astype(np.float32)
        dists, idxs = index.search(q, top_k)
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < 0:
                continue
            tid = track_id_list[idx]
            if tid not in track_min or dist < track_min[tid]:
                track_min[tid] = float(dist)
    return track_min


class DualTrackLookup:
    """
    Loads raw and norm FAISS indices; produces AliasTable via Reciprocal Rank Fusion.
    """

    def __init__(self) -> None:
        self._raw_index,  raw_ids  = load_index(FAISS_RAW_INDEX_PATH,  FAISS_RAW_IDS_PATH)
        self._norm_index, norm_ids = load_index(FAISS_NORM_INDEX_PATH, FAISS_NORM_IDS_PATH)
        self._raw_ids  = [str(t) for t in raw_ids]
        self._norm_ids = [str(t) for t in norm_ids]
        logger.info(
            "DualTrackLookup ready: raw=%d  norm=%d vectors",
            self._raw_index.ntotal, self._norm_index.ntotal,
        )

    def query(
        self,
        emb_raw: np.ndarray | list[np.ndarray],
        emb_norm: np.ndarray | list[np.ndarray],
        rank_fusion_k: int = 60,
        sampling_top_n: int = 20,
        sampling_rank_decay: float = 0.7,
        dwell_k: float = 250.0,
        dwell_floor: float = 0.003,
        brightness_k: float = 80.0,
        brightness_floor: float = 0.003,
        brightness_power: float = 2.0,
        min_dwell: float = 1.0,
        max_dwell: float = 10.0,
    ) -> AliasTable:
        """
        Query both indices, fuse via RRF, build and return an AliasTable.

        emb_raw:  (512,) array or list — raw (no-norm) query embedding(s)
        emb_norm: (512,) array or list — norm-target query embedding(s)
        """
        if isinstance(emb_raw, np.ndarray) and emb_raw.ndim == 1:
            emb_raw = [emb_raw]
        if isinstance(emb_norm, np.ndarray) and emb_norm.ndim == 1:
            emb_norm = [emb_norm]

        top_k = max(self._raw_index.ntotal, 1)

        raw_min_dist  = _query_index(self._raw_index,  self._raw_ids,  emb_raw,  top_k)
        norm_min_dist = _query_index(self._norm_index, self._norm_ids, emb_norm, top_k)

        raw_sorted  = sorted(raw_min_dist.items(),  key=lambda x: x[1])
        norm_sorted = sorted(norm_min_dist.items(), key=lambda x: x[1])

        raw_rank  = {tid: r + 1 for r, (tid, _) in enumerate(raw_sorted)}
        norm_rank = {tid: r + 1 for r, (tid, _) in enumerate(norm_sorted)}

        # Reciprocal Rank Fusion — no ties, better aggregate than single index.
        # Querying the full index (top_k = ntotal) means absent-track penalty
        # is n+1 ≈ 5192.  With k=60 that penalty term is ~0.00019, negligible,
        # so fused scores are dominated by whichever index ranks the track well.
        # A track ranked #1 in one index with any rank in the other will beat
        # a track ranked #2 in both as long as k is small relative to n.
        n = len(raw_rank)
        all_tids = set(raw_rank) | set(norm_rank)
        rrf_scores: dict[str, float] = {
            tid: (
                1.0 / (rank_fusion_k + raw_rank.get(tid,  n + 1)) +
                1.0 / (rank_fusion_k + norm_rank.get(tid, n + 1))
            )
            for tid in all_tids
        }

        top_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = top_items[:sampling_top_n]
        top_ids = [tid for tid, _ in top_items]

        # Geometric weights over final RRF rank
        weights = np.array(
            [sampling_rank_decay ** i for i in range(len(top_ids))],
            dtype=np.float64,
        )

        # d_min_raw for brightness/dwell calibration
        d_min_raw = raw_sorted[0][1] if raw_sorted else 1.0

        logger.debug(
            "DualQuery: raw_top1=%s  norm_top1=%s  fused_top1=%s  d_min_raw=%.4f",
            raw_sorted[0][0] if raw_sorted else "—",
            norm_sorted[0][0] if norm_sorted else "—",
            top_ids[0] if top_ids else "—",
            d_min_raw,
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


# ---------------------------------------------------------------------------
# Legacy single-index lookup — kept for backward compatibility / testing
# ---------------------------------------------------------------------------

class TrackLookup:
    """Single-index lookup (legacy). Use DualTrackLookup for production."""

    def __init__(self) -> None:
        from albart.pipeline.embedder import FAISS_INDEX_PATH, FAISS_IDS_PATH
        self.index, track_ids = load_index(FAISS_INDEX_PATH, FAISS_IDS_PATH)
        self._track_id_list = [str(t) for t in track_ids]
        logger.info("TrackLookup (legacy) ready: %d tracks", self.index.ntotal)

    def query(
        self,
        embeddings: np.ndarray | list[np.ndarray],
        softmax_k: float = 500.0,
        dwell_k: float = 250.0,
        dwell_floor: float = 0.003,
        brightness_k: float = 80.0,
        brightness_floor: float = 0.003,
        brightness_power: float = 2.0,
        min_dwell: float = 1.0,
        max_dwell: float = 10.0,
        sampling_top_n: int = 20,
    ) -> AliasTable:
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embeddings = [embeddings]

        track_min_dist: dict[str, float] = {}
        for emb in embeddings:
            q = emb.reshape(1, -1).astype(np.float32)
            dists, idxs = self.index.search(q, self.index.ntotal)
            for dist, idx in zip(dists[0], idxs[0]):
                if idx < 0:
                    continue
                tid = self._track_id_list[idx]
                if tid not in track_min_dist or dist < track_min_dist[tid]:
                    track_min_dist[tid] = float(dist)

        sorted_items = sorted(track_min_dist.items(), key=lambda x: x[1])
        if sampling_top_n > 0:
            sorted_items = sorted_items[:sampling_top_n]
        top_ids = [t for t, _ in sorted_items]
        d_min_raw = sorted_items[0][1] if sorted_items else 1.0
        dists_arr = np.array([d for _, d in sorted_items], dtype=np.float64)
        raw_weights = np.exp(-softmax_k * dists_arr)

        return AliasTable(
            track_ids=top_ids,
            weights=raw_weights,
            d_min_raw=d_min_raw,
            dwell_k=dwell_k,
            dwell_floor=dwell_floor,
            brightness_k=brightness_k,
            brightness_floor=brightness_floor,
            brightness_power=brightness_power,
            min_dwell=min_dwell,
            max_dwell=max_dwell,
        )
