"""FAISS nearest-neighbor lookup and dwell time computation."""

import logging

import numpy as np

from albart.pipeline.embedder import load_index

logger = logging.getLogger(__name__)


class TrackLookup:
    """Wraps the FAISS index and computes weighted playlists."""

    def __init__(self) -> None:
        self.index, self.track_ids = load_index()

    def query(
        self,
        embedding: np.ndarray,
        top_n: int = 10,
        softmax_k: float = 5.0,
        cycle_length_seconds: float = 60.0,
    ) -> list[dict]:
        """
        Query for the top_n nearest tracks to embedding.
        Returns a list of dicts: {track_id, distance, dwell_seconds}
        sorted by descending weight (best match first).
        """
        query = embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, top_n)
        distances = distances[0]
        indices = indices[0]

        # Filter out invalid results (FAISS returns -1 for missing)
        valid = indices >= 0
        distances = distances[valid]
        indices = indices[valid]

        dwell_times = _compute_dwell_times(distances, cycle_length_seconds, softmax_k)

        results = []
        for i, (idx, dist, dwell) in enumerate(zip(indices, distances, dwell_times)):
            results.append({
                "track_id": str(self.track_ids[idx]),
                "distance": float(dist),
                "dwell_seconds": float(dwell),
            })

        return results


def _compute_dwell_times(
    distances: np.ndarray,
    cycle_length_seconds: float = 60.0,
    k: float = 5.0,
) -> np.ndarray:
    weights = np.exp(-k * distances)
    weights /= weights.sum()
    return weights * cycle_length_seconds
