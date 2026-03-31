"""UDP broadcast to the MapView display.

Sends track embeddings (with noise to avoid exact-match void) so the
map display can show the current position in embedding space.
"""

from __future__ import annotations

import logging
import pickle
import socket
from dataclasses import dataclass, field

import numpy as np

from albart.effects.database import DatabaseClient

logger = logging.getLogger(__name__)


@dataclass
class BroadcastClient:
    """Sends track embedding data to the MapView via UDP."""
    port: int = 57001
    _sock: socket.socket = field(default_factory=lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM), repr=False)
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    _cached_tid: str | None = field(default=None, repr=False)
    _cached_emb: np.ndarray | None = field(default=None, repr=False)

    def broadcast_track(self, track_id: str, db: DatabaseClient) -> None:
        """Send this track's embedding (with noise) to the map display."""
        if track_id != self._cached_tid:
            emb = db.get_embedding_512(track_id)
            if emb is None:
                logger.debug("No embedding for %s — skip broadcast", track_id)
                return
            noisy = emb.astype(np.float64) + self._rng.normal(size=512) * 0.04
            self._cached_emb = (noisy / (np.linalg.norm(noisy) + 1e-8)).astype(np.float32)
            self._cached_tid = track_id

        try:
            data = pickle.dumps({
                "raw": self._cached_emb,
                "top1": track_id,
                "d_min_raw": 0.05,
            })
            self._sock.sendto(data, ("127.0.0.1", self.port))
        except Exception as e:
            logger.warning("Map broadcast failed: %s", e)

    def broadcast_raw(self, embedding: np.ndarray, track_id: str) -> None:
        """Send a raw embedding (e.g. live audio) to the map display."""
        try:
            data = pickle.dumps({
                "raw": embedding,
                "top1": track_id,
                "d_min_raw": 0.05,
            })
            self._sock.sendto(data, ("127.0.0.1", self.port))
        except Exception as e:
            logger.warning("Map broadcast failed: %s", e)
