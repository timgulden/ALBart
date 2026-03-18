"""Main runtime display loop."""

import logging
import queue
import time

import numpy as np

from albart.pipeline.preprocess import load_art_32
from albart.runtime.display import DisplayBackend
from albart.runtime.lookup import TrackLookup

logger = logging.getLogger(__name__)


def crossfade(frame_a: np.ndarray, frame_b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear pixel interpolation. alpha=0 → frame_a, alpha=1 → frame_b."""
    return (frame_a * (1.0 - alpha) + frame_b * alpha).astype(np.uint8)


class DisplayLoop:
    """
    Drives the display on the main thread at a fixed FPS.
    Reads new embeddings from result_queue, looks up playlists,
    and crossfades between album art covers.
    """

    def __init__(
        self,
        display: DisplayBackend,
        lookup: TrackLookup,
        embedding_queue: queue.Queue,
        config: dict,
    ) -> None:
        self.display = display
        self.lookup = lookup
        self.embedding_queue = embedding_queue

        rt = config["runtime"]
        self.fps = rt["display_fps"]
        self.top_n = rt["top_n_neighbors"]
        self.softmax_k = rt["softmax_k"]
        self.cycle_length = rt["cycle_length_seconds"]
        self.crossfade_seconds = rt["crossfade_seconds"]

        self._playlist: list[dict] = []
        self._playlist_index = 0
        self._current_frame: np.ndarray | None = None
        self._next_frame: np.ndarray | None = None
        self._dwell_remaining: float = 0.0
        self._in_crossfade: bool = False
        self._crossfade_elapsed: float = 0.0

    def run(self) -> None:
        frame_duration = 1.0 / self.fps
        logger.info("Display loop started at %d fps", self.fps)

        while True:
            t0 = time.monotonic()

            self._check_for_new_embedding()
            self._tick(frame_duration)

            elapsed = time.monotonic() - t0
            sleep = frame_duration - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _check_for_new_embedding(self) -> None:
        """Drain the embedding queue, keeping only the latest result."""
        latest = None
        try:
            while True:
                latest = self.embedding_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None and not self._in_crossfade:
            self._update_playlist(latest)

    def _update_playlist(self, embedding: np.ndarray) -> None:
        results = self.lookup.query(
            embedding,
            top_n=self.top_n,
            softmax_k=self.softmax_k,
            cycle_length_seconds=self.cycle_length,
        )
        if not results:
            logger.warning("Lookup returned no results")
            return

        self._playlist = results
        self._playlist_index = 0
        self._dwell_remaining = results[0]["dwell_seconds"]
        logger.debug(
            "Playlist updated: best match %s (dwell=%.1fs)",
            results[0]["track_id"],
            results[0]["dwell_seconds"],
        )

        # Load first frame if we have nothing yet
        if self._current_frame is None:
            self._current_frame = self._load_frame(results[0]["track_id"])

    def _tick(self, dt: float) -> None:
        if self._current_frame is None:
            return  # Nothing to show yet

        if self._in_crossfade:
            self._crossfade_elapsed += dt
            alpha = min(self._crossfade_elapsed / self.crossfade_seconds, 1.0)
            frame = crossfade(self._current_frame, self._next_frame, alpha)
            self.display.show_frame(frame)
            if alpha >= 1.0:
                self._current_frame = self._next_frame
                self._next_frame = None
                self._in_crossfade = False
                self._crossfade_elapsed = 0.0
            return

        # Show current frame
        self.display.show_frame(self._current_frame)

        # Count down dwell time
        self._dwell_remaining -= dt
        if self._dwell_remaining <= 0.0 and self._playlist:
            self._advance_playlist()

    def _advance_playlist(self) -> None:
        if not self._playlist:
            return
        self._playlist_index = (self._playlist_index + 1) % len(self._playlist)
        next_entry = self._playlist[self._playlist_index]
        self._dwell_remaining = next_entry["dwell_seconds"]
        self._next_frame = self._load_frame(next_entry["track_id"])
        self._in_crossfade = True
        self._crossfade_elapsed = 0.0

    def _load_frame(self, track_id: str) -> np.ndarray:
        from albart.pipeline.database import DB_PATH, get_connection
        conn = get_connection(DB_PATH)
        row = conn.execute(
            "SELECT art_path_32 FROM tracks WHERE track_id = ?", (track_id,)
        ).fetchone()
        conn.close()
        if row and row["art_path_32"]:
            try:
                return load_art_32(row["art_path_32"])
            except Exception as e:
                logger.error("Failed to load art for %s: %s", track_id, e)
        # Fallback: black frame
        return np.zeros((32, 32, 3), dtype=np.uint8)
