"""Main runtime display loop with alias sampling, brightness fading."""

from __future__ import annotations

import logging
import queue
import time

import numpy as np

from albart.pipeline.preprocess import load_art_32
from albart.runtime.display import DisplayBackend
from albart.runtime.lookup import AliasTable, TrackLookup

logger = logging.getLogger(__name__)


def crossfade(frame_a: np.ndarray, frame_b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear pixel interpolation. alpha=0 → frame_a, alpha=1 → frame_b."""
    return (frame_a * (1.0 - alpha) + frame_b * alpha).astype(np.uint8)


def apply_brightness(frame: np.ndarray, brightness: float) -> np.ndarray:
    """Scale RGB values by brightness [0, 1]."""
    return (frame * brightness).astype(np.uint8)


class DisplayLoop:
    """
    Main display loop — runs on the main thread at display_fps.

    Sampling:
      - Alias table built from full FAISS query (all tracks, weighted by distance)
      - Each cover is sampled with replacement; same cover may follow itself
      - Dwell time per sample is absolute (exp(-dwell_k * d) * max_dwell, clamped)

    Brightness:
      - Driven by nearest-neighbor distance from latest embedding
      - Fades smoothly to new target over brightness_fade_seconds
      - Applied to every rendered frame (including during crossfades)

    Alias table updates:
      - New table computed as soon as a new embedding arrives
      - Held as _pending_table, swapped in at next cover boundary
      - Never interrupts an in-progress crossfade
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
        self.softmax_k = rt["softmax_k"]
        self.dwell_k = rt["dwell_k"]
        self.brightness_k = rt["brightness_k"]
        self.min_dwell = rt["min_dwell_seconds"]
        self.max_dwell = rt["max_dwell_seconds"]
        self.crossfade_seconds = rt["crossfade_seconds"]
        self.brightness_fade_seconds = rt["brightness_fade_seconds"]

        # Display state
        self._current_frame: np.ndarray | None = None
        self._next_frame: np.ndarray | None = None
        self._dwell_remaining: float = 0.0
        self._in_crossfade: bool = False
        self._crossfade_elapsed: float = 0.0

        # Brightness state — linear fade from _brightness_start to _brightness_target
        self._brightness: float = 0.0
        self._brightness_start: float = 0.0
        self._brightness_target: float = 0.0
        self._brightness_fade_elapsed: float = 0.0
        self._brightness_fading: bool = False

        # Alias tables
        self._table: AliasTable | None = None
        self._pending_table: AliasTable | None = None

        # Art cache: track_id → (32,32,3) uint8 array
        self._art_cache: dict[str, np.ndarray] = {}

    # ── Public ────────────────────────────────────────────────────────────

    def run(self) -> None:
        frame_duration = 1.0 / self.fps
        logger.info("Display loop started at %d fps", self.fps)

        while True:
            t0 = time.monotonic()
            self._check_embedding_queue()
            self._tick(frame_duration)
            sleep = frame_duration - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    # ── Embedding / table updates ─────────────────────────────────────────

    def _check_embedding_queue(self) -> None:
        """Drain queue, build alias table from latest embedding."""
        latest = None
        try:
            while True:
                latest = self.embedding_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is None:
            return

        table = self.lookup.query(
            latest,
            softmax_k=self.softmax_k,
            dwell_k=self.dwell_k,
            brightness_k=self.brightness_k,
            min_dwell=self.min_dwell,
            max_dwell=self.max_dwell,
        )

        # Start linear brightness fade toward new target immediately
        self._brightness_start = self._brightness
        self._brightness_target = table.brightness
        self._brightness_fade_elapsed = 0.0
        self._brightness_fading = True
        logger.debug("New table: brightness_target=%.2f", table.brightness)

        if self._in_crossfade:
            # Don't interrupt crossfade — hold for next boundary
            self._pending_table = table
        elif self._table is None:
            # First table — bootstrap display
            self._table = table
            self._start_next_cover()
        else:
            self._pending_table = table

    # ── Per-frame tick ────────────────────────────────────────────────────

    def _tick(self, dt: float) -> None:
        self._update_brightness(dt)

        if self._current_frame is None:
            return

        if self._in_crossfade:
            self._tick_crossfade(dt)
        else:
            self._tick_dwell(dt)

    def _update_brightness(self, dt: float) -> None:
        if not self._brightness_fading:
            return
        self._brightness_fade_elapsed += dt
        alpha = min(self._brightness_fade_elapsed / self.brightness_fade_seconds, 1.0)
        self._brightness = self._brightness_start + alpha * (
            self._brightness_target - self._brightness_start
        )
        if alpha >= 1.0:
            self._brightness = self._brightness_target
            self._brightness_fading = False

    def _tick_crossfade(self, dt: float) -> None:
        self._crossfade_elapsed += dt
        alpha = min(self._crossfade_elapsed / self.crossfade_seconds, 1.0)
        frame = crossfade(self._current_frame, self._next_frame, alpha)
        self.display.show_frame(apply_brightness(frame, self._brightness))

        if alpha >= 1.0:
            self._current_frame = self._next_frame
            self._next_frame = None
            self._in_crossfade = False
            self._crossfade_elapsed = 0.0

    def _tick_dwell(self, dt: float) -> None:
        self.display.show_frame(apply_brightness(self._current_frame, self._brightness))
        self._dwell_remaining -= dt
        if self._dwell_remaining <= 0.0 and self._table is not None:
            self._start_next_cover()

    # ── Cover transitions ─────────────────────────────────────────────────

    def _start_next_cover(self) -> None:
        """Sample the next cover and begin crossfade (or bootstrap first frame)."""
        if self._table is None:
            return

        # Swap in pending table at cover boundary (before sampling)
        if self._pending_table is not None:
            self._table = self._pending_table
            self._pending_table = None

        track_id, dwell = self._table.sample()
        self._dwell_remaining = dwell
        next_frame = self._load_frame(track_id)

        if self._current_frame is None:
            # Bootstrap: show first frame immediately, no crossfade
            self._current_frame = next_frame
            logger.debug("Bootstrap: first cover %s (dwell=%.1fs)", track_id, dwell)
        else:
            self._next_frame = next_frame
            self._in_crossfade = True
            self._crossfade_elapsed = 0.0
            logger.debug("Next cover: %s (dwell=%.1fs)", track_id, dwell)

    # ── Art loading ───────────────────────────────────────────────────────

    def _load_frame(self, track_id: str) -> np.ndarray:
        if track_id in self._art_cache:
            return self._art_cache[track_id]

        from albart.pipeline.database import DB_PATH, get_connection
        conn = get_connection(DB_PATH)
        row = conn.execute(
            "SELECT art_path_32 FROM tracks WHERE track_id = ?", (track_id,)
        ).fetchone()
        conn.close()

        if row and row["art_path_32"]:
            try:
                frame = load_art_32(row["art_path_32"])
                self._art_cache[track_id] = frame
                return frame
            except Exception as e:
                logger.error("Failed to load art for %s: %s", track_id, e)

        blank = np.zeros((32, 32, 3), dtype=np.uint8)
        self._art_cache[track_id] = blank
        return blank
