"""Automatic Gain Control for the macOS runtime mic input.

Periodically measures RMS from the audio buffer and adjusts the macOS
system input volume via osascript to keep the captured level near a target.

Only active on macOS (darwin). On other platforms, start() is a no-op.

Tuning:
    target_rms: RMS level that produces good embeddings.  Should match
        the level used when building the FAISS index (~0.20 for room
        recordings through the PDP/Movo mic).
    interval_seconds: how often to check and adjust.  10s is a reasonable
        default — fast enough to adapt to track changes, slow enough to
        avoid oscillation.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# Discrete levels available on macOS (same as recording tool)
_LEVELS = [20, 25, 32, 40, 50, 63, 79, 100]

# RMS tolerance band — only adjust if outside [target * lo, target * hi]
_LO_FACTOR = 0.65
_HI_FACTOR = 1.60


def _get_input_volume() -> int | None:
    try:
        result = subprocess.run(
            ["osascript", "-e", "input volume of (get volume settings)"],
            capture_output=True, text=True, timeout=2,
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def _set_input_volume(level: int) -> None:
    subprocess.run(
        ["osascript", "-e", f"set volume input volume {level}"],
        capture_output=True, timeout=2,
    )


def _nearest_level(current: int) -> int:
    """Return the _LEVELS entry closest to current."""
    return min(_LEVELS, key=lambda v: abs(v - current))


class AGCWorker:
    """
    Background thread that periodically measures audio buffer RMS and
    adjusts macOS input volume to keep it near target_rms.
    """

    def __init__(
        self,
        audio_buffer,
        target_rms: float = 0.20,
        interval_seconds: float = 10.0,
    ) -> None:
        self._buffer = audio_buffer
        self.target_rms = target_rms
        self.interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = sys.platform == "darwin"
        if not self._enabled:
            logger.info("AGC: not on macOS — disabled")

    def start(self) -> None:
        if not self._enabled:
            return
        vol = _get_input_volume()
        if vol is None:
            logger.warning("AGC: cannot read input volume — disabled")
            self._enabled = False
            return
        logger.info("AGC started (target_rms=%.2f  current_volume=%d)", self.target_rms, vol)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AGCWorker"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.wait(timeout=self.interval):
            try:
                self._tick()
            except Exception as e:
                logger.warning("AGC tick error: %s", e)

    def _tick(self) -> None:
        audio = self._buffer.read()
        if len(audio) == 0:
            return
        rms = float(np.sqrt(np.mean(audio ** 2)))

        vol = _get_input_volume()
        if vol is None:
            return

        lo = self.target_rms * _LO_FACTOR
        hi = self.target_rms * _HI_FACTOR

        if lo <= rms <= hi:
            logger.debug("AGC: rms=%.3f  volume=%d  → OK", rms, vol)
            return

        current_idx = _LEVELS.index(_nearest_level(vol))

        if rms < lo:
            # Too quiet — step up one level
            if current_idx < len(_LEVELS) - 1:
                new_vol = _LEVELS[current_idx + 1]
                _set_input_volume(new_vol)
                logger.info("AGC: rms=%.3f < %.3f  volume %d → %d ↑", rms, lo, vol, new_vol)
            else:
                logger.debug("AGC: rms=%.3f low but already at max volume (%d)", rms, vol)
        else:
            # Too loud — step down one level
            if current_idx > 0:
                new_vol = _LEVELS[current_idx - 1]
                _set_input_volume(new_vol)
                logger.info("AGC: rms=%.3f > %.3f  volume %d → %d ↓", rms, hi, vol, new_vol)
            else:
                logger.debug("AGC: rms=%.3f high but already at min volume (%d)", rms, vol)
