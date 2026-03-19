"""Startup animation: white ripples emanating from center of the 32x32 display."""

from __future__ import annotations

import threading
import time

import numpy as np

from albart.runtime.display import DisplayBackend

# Precompute distance and envelope grids (constant across all frames)
_y, _x = np.mgrid[0:32, 0:32]
_R = np.sqrt((_x - 15.5) ** 2 + (_y - 15.5) ** 2).astype(np.float32)
_ENVELOPE = np.exp(-_R * 0.12).astype(np.float32)  # falloff from center

RIPPLE_FREQ  = 0.75   # spatial: rings per pixel
RIPPLE_SPEED = 2.5    # temporal: radians per second
FADE_SECONDS = 0.75   # fade-to-black duration after ready signal


def _ripple_frame(t: float, brightness_scale: float = 1.0) -> np.ndarray:
    """Return a (32, 32, 3) uint8 ripple frame at time t."""
    wave = np.cos(_R * RIPPLE_FREQ - t * RIPPLE_SPEED)
    intensity = np.clip(wave * _ENVELOPE, 0.0, 1.0) * brightness_scale
    rgb = (intensity[:, :, np.newaxis] * 255).astype(np.uint8)
    # Tint slightly cool (blue-white) to feel like a waking LED panel
    rgb[:, :, 2] = np.clip(rgb[:, :, 2].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    return rgb


def run_startup(
    display: DisplayBackend,
    ready_event: threading.Event,
    fps: int = 30,
) -> None:
    """
    Animate ripples on `display` until `ready_event` is set,
    then fade to black over FADE_SECONDS.
    Blocks until the fade-out completes.
    """
    frame_duration = 1.0 / fps
    t_start = time.monotonic()

    # --- Ripple loop ---
    while not ready_event.is_set():
        t0 = time.monotonic()
        t = t0 - t_start
        display.show_frame(_ripple_frame(t))
        sleep = frame_duration - (time.monotonic() - t0)
        if sleep > 0:
            time.sleep(sleep)

    # --- Fade to black ---
    fade_start = time.monotonic()
    while True:
        t0 = time.monotonic()
        elapsed = t0 - fade_start
        if elapsed >= FADE_SECONDS:
            display.show_frame(np.zeros((32, 32, 3), dtype=np.uint8))
            break
        brightness_scale = 1.0 - (elapsed / FADE_SECONDS)
        t = t0 - t_start
        display.show_frame(_ripple_frame(t, brightness_scale))
        sleep = frame_duration - (time.monotonic() - t0)
        if sleep > 0:
            time.sleep(sleep)
