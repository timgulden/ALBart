"""UMAP music map visualization.

Renders all tracks as micro-thumbnails at their 2D UMAP positions on a dark
canvas.  A grey dot traces the live audio embedding (via UMAP transform).  The
closest matching album cover is magnified near the dot and labeled.  A long
fading trail records the recent path through music space.

Designed to run as a separate process, receiving DualEmbedding payloads over
a local UDP socket from the main ALBart runtime.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pygame
from PIL import Image

from albart.pipeline.database import DB_PATH, get_connection
from albart.utils import DATA_DIR

logger = logging.getLogger(__name__)

UMAP_2D_PATH    = DATA_DIR / "umap_2d.npy"
UMAP_IDS_PATH   = DATA_DIR / "umap_ids.npy"
UMAP_MODEL_PATH = DATA_DIR / "umap_model.joblib"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class TrailPoint:
    x: float           # screen x
    y: float           # screen y
    confidence: float  # [0, 1] at time of capture
    timestamp: float   # monotonic


# ── UMAP transform thread ─────────────────────────────────────────────────────

class UMAPTransformer:
    """
    Wraps umap model.transform() in a background thread so it never blocks
    the render loop.  Always returns the most recently completed result.
    """

    def __init__(self, model, xy_min: np.ndarray, xy_range: np.ndarray) -> None:
        self._model     = model
        self._xy_min    = xy_min
        self._xy_range  = xy_range
        self._result:  Optional[tuple[float, float]] = None
        self._pending: Optional[np.ndarray] = None
        self._lock  = threading.Lock()
        self._event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="UMAPTransformer"
        )
        self._thread.start()

    def submit(self, emb: np.ndarray) -> None:
        """Queue a new embedding for transformation (replaces any pending work)."""
        with self._lock:
            self._pending = emb.copy()
        self._event.set()

    def get_normalized(self) -> Optional[tuple[float, float]]:
        """Return latest (nx, ny) in [0, 1]², or None before first result."""
        with self._lock:
            return self._result

    def _run(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()
            with self._lock:
                emb = self._pending
                self._pending = None
            if emb is None:
                continue
            try:
                t0 = time.monotonic()
                xy = self._model.transform(emb.reshape(1, -1))[0]  # (2,)
                xy_norm = (xy - self._xy_min) / (self._xy_range + 1e-8)
                with self._lock:
                    self._result = (float(xy_norm[0]), float(xy_norm[1]))
                logger.debug(
                    "UMAP transform %.0fms → (%.3f, %.3f)",
                    (time.monotonic() - t0) * 1000, xy_norm[0], xy_norm[1],
                )
            except Exception as e:
                logger.error("UMAP transform error: %s", e)


# ── Main display class ────────────────────────────────────────────────────────

class MapDisplay:
    """
    UMAP music map visualization.

    Call update(emb_raw, emb_norm) each time a new DualEmbedding arrives.
    Call render(dt) at display_fps to advance state and draw a frame.
    """

    def __init__(
        self,
        canvas_size:         tuple[int, int],
        thumb_size:          int,
        trail_max_seconds:   float,
        brightness_k:        float,
        brightness_floor:    float,
        brightness_power:    float,
    ) -> None:
        self._canvas_w, self._canvas_h = canvas_size
        self._thumb_size         = thumb_size
        self._trail_max_seconds  = trail_max_seconds
        self._brightness_k       = brightness_k
        self._brightness_floor   = brightness_floor
        self._brightness_power   = brightness_power

        # Live state
        self._dot_pos:       Optional[tuple[float, float]] = None  # screen coords
        self._top1_track_id: Optional[str]  = None
        self._confidence:    float          = 0.0
        self._trail: collections.deque[TrailPoint] = collections.deque()

        # Magnified cover state
        self._mag_track_id:    Optional[str]           = None
        self._mag_orig_image:  Optional[Image.Image]   = None
        self._mag_surface:     Optional[pygame.Surface] = None
        self._mag_label:       str = ""
        self._mag_size:        float = 0.0   # current displayed size (lerped)
        self._mag_last_px:     int   = 0     # size at last surface scale

        # (No FAISS index needed — top-1 and d_min_raw arrive in the UDP payload,
        #  already computed by the main process using RRF dual-index fusion.)

        # UMAP model + coordinates
        if not UMAP_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"UMAP model not found at {UMAP_MODEL_PATH}. "
                "Run: python tools/build_umap.py"
            )
        logger.info("Loading UMAP model...")
        umap_model = joblib.load(str(UMAP_MODEL_PATH))
        umap_2d    = np.load(str(UMAP_2D_PATH))                           # (N, 2)
        umap_ids   = np.load(str(UMAP_IDS_PATH), allow_pickle=True)       # (N,)

        xy_min   = umap_2d.min(axis=0)
        xy_range = umap_2d.max(axis=0) - xy_min
        umap_2d_norm = (umap_2d - xy_min) / (xy_range + 1e-8)            # → [0, 1]²

        # Layout margins: keep thumbnails fully on-canvas
        self._margin = thumb_size + 8

        self._id_to_screen: dict[str, tuple[int, int]] = {}
        for i, tid in enumerate(umap_ids):
            nx = float(umap_2d_norm[i, 0])
            ny = float(umap_2d_norm[i, 1])
            self._id_to_screen[str(tid)] = self._norm_to_screen(nx, ny)

        self._transformer = UMAPTransformer(umap_model, xy_min, xy_range)

        # Track metadata + art (loaded from DB)
        conn = get_connection(DB_PATH)
        rows = conn.execute(
            "SELECT track_id, title, artist, art_path_32, art_path_original FROM tracks"
        ).fetchall()
        conn.close()
        self._db_rows: dict = {row["track_id"]: row for row in rows}

        # pygame setup
        pygame.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode(
            (self._canvas_w, self._canvas_h), pygame.NOFRAME
        )
        pygame.display.set_caption("ALBart — Music Map")
        font_size = max(14, self._canvas_w // 200)
        self._font = pygame.font.SysFont("Arial", font_size)

        # Derived size constants
        self._dot_radius   = max(3, self._canvas_w // 640)
        self._trail_radius = max(2, self._canvas_w // 1000)
        self._mag_max_size = self._canvas_w // 13   # ~295px at 4K, ~148px at 1080p

        # Pre-render all thumbnails to background surface (done once at startup)
        logger.info("Pre-rendering background (%d thumbnails)...", len(umap_ids))
        self._bg_surface = self._build_background(umap_ids)
        logger.info(
            "MapDisplay ready — canvas=%dx%d  thumbs=%dpx  trail=%.0fs",
            self._canvas_w, self._canvas_h, thumb_size, trail_max_seconds,
        )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _norm_to_screen(self, nx: float, ny: float) -> tuple[int, int]:
        """Map normalized [0, 1] UMAP coords to screen pixel position."""
        m = self._margin
        x = int(m + nx * (self._canvas_w - 2 * m))
        y = int(m + ny * (self._canvas_h - 2 * m))
        return x, y

    # ── Startup ───────────────────────────────────────────────────────────────

    def _load_thumb_surface(self, track_id: str) -> Optional[pygame.Surface]:
        """Load art_path_32, scale to thumb_size, return pygame Surface."""
        row = self._db_rows.get(track_id)
        if row is None or not row["art_path_32"]:
            return None
        try:
            img = Image.open(DATA_DIR / row["art_path_32"]).convert("RGB")
            if img.size != (self._thumb_size, self._thumb_size):
                img = img.resize(
                    (self._thumb_size, self._thumb_size), Image.LANCZOS
                )
            arr = np.array(img, dtype=np.uint8)
            return pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        except Exception as e:
            logger.debug("Thumb load failed for %s: %s", track_id, e)
            return None

    def _build_background(self, umap_ids: np.ndarray) -> pygame.Surface:
        """Pre-blit all thumbnails onto a black surface at their UMAP positions."""
        surface = pygame.Surface((self._canvas_w, self._canvas_h))
        surface.fill((0, 0, 0))
        half = self._thumb_size // 2
        missing = 0
        for tid in umap_ids:
            tid = str(tid)
            pos = self._id_to_screen.get(tid)
            if pos is None:
                continue
            surf = self._load_thumb_surface(tid)
            if surf is not None:
                surface.blit(surf, (pos[0] - half, pos[1] - half))
            else:
                missing += 1
        if missing:
            logger.warning("Missing thumbnails: %d / %d", missing, len(umap_ids))
        return surface

    # ── Runtime update ────────────────────────────────────────────────────────

    def update(
        self,
        emb_raw: np.ndarray,
        emb_norm: np.ndarray,  # noqa: ARG002
        top1_track_id: str,
        d_min_raw: float,
    ) -> None:
        """
        Called when a new embedding arrives from the main process (~1s interval).
        top1_track_id and d_min_raw come from the main process's RRF dual-index
        query, so they match exactly what the LED display is showing.
        """
        self._top1_track_id = top1_track_id

        # Confidence from raw nearest-neighbor distance (same formula as main runtime)
        d_eff = max(0.0, d_min_raw - self._brightness_floor)
        base  = float(np.exp(-self._brightness_k * d_eff))
        self._confidence = base ** self._brightness_power

        # Submit raw embedding to background UMAP transformer for dot position
        self._transformer.submit(emb_raw)

    # ── Per-frame tick ────────────────────────────────────────────────────────

    def _tick(self, dt: float) -> None:
        """Advance animated state: dot position, trail, magnified cover size."""
        # Update dot from latest UMAP result
        result = self._transformer.get_normalized()
        if result is not None:
            nx = max(0.0, min(1.0, result[0]))
            ny = max(0.0, min(1.0, result[1]))
            new_pos = self._norm_to_screen(nx, ny)

            if self._dot_pos is not None and new_pos != self._dot_pos:
                # Record current position before moving
                px, py = self._dot_pos
                self._trail.append(TrailPoint(
                    x=float(px), y=float(py),
                    confidence=self._confidence,
                    timestamp=time.monotonic(),
                ))
            self._dot_pos = new_pos

        # Prune expired trail points
        cutoff = time.monotonic() - self._trail_max_seconds
        while self._trail and self._trail[0].timestamp < cutoff:
            self._trail.popleft()

        # The magnified cover is the RRF top-1 from the main process — the track
        # most similar to the live audio in full 512-dim embedding space.
        # It is drawn at that track's own UMAP position on the map, not near the
        # dot, so the gap (or lack thereof) between dot and cover is itself
        # informative: small gap = confident match, large gap = uncertain.
        if self._top1_track_id != self._mag_track_id:
            self._mag_track_id = self._top1_track_id
            self._mag_surface  = None
            self._mag_last_px  = 0
            if self._top1_track_id:
                row = self._db_rows.get(self._top1_track_id)
                if row:
                    title  = row["title"]  or ""
                    artist = row["artist"] or ""
                    self._mag_label = f"{title} — {artist}" if title else ""
                    if row["art_path_original"]:
                        try:
                            self._mag_orig_image = Image.open(
                                DATA_DIR / row["art_path_original"]
                            ).convert("RGB")
                        except Exception:
                            self._mag_orig_image = None
                    else:
                        self._mag_orig_image = None

        # Smooth magnified cover size toward target.
        # Size scales with confidence, but a visible minimum ensures the nearest
        # track is always shown even when the match is uncertain (room-mic conditions).
        min_px    = max(self._mag_max_size // 5, 40)
        target_px = max(min_px, int(self._mag_max_size * self._confidence))
        diff = target_px - self._mag_size
        self._mag_size += diff * min(1.0, dt * 3.0)  # ~3× per second lerp

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, dt: float) -> None:
        """Draw one frame. Returns False if the window was closed."""
        # Event pump
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise SystemExit(0)

        self._tick(dt)

        # 1 — Background (all thumbnails, pre-rendered)
        self._screen.blit(self._bg_surface, (0, 0))

        # 2 — Trail (oldest → newest, so newest draws on top)
        now = time.monotonic()
        for pt in self._trail:
            age = now - pt.timestamp
            if age >= self._trail_max_seconds:
                continue
            time_factor = 1.0 - age / self._trail_max_seconds
            v = max(0, min(255, int(pt.confidence * time_factor * 255)))
            pygame.draw.circle(
                self._screen, (v, v, v),
                (int(pt.x), int(pt.y)), self._trail_radius,
            )

        # 3 — Grey query dot
        if self._dot_pos is not None:
            dx, dy = int(self._dot_pos[0]), int(self._dot_pos[1])
            pygame.draw.circle(
                self._screen, (120, 120, 120), (dx, dy), self._dot_radius + 2
            )
            pygame.draw.circle(
                self._screen, (220, 220, 220), (dx, dy), self._dot_radius
            )

        # 4 — Magnified cover + label at the track's own UMAP map position.
        # The cover grows from the thumbnail's location, so the small tile and
        # the large cover are always co-located.  The grey dot is independent.
        if self._mag_track_id and self._mag_orig_image is not None:
            size = max(8, int(self._mag_size))
            if size != self._mag_last_px:
                scaled = self._mag_orig_image.resize((size, size), Image.LANCZOS)
                arr = np.array(scaled, dtype=np.uint8)
                self._mag_surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
                self._mag_last_px = size

            if self._mag_surface is not None:
                # Center the cover on the track's fixed UMAP screen position
                tx, ty = self._id_to_screen.get(self._mag_track_id, (self._canvas_w // 2, self._canvas_h // 2))
                cx = max(0, min(self._canvas_w - size, tx - size // 2))
                cy = max(0, min(self._canvas_h - size, ty - size // 2))
                self._screen.blit(self._mag_surface, (cx, cy))

                # Label below cover
                if self._mag_label:
                    label_surf = self._font.render(
                        self._mag_label, True, (220, 220, 220)
                    )
                    lx = cx + (size - label_surf.get_width()) // 2
                    ly = cy + size + 6
                    if ly + label_surf.get_height() < self._canvas_h:
                        bg = pygame.Surface(
                            (label_surf.get_width() + 10, label_surf.get_height() + 6),
                            pygame.SRCALPHA,
                        )
                        bg.fill((0, 0, 0, 160))
                        self._screen.blit(bg, (lx - 5, ly - 3))
                        self._screen.blit(label_surf, (lx, ly))

        pygame.display.flip()
