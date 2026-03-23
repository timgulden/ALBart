"""UMAP music map visualization.

Renders all tracks as micro-thumbnails at their 2D UMAP positions on a dark
canvas.  A grey dot traces the live audio embedding (via UMAP transform).  The
closest matching album cover is magnified near the dot and labeled.  A long
fading trail records the recent path through music space.

Designed to run as a separate process, receiving DualEmbedding payloads over
a local UDP socket from the main ALBart runtime.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import threading
import time
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
                with contextlib.redirect_stderr(io.StringIO()):
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
        show_voronoi:        bool = True,
    ) -> None:
        self._canvas_w, self._canvas_h = canvas_size
        self._thumb_size         = thumb_size
        self._trail_max_seconds  = trail_max_seconds
        self._brightness_k       = brightness_k
        self._brightness_floor   = brightness_floor
        self._brightness_power   = brightness_power
        self._show_voronoi       = show_voronoi

        # Live state
        self._dot_pos:       Optional[tuple[float, float]] = None  # screen coords
        self._top1_track_id: Optional[str]  = None
        self._confidence:    float          = 0.0

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
        umap_model.verbose = False
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

        # Voronoi regions (optional — graceful if not built yet)
        voronoi_path = DATA_DIR / "voronoi_regions.json"
        self._voronoi_regions: list[dict] = []
        if voronoi_path.exists() and show_voronoi:
            with open(voronoi_path) as f:
                self._voronoi_regions = json.load(f)
            logger.info("Loaded %d Voronoi regions", len(self._voronoi_regions))
        elif show_voronoi:
            logger.info("No voronoi_regions.json found — run tools/build_voronoi.py")

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
        self._dot_radius    = max(8, self._canvas_w // 240)
        self._mag_max_size  = self._canvas_w // 13   # ~295px at 4K, ~148px at 1080p
        self._tooltip_size  = self._canvas_w // 16   # ~120px at 1080p

        # Persistent SRCALPHA heatmap surface: dots are stamped in and decay over time
        self._heatmap_surf = pygame.Surface(
            (self._canvas_w, self._canvas_h), pygame.SRCALPHA
        )
        self._heatmap_surf.fill((0, 0, 0, 0))
        # Pre-built stamp: white circle, 100% opaque, 2× dot_radius
        r = self._dot_radius * 2
        self._heatmap_stamp_r = r
        _stamp = pygame.Surface((r * 2 + 1, r * 2 + 1), pygame.SRCALPHA)
        _stamp.fill((0, 0, 0, 0))
        pygame.draw.circle(_stamp, (255, 255, 255, 255), (r, r), r)
        self._heatmap_dot_stamp = _stamp
        # Precompute per-frame decay rate: alpha reaches ~0 after trail_max_seconds.
        # ln(128) gives a gentler fade than ln(256) — starts at 255, fades to ~2.
        self._heatmap_decay_rate = np.log(128.0) / trail_max_seconds

        # Hover / tooltip state
        self._hover_track_id: Optional[str]           = None
        self._hover_alpha:    float                   = 0.0
        self._hover_art_cache: dict[str, Optional[pygame.Surface]] = {}

        # Spatial index for hit-detection: parallel arrays of screen positions and IDs
        _ids  = list(self._id_to_screen.keys())
        self._hit_ids = _ids
        self._hit_pos = np.array(
            [self._id_to_screen[tid] for tid in _ids], dtype=np.float32
        )  # (N, 2)

        # Pre-render all thumbnails to background surface (done once at startup)
        logger.info("Pre-rendering background (%d thumbnails)...", len(umap_ids))
        self._bg_surface = self._build_background(umap_ids)

        # Pre-compute Voronoi label layout for runtime rendering (SRCALPHA needs screen)
        _vfont_size = max(13, self._canvas_w // 110)
        self._voronoi_font = pygame.font.SysFont("Arial", _vfont_size)
        _vlh = self._voronoi_font.get_linesize()
        self._voronoi_labels: list[tuple[int, int, list[str]]] = []  # (cx, cy, lines)
        for region in self._voronoi_regions:
            cx, cy = self._norm_to_screen(*region["centroid"])
            lines = self._wrap_text(region["label"], 20)
            self._voronoi_labels.append((cx, cy, lines))
        self._voronoi_line_h = _vlh

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

    # ── Text helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> list[str]:
        """Wrap text at word boundaries so no line exceeds max_chars."""
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > max_chars:
                lines.append(current)
                current = word
            else:
                current = (current + " " + word).strip()
        if current:
            lines.append(current)
        return lines or [text]

    # ── Startup ───────────────────────────────────────────────────────────────

    def _get_tooltip_surface(self, track_id: str) -> Optional[pygame.Surface]:
        """Return a cached pygame Surface scaled to tooltip_size, or None."""
        if track_id in self._hover_art_cache:
            return self._hover_art_cache[track_id]
        row = self._db_rows.get(track_id)
        surf = None
        if row:
            path = row["art_path_original"] or row["art_path_32"]
            if path:
                try:
                    img = Image.open(DATA_DIR / path).convert("RGB")
                    img = img.resize(
                        (self._tooltip_size, self._tooltip_size), Image.LANCZOS
                    )
                    arr = np.array(img, dtype=np.uint8)
                    surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
                except Exception as e:
                    logger.debug("Tooltip art load failed for %s: %s", track_id, e)
        self._hover_art_cache[track_id] = surf
        return surf

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
        """Advance animated state: dot position, heatmap, magnified cover size."""
        # Update dot from latest UMAP result
        result = self._transformer.get_normalized()
        if result is not None:
            nx = max(0.0, min(1.0, result[0]))
            ny = max(0.0, min(1.0, result[1]))
            new_pos = self._norm_to_screen(nx, ny)

            if self._dot_pos is not None and new_pos != self._dot_pos:
                # Stamp a white dot at the previous position onto the heatmap
                px, py = self._dot_pos
                r = self._heatmap_stamp_r
                self._heatmap_surf.blit(
                    self._heatmap_dot_stamp,
                    (int(px) - r, int(py) - r),
                )
            self._dot_pos = new_pos

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
        min_px    = max(self._mag_max_size // 5, 40)
        target_px = max(min_px, int(self._mag_max_size * self._confidence))
        diff = target_px - self._mag_size
        self._mag_size += diff * min(1.0, dt * 3.0)  # ~3× per second lerp

        # Hover detection: find nearest thumbnail to mouse cursor
        mx, my = pygame.mouse.get_pos()
        diffs = self._hit_pos - np.array([mx, my], dtype=np.float32)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        idx   = int(np.argmin(dists))
        new_hover = self._hit_ids[idx] if dists[idx] < self._thumb_size else None
        if new_hover != self._hover_track_id:
            self._hover_track_id = new_hover

        # Fade hover tooltip in/out (4× per second → 0.25s transition)
        fade = dt * 4.0
        if self._hover_track_id is not None:
            self._hover_alpha = min(1.0, self._hover_alpha + fade)
        else:
            self._hover_alpha = max(0.0, self._hover_alpha - fade)

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

        # 1 — Background (all thumbnails + Voronoi borders, pre-rendered)
        self._screen.blit(self._bg_surface, (0, 0))

        # 1b — Voronoi labels with SRCALPHA backing (must draw on screen, not bg surface)
        lh = self._voronoi_line_h
        pad = 4
        for cx, cy, lines in self._voronoi_labels:
            block_w = max(self._voronoi_font.size(ln)[0] for ln in lines)
            block_h = len(lines) * lh
            bg = pygame.Surface((block_w + pad * 2, block_h + pad * 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 150))
            self._screen.blit(bg, (cx - block_w // 2 - pad, cy - block_h // 2 - pad))
            for j, line in enumerate(lines):
                surf = self._voronoi_font.render(line, True, (180, 180, 180))
                self._screen.blit(surf, (cx - surf.get_width() // 2, cy - block_h // 2 + j * lh))

        # 2 — Heatmap: decay all alphas per-frame, then blit to screen
        alpha_px = pygame.surfarray.pixels_alpha(self._heatmap_surf)
        decay = float(np.exp(-dt * self._heatmap_decay_rate))
        np.multiply(alpha_px, decay, out=alpha_px, casting="unsafe")
        del alpha_px  # release pixel lock
        self._screen.blit(self._heatmap_surf, (0, 0))

        # 3 — Grey query dot
        if self._dot_pos is not None:
            dx, dy = int(self._dot_pos[0]), int(self._dot_pos[1])
            pygame.draw.circle(
                self._screen, (100, 100, 100), (dx, dy), self._dot_radius + 2
            )
            pygame.draw.circle(
                self._screen, (210, 210, 210), (dx, dy), self._dot_radius
            )

        # 4 — Magnified cover + label, centered above the live dot.
        # The dot shows where the live audio sits in UMAP space; the cover
        # confirms what the system is matching at that location.
        if self._mag_track_id and self._mag_orig_image is not None:
            size = max(8, int(self._mag_size))
            if size != self._mag_last_px:
                scaled = self._mag_orig_image.resize((size, size), Image.LANCZOS)
                arr = np.array(scaled, dtype=np.uint8)
                self._mag_surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
                self._mag_last_px = size

            if self._mag_surface is not None:
                # Center the cover above the dot with a small gap
                dot_x = self._dot_pos[0] if self._dot_pos is not None else self._canvas_w // 2
                dot_y = self._dot_pos[1] if self._dot_pos is not None else self._canvas_h // 2
                cx = max(0, min(self._canvas_w - size, int(dot_x) - size // 2))
                cy = max(0, min(self._canvas_h - size, int(dot_y) - size - self._dot_radius - 6))
                self._screen.blit(self._mag_surface, (cx, cy))

                # Label above cover — wrapped, semi-transparent background
                if self._mag_label:
                    # Split "Title — Artist" onto separate lines, wrap each at 20 chars
                    parts = self._mag_label.split(" — ", 1)
                    raw_lines: list[str] = []
                    for part in parts:
                        raw_lines.extend(self._wrap_text(part, 20))
                    lh = self._font.get_linesize()
                    block_h = len(raw_lines) * lh
                    block_w = max(
                        self._font.size(line)[0] for line in raw_lines
                    )
                    lx = cx + (size - block_w) // 2
                    ly = cy - block_h - 8
                    if ly >= 0:
                        bg = pygame.Surface((block_w + 10, block_h + 6), pygame.SRCALPHA)
                        bg.fill((0, 0, 0, 175))
                        self._screen.blit(bg, (lx - 5, ly - 3))
                        for j, line in enumerate(raw_lines):
                            line_surf = self._font.render(line, True, (220, 220, 220))
                            self._screen.blit(line_surf, (lx, ly + j * lh))

        # 5 — Hover tooltip: larger art + label near cursor
        if self._hover_alpha > 0.0 and self._hover_track_id is not None:
            tip_surf = self._get_tooltip_surface(self._hover_track_id)
            if tip_surf is not None:
                tip_surf.set_alpha(int(self._hover_alpha * 255))
                mx, my = pygame.mouse.get_pos()
                sz = tip_surf.get_width()
                lh = self._font.get_linesize()
                row = self._db_rows.get(self._hover_track_id)
                tip_lines: list[str] = []
                if row:
                    parts = []
                    if row["title"]:  parts.extend(self._wrap_text(row["title"],  20))
                    if row["artist"]: parts.extend(self._wrap_text(row["artist"], 20))
                    tip_lines = parts
                block_h = len(tip_lines) * lh
                tip_w = sz
                tip_h = sz + 6 + block_h
                # Position: above and right of cursor, clamped to screen
                tx = min(mx + 16, self._canvas_w  - tip_w - 4)
                ty = max(4,       my - tip_h - 16)
                # Background
                bg = pygame.Surface((tip_w + 8, tip_h + 8), pygame.SRCALPHA)
                bg.fill((0, 0, 0, int(190 * self._hover_alpha)))
                self._screen.blit(bg, (tx - 4, ty - 4))
                # Art
                self._screen.blit(tip_surf, (tx, ty))
                # Label lines
                for j, line in enumerate(tip_lines):
                    ls = self._font.render(line, True, (220, 220, 220))
                    ls.set_alpha(int(self._hover_alpha * 255))
                    self._screen.blit(ls, (tx, ty + sz + 6 + j * lh))

        pygame.display.flip()
