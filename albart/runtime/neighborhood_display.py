"""3D perspective neighborhood visualization.

Shows the live audio embedding as a shaded sphere at screen-center, surrounded
by album art thumbnails from the entire library.

Hybrid projection:
  - Pre-computed 5-D UMAP captures global genre/mood structure.
  - Per-cycle local PCA (5D→3D) on K nearest neighbors adapts the 3-D view
    to the current query — showing the most interesting slice of the 5-D space.
  - Actual 512-D L2 distances, shifted so the nearest track is at the origin,
    determine radial distance from the sphere.
  - Each track is placed at:  PCA_direction × (L2 - L2_min) / median_shifted
  - |z| perspective with Z cutoff and depth fade.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pygame
from PIL import Image

from albart.pipeline.database import DB_PATH, get_connection
from albart.pipeline.embedder import FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH
from albart.utils import DATA_DIR

logger = logging.getLogger(__name__)

np.seterr(all="ignore")

EMBEDDINGS_PATH = DATA_DIR / "embeddings_norm.npy"
UMAP_5D_PATH        = DATA_DIR / "umap_5d.npy"


class NeighborhoodDisplay:
    """3D perspective neighborhood visualization around the live audio.

    Interface matches MapDisplay: call update() on new embeddings,
    render(dt) at display_fps.
    """

    def __init__(
        self,
        canvas_size: tuple[int, int],
        neighborhood_k: int = 100,
        base_thumb_px: int = 80,
        focal_length: float = 2.0,
        z_near: float = 0.5,
        zoom: float = 5.0,
        lerp_speed: float = 0.8,
        recompute_threshold: float = 0.15,
        recompute_consecutive: int = 3,
        trail_max_seconds: float = 90.0,
        brightness_k: float = 25.0,
        brightness_floor: float = 0.07,
        brightness_power: float = 1.5,
        sphere_base_radius: int = 50,
    ) -> None:
        self._canvas_w, self._canvas_h = canvas_size
        self._neighborhood_k = neighborhood_k
        self._base_thumb_px = base_thumb_px
        self._focal_length = focal_length
        self._z_near = z_near
        self._lerp_speed = lerp_speed
        self._brightness_k = brightness_k
        self._brightness_floor = brightness_floor
        self._brightness_power = brightness_power

        self._confidence: float = 0.0
        self._top1_track_id: Optional[str] = None
        self._label_text: str = ""

        # ── Load embeddings ────────────────────────────────────────────────
        logger.info("Loading embeddings...")
        faiss_ids = np.load(str(FAISS_RAW_IDS_PATH), allow_pickle=True)
        self._id_list: list[str] = [str(t) for t in faiss_ids]
        self._N = len(self._id_list)

        all_emb = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)

        # ── Load 5D UMAP positions ─────────────────────────────────────────
        logger.info("Loading 5D UMAP positions...")
        all_umap_5d = np.load(str(UMAP_5D_PATH)).astype(np.float32)

        # ── Deduplicate identical embeddings ───────────────────────────────
        _, unique_idx = np.unique(
            all_emb.view(np.uint8).reshape(all_emb.shape[0], -1),
            axis=0, return_index=True,
        )
        unique_idx.sort()
        n_dupes = self._N - len(unique_idx)
        if n_dupes > 0:
            logger.info("Deduplicated %d tracks (%d → %d)",
                        n_dupes, self._N, len(unique_idx))
            self._id_list = [self._id_list[i] for i in unique_idx]
            all_emb = all_emb[unique_idx]
            all_umap_5d = all_umap_5d[unique_idx]
            self._N = len(self._id_list)

        self._all_embeddings = all_emb
        self._all_umap_5d = all_umap_5d  # (N, 5) — fixed global structure

        # Pre-compute squared norms for fast L2 distance computation
        self._all_norms_sq = np.sum(
            self._all_embeddings.astype(np.float64) ** 2, axis=1
        )

        # ── Track metadata + art ─────────────────────────────────────────
        conn = get_connection(DB_PATH)
        rows = conn.execute(
            "SELECT track_id, title, artist, art_path_32, art_path_original "
            "FROM tracks"
        ).fetchall()
        conn.close()
        self._db_rows: dict = {row["track_id"]: row for row in rows}

        logger.info("Pre-loading thumbnail images...")
        self._pil_thumbs: dict[str, Optional[Image.Image]] = {}
        for tid in self._id_list:
            row = self._db_rows.get(tid)
            if row and (row["art_path_original"] or row["art_path_32"]):
                path = row["art_path_original"] or row["art_path_32"]
                try:
                    self._pil_thumbs[tid] = Image.open(
                        DATA_DIR / path
                    ).convert("RGB")
                except Exception:
                    self._pil_thumbs[tid] = None
            else:
                self._pil_thumbs[tid] = None

        # ── Pygame setup ─────────────────────────────────────────────────
        pygame.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode(
            (self._canvas_w, self._canvas_h), pygame.NOFRAME
        )
        pygame.display.set_caption("ALBart — Neighborhood")
        font_size = max(14, self._canvas_w // 160)
        self._font = pygame.font.SysFont("Arial", font_size)

        self._cx = self._canvas_w // 2
        self._cy = self._canvas_h // 2

        # ── Pre-render sphere sprites ────────────────────────────────────
        self._sphere_sprites = self._build_sphere_sprites(sphere_base_radius)
        self._sphere_base_radius = sphere_base_radius

        # ── Thumbnail surface cache ──────────────────────────────────────
        self._thumb_cache: dict[tuple[str, int], pygame.Surface] = {}

        # ── Projection state ─────────────────────────────────────────────
        self._L2_median_shift: float = 1.0

        # Cached PCA basis (5D→3D) — recomputed only on big embedding moves
        self._W: Optional[np.ndarray] = None       # (5, 3) float32
        self._cached_q_5d: Optional[np.ndarray] = None  # (5,) float32

        # Recompute gate for PCA basis
        self._last_pca_emb: Optional[np.ndarray] = None
        self._recompute_threshold = recompute_threshold
        self._recompute_consecutive = recompute_consecutive
        self._consecutive_far: int = 0

        # Auto-zoom
        self._ppu: float = 1695.0
        self._target_on_screen: int = 500

        # Hybrid positions (updated per cycle)
        self._all_proj:        Optional[np.ndarray] = None  # (N, 3) float32
        self._all_proj_smooth: Optional[np.ndarray] = None  # (N, 3) float32

        # ── Heatmap trail ────────────────────────────────────────────────
        self._heatmap_surf = pygame.Surface(
            (self._canvas_w, self._canvas_h), pygame.SRCALPHA
        )
        self._heatmap_surf.fill((0, 0, 0, 0))
        self._heatmap_decay_rate = np.log(128.0) / trail_max_seconds
        dot_r = max(8, self._canvas_w // 240)
        stamp = pygame.Surface(
            (dot_r * 2 + 1, dot_r * 2 + 1), pygame.SRCALPHA
        )
        stamp.fill((0, 0, 0, 0))
        pygame.draw.circle(stamp, (255, 255, 255, 255), (dot_r, dot_r), dot_r)
        self._heatmap_stamp   = stamp
        self._heatmap_stamp_r = dot_r
        self._prev_center: Optional[tuple[int, int]] = None

        # ── Hover / tooltip state ────────────────────────────────────────
        self._hover_track_id:  Optional[str] = None
        self._hover_alpha:     float = 0.0
        self._hover_art_cache: dict[str, Optional[pygame.Surface]] = {}
        self._tooltip_size     = self._canvas_w // 16
        self._frame_vis:    np.ndarray = np.empty(0, dtype=np.int32)
        self._frame_sx:     np.ndarray = np.empty(0, dtype=np.int32)
        self._frame_sy:     np.ndarray = np.empty(0, dtype=np.int32)
        self._frame_halfpx: np.ndarray = np.empty(0, dtype=np.int32)

        logger.info(
            "NeighborhoodDisplay ready — canvas=%dx%d  K=%d  "
            "lerp=%.2f  N=%d  (5D UMAP + local PCA)",
            self._canvas_w, self._canvas_h, neighborhood_k,
            lerp_speed, self._N,
        )

    # ── Sphere pre-rendering ─────────────────────────────────────────────

    @staticmethod
    def _build_sphere_sprites(
        max_radius: int, num_sizes: int = 24
    ) -> dict[int, pygame.Surface]:
        sprites: dict[int, pygame.Surface] = {}
        for r in np.linspace(6, max_radius, num_sizes).astype(int):
            r = int(r)
            size = r * 2 + 1
            yy, xx = np.mgrid[:size, :size]
            d_center    = np.sqrt((xx - r) ** 2.0 + (yy - r) ** 2.0)
            hx, hy      = r * 0.65, r * 0.65
            d_highlight = np.sqrt((xx - hx) ** 2.0 + (yy - hy) ** 2.0)
            mask        = d_center <= r
            brightness  = np.clip(
                1.0 - d_highlight / (r * 1.4), 0.0, 1.0
            ) ** 0.55
            edge_alpha  = np.clip(1.0 - (d_center / r) ** 2.0, 0.0, 1.0)
            rgba = np.zeros((size, size, 4), dtype=np.uint8)
            rgba[..., 0] = (210 * brightness * mask).astype(np.uint8)
            rgba[..., 1] = (210 * brightness * mask).astype(np.uint8)
            rgba[..., 2] = (
                np.clip(210 * brightness + 25, 0, 255).astype(np.uint8) * mask
            )
            rgba[..., 3] = (255 * edge_alpha * mask).astype(np.uint8)
            surf = pygame.image.frombuffer(
                rgba.tobytes(), (size, size), "RGBA"
            )
            sprites[r] = surf.convert_alpha()
        return sprites

    def _get_sphere(self, target_radius: int) -> pygame.Surface:
        keys = sorted(self._sphere_sprites.keys())
        best = min(keys, key=lambda k: abs(k - target_radius))
        return self._sphere_sprites[best]

    # ── Text helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> list[str]:
        words   = text.split()
        lines:  list[str] = []
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

    # ── Thumbnail cache ──────────────────────────────────────────────────

    def _get_cached_thumb(
        self, track_id: str, target_px: int
    ) -> Optional[pygame.Surface]:
        size = max(2, target_px)
        key  = (track_id, size)
        if key not in self._thumb_cache:
            pil_img = self._pil_thumbs.get(track_id)
            if pil_img is None:
                return None
            try:
                resized = pil_img.resize((size, size), Image.LANCZOS)
                arr     = np.array(resized, dtype=np.uint8)
                self._thumb_cache[key] = pygame.surfarray.make_surface(
                    arr.swapaxes(0, 1)
                )
            except Exception:
                return None
        return self._thumb_cache[key]

    # ── Tooltip ──────────────────────────────────────────────────────────

    def _get_tooltip_surface(self, track_id: str) -> Optional[pygame.Surface]:
        if track_id in self._hover_art_cache:
            return self._hover_art_cache[track_id]
        pil_img = self._pil_thumbs.get(track_id)
        surf    = None
        if pil_img is not None:
            try:
                resized = pil_img.resize(
                    (self._tooltip_size, self._tooltip_size), Image.LANCZOS
                )
                arr  = np.array(resized, dtype=np.uint8)
                surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            except Exception:
                pass
        self._hover_art_cache[track_id] = surf
        return surf

    # ── Projection pipeline ──────────────────────────────────────────────

    def _recompute_pca_basis(self, emb: np.ndarray, L2: np.ndarray) -> None:
        """Recompute the 5D→3D PCA basis from K nearest neighbors.

        Called infrequently (gated by recompute_threshold/consecutive).
        """
        K_pos = min(self._neighborhood_k, self._N)
        nearest_idx = np.argpartition(L2, K_pos)[:K_pos]
        nearest_L2 = L2[nearest_idx]

        # Query's 5D UMAP position (weighted avg of nearest tracks)
        weights = 1.0 / np.maximum(nearest_L2, 1e-8)
        weights /= weights.sum()
        q_5d = (weights[:, None] * self._all_umap_5d[nearest_idx]).sum(axis=0)
        self._cached_q_5d = q_5d.astype(np.float32)

        # Local PCA on K nearest neighbors in 5D UMAP space
        rel_5d_nb = (self._all_umap_5d[nearest_idx] - q_5d).astype(np.float64)
        cov = rel_5d_nb.T @ rel_5d_nb  # (5, 5)
        _, _, Vt = np.linalg.svd(cov)
        W = Vt[:3, :].T.astype(np.float32)  # (5, 3)

        # Procrustes: align to previous basis
        if self._W is not None:
            M = self._W.T @ W  # (3, 3)
            U_p, _, Vt_p = np.linalg.svd(M.astype(np.float64))
            d = np.sign(np.linalg.det(Vt_p.T @ U_p.T))
            R = (Vt_p.T @ np.diag([1.0, 1.0, d]) @ U_p.T).astype(np.float32)
            W = W @ R

        self._W = W
        self._last_pca_emb = np.nan_to_num(emb).copy()
        self._consecutive_far = 0
        logger.info("PCA basis recomputed (5D→3D)")

    def _compute_positions(self, emb: np.ndarray) -> np.ndarray:
        """Compute hybrid 3D positions using cached PCA basis + fresh L2.

        Called every cycle.
        """
        q64 = np.nan_to_num(
            emb, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float64).ravel()

        # L2 distances to all tracks
        q_norm_sq = float(np.dot(q64, q64))
        dots = self._all_embeddings.astype(np.float64) @ q64
        L2_sq = self._all_norms_sq + q_norm_sq - 2.0 * dots
        np.clip(L2_sq, 0.0, None, out=L2_sq)
        L2 = np.sqrt(L2_sq)

        # Shifted L2: nearest at 0, normalized by median of K-nearest
        L2_min = float(L2.min())
        L2_shifted = np.maximum(L2 - L2_min, 0.0)
        top_K = np.sort(L2_shifted)[:self._neighborhood_k]
        nz = top_K > 1e-8
        med = float(np.median(top_K[nz])) if nz.any() else 1.0
        self._L2_median_shift = max(med, 1e-8)
        L2_norm = (L2_shifted / self._L2_median_shift).astype(np.float32)

        # Recompute gate: only update PCA basis on big moves
        if self._last_pca_emb is not None:
            dist_moved = float(np.linalg.norm(q64 - self._last_pca_emb))
            if dist_moved > self._recompute_threshold:
                self._consecutive_far += 1
            else:
                self._consecutive_far = 0
            needs_recompute = self._consecutive_far >= self._recompute_consecutive
        else:
            needs_recompute = True

        if needs_recompute:
            self._recompute_pca_basis(emb, L2)

        # Update query's 5D position (weighted avg — smooth, not a basis change)
        K_pos = min(self._neighborhood_k, self._N)
        nearest_idx = np.argpartition(L2, K_pos)[:K_pos]
        nearest_L2 = L2[nearest_idx]
        weights = 1.0 / np.maximum(nearest_L2, 1e-8)
        weights /= weights.sum()
        q_5d = (weights[:, None] * self._all_umap_5d[nearest_idx]).sum(axis=0)

        # Project all tracks through cached W
        rel_5d = self._all_umap_5d - q_5d   # (N, 5)
        rel_3d = rel_5d @ self._W            # (N, 3)

        # Hybrid: 3D direction × shifted L2
        r_3d = np.linalg.norm(rel_3d, axis=1, keepdims=True)
        dirs = rel_3d / np.maximum(r_3d, 1e-8)
        return (dirs * L2_norm[:, None]).astype(np.float32)

    # ── Update (called on new UDP embedding) ──────────────────────────────

    def update(
        self,
        emb_raw:       np.ndarray,
        top1_track_id: str,
        d_min_raw:     float,
    ) -> None:
        self._top1_track_id = top1_track_id

        row = self._db_rows.get(top1_track_id) if top1_track_id else None
        if row:
            title  = row["title"]  or ""
            artist = row["artist"] or ""
            self._label_text = (
                f"{title}\n{artist}" if (title and artist) else title or artist
            )
        else:
            self._label_text = ""

        d_eff            = max(0.0, d_min_raw - self._brightness_floor)
        base             = float(np.exp(-self._brightness_k * d_eff))
        self._confidence = base ** self._brightness_power

        emb_clean = np.nan_to_num(emb_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Rate-limit: skip if less than 10s since last computation
        now = time.monotonic()
        if not hasattr(self, "_last_update_time"):
            self._last_update_time = 0.0
        if now - self._last_update_time < 9.5 and self._all_proj is not None:
            return
        self._last_update_time = now

        self._all_proj = self._compute_positions(emb_clean)

        if self._all_proj_smooth is None:
            self._all_proj_smooth = self._all_proj.copy()

    # ── Per-frame tick ───────────────────────────────────────────────────

    def _tick(self, dt: float) -> None:
        if self._all_proj is not None and self._all_proj_smooth is not None:
            alpha = min(1.0, self._lerp_speed * dt)
            self._all_proj_smooth += alpha * (
                self._all_proj - self._all_proj_smooth
            )

        # Hover detection
        if not pygame.mouse.get_focused():
            self._hover_track_id = None
        elif len(self._frame_vis) > 0:
            mx, my = pygame.mouse.get_pos()
            dists  = np.hypot(
                self._frame_sx - mx, self._frame_sy - my
            ).astype(np.float32)
            best      = int(np.argmin(dists))
            hit_r     = max(int(self._frame_halfpx[best]), 20)
            track_idx = int(self._frame_vis[best])
            tid       = self._id_list[track_idx]
            new_hover = tid if dists[best] < hit_r else None
            if new_hover != self._hover_track_id:
                self._hover_track_id = new_hover

        fade = dt * 4.0
        if self._hover_track_id is not None:
            self._hover_alpha = min(1.0, self._hover_alpha + fade)
        else:
            self._hover_alpha = max(0.0, self._hover_alpha - fade)

    # ── Rendering ────────────────────────────────────────────────────────

    def render(self, dt: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise SystemExit(0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

        self._tick(dt)
        self._screen.fill((0, 0, 0))

        # Heatmap decay + blit
        alpha_px = pygame.surfarray.pixels_alpha(self._heatmap_surf)
        decay    = float(np.exp(-dt * self._heatmap_decay_rate))
        np.multiply(alpha_px, decay, out=alpha_px, casting="unsafe")
        del alpha_px
        self._screen.blit(self._heatmap_surf, (0, 0))

        if self._all_proj_smooth is None:
            self._draw_sphere()
            pygame.display.flip()
            return

        # ── |z| perspective projection ────────────────────────────────────
        rel   = self._all_proj_smooth
        abs_z = np.abs(rel[:, 2])

        # Z cutoff: limit depth to the visible Y extent at z=0
        scale_at_z0 = self._focal_length / (self._z_near + self._focal_length)
        max_z = (self._canvas_h / 2.0) / (scale_at_z0 * self._ppu)
        z_ok  = abs_z <= max_z

        depth = abs_z + self._z_near
        scale = self._focal_length / (depth + self._focal_length)
        sx    = (self._cx + rel[:, 0] * scale * self._ppu).astype(np.int32)
        sy    = (self._cy + rel[:, 1] * scale * self._ppu).astype(np.int32)

        # Size: perspective × z-fade (shrinks to 10px at depth limit)
        z_frac    = np.clip(abs_z / max(max_z, 1e-8), 0.0, 1.0)
        min_thumb = 10.0
        max_thumb = float(self._base_thumb_px) * scale_at_z0
        z_thumb   = max_thumb * (1.0 - z_frac) + min_thumb * z_frac
        tpx       = np.maximum(1, np.minimum(
            self._base_thumb_px * scale, z_thumb
        ) + 0.5).astype(np.int32)

        # On-screen mask (x,y bounds + z cutoff)
        half   = tpx // 2
        on_scr = (
            z_ok &
            (sx + half >= 0) & (sx - half < self._canvas_w) &
            (sy + half >= 0) & (sy - half < self._canvas_h)
        )

        vis   = np.where(on_scr)[0]
        order = vis[np.argsort(-depth[vis])]

        self._frame_vis    = vis
        self._frame_sx     = sx[vis]
        self._frame_sy     = sy[vis]
        self._frame_halfpx = half[vis]

        # Auto-zoom
        n_vis = len(vis)
        if n_vis > self._target_on_screen * 1.1:
            self._ppu *= 1.002
        elif n_vis < self._target_on_screen * 0.9:
            self._ppu *= 0.998
        self._ppu = max(200.0, min(5000.0, self._ppu))

        # Draw thumbnails
        for i in order:
            tid = self._id_list[i]
            sz  = int(tpx[i])
            if sz < 2:
                self._screen.set_at((int(sx[i]), int(sy[i])), (120, 120, 120))
                continue
            surf = self._get_cached_thumb(tid, sz)
            if surf is not None:
                h = surf.get_width() // 2
                self._screen.blit(surf, (int(sx[i]) - h, int(sy[i]) - h))
            else:
                c = max(40, min(180, sz * 2))
                pygame.draw.circle(
                    self._screen, (c, c, c),
                    (int(sx[i]), int(sy[i])), max(1, sz // 4),
                )

        # Heatmap stamp
        center = (self._cx, self._cy)
        if self._prev_center is not None and center != self._prev_center:
            px, py = self._prev_center
            rr     = self._heatmap_stamp_r
            self._heatmap_surf.blit(
                self._heatmap_stamp, (px - rr, py - rr)
            )
        self._prev_center = center

        self._draw_sphere()
        self._draw_label()
        self._draw_tooltip()

        pygame.display.flip()

    def _handle_click(self, pos: tuple[int, int]) -> None:
        """On click, write the hovered track ID to dj_override.txt."""
        if self._hover_track_id is None:
            return
        override_path = DATA_DIR / "dj_override.txt"
        try:
            override_path.write_text(self._hover_track_id)
            row = self._db_rows.get(self._hover_track_id)
            name = f"{row['title']} — {row['artist']}" if row else self._hover_track_id
            logger.info("Click → DJ override: %s", name)
        except Exception as e:
            logger.warning("Failed to write DJ override: %s", e)

    def _draw_sphere(self) -> None:
        sphere_r = max(8, int(
            self._sphere_base_radius * (0.7 + 0.3 * self._confidence)
        ))
        sphere = self._get_sphere(sphere_r)
        sphere.set_alpha(int(180 + 75 * self._confidence))
        sr = sphere.get_width() // 2
        self._screen.blit(sphere, (self._cx - sr, self._cy - sr))

    def _draw_label(self) -> None:
        if not self._label_text:
            return
        sphere_r = max(8, int(
            self._sphere_base_radius * (0.7 + 0.3 * self._confidence)
        ))
        lines   = self._label_text.split("\n")
        lh      = self._font.get_linesize()
        block_h = len(lines) * lh
        block_w = max(self._font.size(ln)[0] for ln in lines)
        lx      = self._cx - block_w // 2
        ly      = self._cy + sphere_r + 10
        if ly + block_h >= self._canvas_h:
            return
        bg = pygame.Surface((block_w + 12, block_h + 8), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        self._screen.blit(bg, (lx - 6, ly - 4))
        for j, line in enumerate(lines):
            ls = self._font.render(line, True, (220, 220, 220))
            self._screen.blit(ls, (lx, ly + j * lh))

    def _draw_tooltip(self) -> None:
        if self._hover_alpha <= 0.0 or self._hover_track_id is None:
            return
        tip_surf = self._get_tooltip_surface(self._hover_track_id)
        if tip_surf is None:
            return
        tip_surf.set_alpha(int(self._hover_alpha * 255))
        mx, my = pygame.mouse.get_pos()
        sz     = tip_surf.get_width()
        lh     = self._font.get_linesize()
        row    = self._db_rows.get(self._hover_track_id)
        tip_lines: list[str] = []
        if row:
            if row["title"]:
                tip_lines.extend(self._wrap_text(row["title"], 20))
            if row["artist"]:
                tip_lines.extend(self._wrap_text(row["artist"], 20))
        block_h      = len(tip_lines) * lh
        tip_w, tip_h = sz, sz + 6 + block_h
        tx = min(mx + 16, self._canvas_w - tip_w - 4)
        ty = max(4, my - tip_h - 16)
        bg = pygame.Surface((tip_w + 8, tip_h + 8), pygame.SRCALPHA)
        bg.fill((0, 0, 0, int(190 * self._hover_alpha)))
        self._screen.blit(bg, (tx - 4, ty - 4))
        self._screen.blit(tip_surf, (tx, ty))
        for j, line in enumerate(tip_lines):
            ls = self._font.render(line, True, (220, 220, 220))
            ls.set_alpha(int(self._hover_alpha * 255))
            self._screen.blit(ls, (tx, ty + sz + 6 + j * lh))
