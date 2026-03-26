"""3D perspective neighborhood visualization — OpenGL GPU-accelerated.

Shows the live audio embedding as a shaded sphere at screen-center, surrounded
by album art thumbnails from the entire library.

Hybrid projection:
  - Pre-computed 5-D UMAP captures global genre/mood structure.
  - Per-cycle local PCA (5D→3D) on K nearest neighbors adapts the 3-D view
    to the current query — showing the most interesting slice of the 5-D space.
  - Actual 512-D L2 distances, shifted so the nearest track is at the origin,
    determine radial distance from the sphere.
  - OpenGL textured quads at float positions → GPU handles sub-pixel
    interpolation → perfectly smooth rotation and lerp.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, NOFRAME, OPENGL
from OpenGL.GL import *  # noqa: F403,F401
from PIL import Image

from albart.pipeline.database import DB_PATH, get_connection
from albart.pipeline.embedder import FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH
from albart.utils import DATA_DIR

logger = logging.getLogger(__name__)

np.seterr(all="ignore")

EMBEDDINGS_PATH = DATA_DIR / "embeddings_norm.npy"
UMAP_5D_PATH    = DATA_DIR / "umap_5d.npy"


class NeighborhoodDisplay:
    """3D perspective neighborhood visualization — OpenGL rendering.

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
        self._all_umap_5d = all_umap_5d

        self._all_norms_sq = np.sum(
            self._all_embeddings.astype(np.float64) ** 2, axis=1
        )

        # ── Track metadata ────────────────────────────────────────────────
        conn = get_connection(DB_PATH)
        rows = conn.execute(
            "SELECT track_id, title, artist, art_path_32, art_path_original "
            "FROM tracks"
        ).fetchall()
        conn.close()
        self._db_rows: dict = {row["track_id"]: row for row in rows}

        # ── Load PIL images (for GL texture upload) ───────────────────────
        logger.info("Pre-loading thumbnail images...")
        self._pil_thumbs: dict[str, Optional[Image.Image]] = {}
        for tid in self._id_list:
            row = self._db_rows.get(tid)
            if row and (row["art_path_original"] or row["art_path_32"]):
                path = row["art_path_original"] or row["art_path_32"]
                try:
                    self._pil_thumbs[tid] = Image.open(
                        DATA_DIR / path
                    ).convert("RGBA")
                except Exception:
                    self._pil_thumbs[tid] = None
            else:
                self._pil_thumbs[tid] = None

        # ── Pygame + OpenGL setup ─────────────────────────────────────────
        pygame.init()
        pygame.font.init()
        # Enable 4× multisampling for smooth polygon edges
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        # VSync: sync frame swaps to display refresh for even timing
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)
        self._screen = pygame.display.set_mode(
            (self._canvas_w, self._canvas_h), OPENGL | DOUBLEBUF | NOFRAME
        )
        pygame.display.set_caption("ALBart — Neighborhood")

        # OpenGL state
        glViewport(0, 0, self._canvas_w, self._canvas_h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self._canvas_w, self._canvas_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_MULTISAMPLE)
        glClearColor(0, 0, 0, 1)

        self._cx = self._canvas_w / 2.0
        self._cy = self._canvas_h / 2.0

        font_size = max(14, self._canvas_w // 160)
        self._font = pygame.font.SysFont("Arial", font_size)
        label_size = max(22, self._canvas_w // 80)
        self._label_font = pygame.font.SysFont("Helvetica", label_size, bold=True)
        self._label_font_shadow = pygame.font.SysFont("Helvetica", label_size, bold=True)

        # ── GL texture caches ─────────────────────────────────────────────
        self._gl_textures: dict[str, int] = {}  # tid → GL texture
        self._sphere_tex: int = 0
        self._sphere_base_radius = sphere_base_radius
        self._sphere_tex = self._build_sphere_texture(sphere_base_radius)
        self._label_tex: int = 0
        self._label_tex_w: int = 0
        self._label_tex_h: int = 0
        self._prev_label_text: str = ""

        # ── Projection state ─────────────────────────────────────────────
        self._L2_median_shift: float = 1.0

        self._W: Optional[np.ndarray] = None
        self._cached_q_5d: Optional[np.ndarray] = None
        self._last_pca_emb: Optional[np.ndarray] = None
        self._recompute_threshold = recompute_threshold
        self._recompute_consecutive = recompute_consecutive
        self._consecutive_far: int = 0

        self._ppu: float = 1695.0
        self._target_on_screen: int = 500

        self._all_proj:        Optional[np.ndarray] = None
        self._all_proj_smooth: Optional[np.ndarray] = None

        # Camera orbit (constant speed, constant direction)
        self._cam_angle: float = 0.0            # azimuth
        self._cam_tilt:  float = 0.12           # fixed elevation
        self._cam_rate:  float = 0.012          # rad/s (~0.7°/s)

        # ── Hover / tooltip state ────────────────────────────────────────
        self._hover_track_id:  Optional[str] = None
        self._hover_alpha:     float = 0.0
        self._tooltip_size     = self._canvas_w // 16
        self._frame_vis:    np.ndarray = np.empty(0, dtype=np.int32)
        self._frame_sx:     np.ndarray = np.empty(0, dtype=np.float32)
        self._frame_sy:     np.ndarray = np.empty(0, dtype=np.float32)
        self._frame_halfpx: np.ndarray = np.empty(0, dtype=np.float32)

        logger.info(
            "NeighborhoodDisplay ready — canvas=%dx%d  K=%d  "
            "lerp=%.2f  N=%d  (OpenGL + 5D UMAP)",
            self._canvas_w, self._canvas_h, neighborhood_k,
            lerp_speed, self._N,
        )

    # ── GL texture helpers ────────────────────────────────────────────────

    def _upload_rgba(self, data: bytes, w: int, h: int,
                     mipmap: bool = False) -> int:
        """Upload RGBA pixel data to a GL texture. Returns texture ID."""
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)
        if mipmap:
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                            GL_LINEAR_MIPMAP_LINEAR)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        return tex

    def _get_track_texture(self, tid: str) -> int:
        """Get or create GL texture for a track's album art."""
        if tid in self._gl_textures:
            return self._gl_textures[tid]
        pil_img = self._pil_thumbs.get(tid)
        if pil_img is None:
            self._gl_textures[tid] = 0
            return 0
        # Add 1px transparent-black border to prevent white fringe
        # from bilinear filtering at edges
        w, h = pil_img.width + 2, pil_img.height + 2
        bordered = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bordered.paste(pil_img, (1, 1))
        # Flip vertically for OpenGL's bottom-up texture layout
        flipped = bordered.transpose(Image.FLIP_TOP_BOTTOM)
        data = flipped.tobytes()
        tex = self._upload_rgba(data, w, h, mipmap=True)
        self._gl_textures[tid] = tex
        return tex

    def _build_sphere_texture(self, radius: int) -> int:
        """Build the glowing sphere as a GL texture."""
        size = radius * 2 + 1
        yy, xx = np.mgrid[:size, :size]
        d_center = np.sqrt((xx - radius) ** 2.0 + (yy - radius) ** 2.0)
        hx, hy = radius * 0.65, radius * 0.65
        d_highlight = np.sqrt((xx - hx) ** 2.0 + (yy - hy) ** 2.0)
        mask = d_center <= radius
        brightness = np.clip(
            1.0 - d_highlight / (radius * 1.4), 0.0, 1.0
        ) ** 0.55
        edge_alpha = np.clip(1.0 - (d_center / radius) ** 2.0, 0.0, 1.0)
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[..., 0] = (210 * brightness * mask).astype(np.uint8)
        rgba[..., 1] = (210 * brightness * mask).astype(np.uint8)
        rgba[..., 2] = (
            np.clip(210 * brightness + 25, 0, 255).astype(np.uint8) * mask
        )
        rgba[..., 3] = (255 * edge_alpha * mask).astype(np.uint8)
        return self._upload_rgba(rgba[::-1].tobytes(), size, size)

    def _draw_quad(self, tex: int, cx: float, cy: float,
                   half_w: float, half_h: float, alpha: float = 1.0,
                   border: bool = False) -> None:
        """Draw a textured quad centered at (cx, cy) with given half-sizes.

        border=True: texture has a 1px transparent border; adjust UVs to
        sample only the inner art region.
        """
        if tex == 0:
            return
        glBindTexture(GL_TEXTURE_2D, tex)
        glColor4f(1.0, 1.0, 1.0, alpha)
        if border:
            # Query actual texture size to compute UV insets
            tw = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
            th = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
            u0 = 1.0 / tw if tw > 2 else 0.0
            v0 = 1.0 / th if th > 2 else 0.0
            u1 = 1.0 - u0
            v1 = 1.0 - v0
        else:
            u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0
        glBegin(GL_QUADS)
        glTexCoord2f(u0, v1); glVertex2f(cx - half_w, cy - half_h)
        glTexCoord2f(u1, v1); glVertex2f(cx + half_w, cy - half_h)
        glTexCoord2f(u1, v0); glVertex2f(cx + half_w, cy + half_h)
        glTexCoord2f(u0, v0); glVertex2f(cx - half_w, cy + half_h)
        glEnd()

    def _update_label_texture(self) -> None:
        """Re-render label text to GL texture when text changes."""
        if self._label_text == self._prev_label_text:
            return
        self._prev_label_text = self._label_text
        if self._label_tex:
            glDeleteTextures([self._label_tex])
            self._label_tex = 0
        if not self._label_text:
            return
        lines = self._label_text.split("\n")
        lh = self._label_font.get_linesize()
        # Render text with shadow for readability
        text_surfs = []
        for ln in lines:
            shadow = self._label_font_shadow.render(ln, True, (0, 0, 0))
            fg = self._label_font.render(ln, True, (240, 240, 240))
            w_line = fg.get_width() + 4
            h_line = fg.get_height() + 2
            combined_line = pygame.Surface((w_line, h_line), pygame.SRCALPHA)
            # Draw shadow offset by 1-2px
            combined_line.blit(shadow, (2, 2))
            combined_line.blit(shadow, (1, 1))
            combined_line.blit(fg, (0, 0))
            text_surfs.append(combined_line)
        w = max(s.get_width() for s in text_surfs) + 16
        h = len(text_surfs) * lh + 12
        combined = pygame.Surface((w, h), pygame.SRCALPHA)
        combined.fill((0, 0, 0, 120))
        for j, s in enumerate(text_surfs):
            combined.blit(s, (8 + (w - 16 - s.get_width()) // 2, 6 + j * lh))
        data = pygame.image.tostring(combined, "RGBA", True)
        self._label_tex = self._upload_rgba(data, w, h)
        self._label_tex_w = w
        self._label_tex_h = h

    # ── Text helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> list[str]:
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

    # ── Projection pipeline (same math as before) ────────────────────────

    def _recompute_pca_basis(self, emb: np.ndarray, L2: np.ndarray) -> None:
        K_pos = min(self._neighborhood_k, self._N)
        nearest_idx = np.argpartition(L2, K_pos)[:K_pos]
        nearest_L2 = L2[nearest_idx]
        weights = 1.0 / np.maximum(nearest_L2, 1e-8)
        weights /= weights.sum()
        q_5d = (weights[:, None] * self._all_umap_5d[nearest_idx]).sum(axis=0)
        self._cached_q_5d = q_5d.astype(np.float32)
        rel_5d_nb = (self._all_umap_5d[nearest_idx] - q_5d).astype(np.float64)
        cov = rel_5d_nb.T @ rel_5d_nb
        _, _, Vt = np.linalg.svd(cov)
        W = Vt[:3, :].T.astype(np.float32)
        if self._W is not None:
            M = self._W.T @ W
            U_p, _, Vt_p = np.linalg.svd(M.astype(np.float64))
            d = np.sign(np.linalg.det(Vt_p.T @ U_p.T))
            R = (Vt_p.T @ np.diag([1.0, 1.0, d]) @ U_p.T).astype(np.float32)
            W = W @ R
        self._W = W
        self._last_pca_emb = np.nan_to_num(emb).copy()
        self._consecutive_far = 0
        logger.info("PCA basis recomputed (5D→3D)")

    def _compute_positions(self, emb: np.ndarray) -> np.ndarray:
        q64 = np.nan_to_num(
            emb, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float64).ravel()
        q_norm_sq = float(np.dot(q64, q64))
        dots = self._all_embeddings.astype(np.float64) @ q64
        L2_sq = self._all_norms_sq + q_norm_sq - 2.0 * dots
        np.clip(L2_sq, 0.0, None, out=L2_sq)
        L2 = np.sqrt(L2_sq)
        L2_min = float(L2.min())
        L2_shifted = np.maximum(L2 - L2_min, 0.0)
        top_K = np.sort(L2_shifted)[:self._neighborhood_k]
        nz = top_K > 1e-8
        med = float(np.median(top_K[nz])) if nz.any() else 1.0
        self._L2_median_shift = max(med, 1e-8)
        L2_norm = (L2_shifted / self._L2_median_shift).astype(np.float32)

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

        K_pos = min(self._neighborhood_k, self._N)
        nearest_idx = np.argpartition(L2, K_pos)[:K_pos]
        nearest_L2 = L2[nearest_idx]
        weights = 1.0 / np.maximum(nearest_L2, 1e-8)
        weights /= weights.sum()
        q_5d = (weights[:, None] * self._all_umap_5d[nearest_idx]).sum(axis=0)
        rel_5d = self._all_umap_5d - q_5d
        rel_3d = rel_5d @ self._W
        r_3d = np.linalg.norm(rel_3d, axis=1, keepdims=True)
        dirs = rel_3d / np.maximum(r_3d, 1e-8)
        return (dirs * L2_norm[:, None]).astype(np.float32)

    # ── Update ────────────────────────────────────────────────────────────

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

        d_eff = max(0.0, d_min_raw - self._brightness_floor)
        base = float(np.exp(-self._brightness_k * d_eff))
        self._confidence = base ** self._brightness_power

        emb_clean = np.nan_to_num(emb_raw, nan=0.0, posinf=0.0, neginf=0.0)

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

        # Camera orbit: constant speed, fixed direction
        self._cam_angle += self._cam_rate * dt

        # Hover detection
        if not pygame.mouse.get_focused():
            self._hover_track_id = None
        elif len(self._frame_vis) > 0:
            mx, my = pygame.mouse.get_pos()
            dists = np.hypot(
                self._frame_sx - mx, self._frame_sy - my
            )
            best = int(np.argmin(dists))
            hit_r = max(float(self._frame_halfpx[best]), 20.0)
            track_idx = int(self._frame_vis[best])
            tid = self._id_list[track_idx]
            new_hover = tid if dists[best] < hit_r else None
            if new_hover != self._hover_track_id:
                self._hover_track_id = new_hover

        fade = dt * 4.0
        if self._hover_track_id is not None:
            self._hover_alpha = min(1.0, self._hover_alpha + fade)
        else:
            self._hover_alpha = max(0.0, self._hover_alpha - fade)

    # ── Rendering (OpenGL) ────────────────────────────────────────────────

    def render(self, dt: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise SystemExit(0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

        self._tick(dt)
        glClear(GL_COLOR_BUFFER_BIT)

        if self._all_proj_smooth is None:
            self._draw_sphere()
            pygame.display.flip()
            return

        # ── Camera orbit rotation ─────────────────────────────────────────
        ca, sa = np.cos(self._cam_angle), np.sin(self._cam_angle)
        ce, se = np.cos(self._cam_tilt), np.sin(self._cam_tilt)
        R = np.array([
            [ ca,   sa * se,  sa * ce],
            [ 0.0,  ce,      -se     ],
            [-sa,   ca * se,  ca * ce],
        ], dtype=np.float32)
        rel = self._all_proj_smooth @ R.T

        # ── Perspective projection (raw z, float positions) ───────────────
        z = rel[:, 2]
        in_front = z > -self._z_near

        depth = np.maximum(z + self._z_near, 0.01)
        scale = self._focal_length / (depth + self._focal_length)

        # Float positions — GPU handles sub-pixel interpolation
        sx = self._cx + rel[:, 0] * scale * self._ppu
        sy = self._cy + rel[:, 1] * scale * self._ppu

        scale_at_z0 = self._focal_length / (self._z_near + self._focal_length)
        max_z = (self._canvas_h / 2.0) / (scale_at_z0 * self._ppu)
        z_ok = z <= max_z

        # Size: perspective scale for foreground, depth fade for background
        persp_size = self._base_thumb_px * scale  # raw perspective size

        # Background depth fade (z > 0): shrinks to 10px at max_z
        z_frac = np.clip(z / max(max_z, 1e-8), 0.0, 1.0)
        min_thumb = 10.0
        max_thumb = float(self._base_thumb_px) * scale_at_z0
        bg_size = max_thumb * (1.0 - z_frac) + min_thumb * z_frac

        # Foreground (z < 0): use full perspective size (grows as tracks approach)
        # Background (z > 0): use the smaller of perspective and depth fade
        tpx_base = np.where(z < 0, persp_size, np.minimum(persp_size, bg_size))

        tpx = np.maximum(1.0, tpx_base)

        # Alpha fade near depth boundaries (prevents pop-in/pop-out)
        # Fade out near max_z (back edge)
        fade_zone = max_z * 0.15
        back_alpha = np.clip((max_z - z) / max(fade_zone, 1e-8), 0.0, 1.0)
        # Foreground fade: tracks between camera and sphere (z<0) become
        # transparent.  Fully opaque at z=0, fully transparent at z = -fade_front.
        fade_front = self._z_near * 0.8  # fade quickly — foreground covers are large
        front_alpha = np.clip(z / fade_front + 1.0, 0.0, 1.0)
        # Axis-proximity fade: foreground tracks near the z-axis (camera-to-cover
        # line) become transparent so the central cover always shows through.
        # Tracks further from the axis stay opaque.
        r_xy = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
        cover_half_3d = float(self._base_thumb_px) * scale_at_z0 / (2.0 * self._ppu)
        cyl_radius_3d = cover_half_3d * 3.0  # fade zone = 3× cover radius
        axis_alpha = np.where(
            z < 0,  # only apply to foreground
            np.clip(r_xy / max(cyl_radius_3d, 1e-8), 0.0, 1.0),
            1.0,
        )

        track_alpha = back_alpha * front_alpha * axis_alpha

        # On-screen mask: must be in front AND visible
        half = tpx / 2.0
        on_scr = (
            in_front & (track_alpha > 0.01) &
            (sx + half >= 0) & (sx - half < self._canvas_w) &
            (sy + half >= 0) & (sy - half < self._canvas_h)
        )

        vis = np.where(on_scr)[0]
        order = vis[np.argsort(-depth[vis])]  # back to front

        # Cache for hover detection (float positions)
        self._frame_vis    = vis
        self._frame_sx     = sx[vis].astype(np.float32)
        self._frame_sy     = sy[vis].astype(np.float32)
        self._frame_halfpx = half[vis].astype(np.float32)

        # Auto-zoom
        n_vis = max(len(vis), 1)
        error = (n_vis - self._target_on_screen) / self._target_on_screen
        zoom_rate = 0.06
        self._ppu *= (1.0 + error * zoom_rate * dt)
        self._ppu = max(200.0, min(5000.0, self._ppu))


        # ── Draw tracks + glow frame + label layered at Z=0 ──────────────
        # Background tracks (z > 0) first, then glow/label, then foreground
        self._update_label_texture()
        glow_drawn = False
        for i in order:
            # Insert glow frame + label at the Z=0 boundary
            if not glow_drawn and z[i] <= 0:
                self._draw_label()   # text behind glow frame
                self._draw_sphere()  # glow on top of text background
                glow_drawn = True
            tid = self._id_list[i]
            tex = self._get_track_texture(tid)
            if tex == 0:
                continue
            h = float(tpx[i]) / 2.0
            self._draw_quad(tex, float(sx[i]), float(sy[i]), h, h,
                            float(track_alpha[i]), border=True)
        if not glow_drawn:
            self._draw_label()
            self._draw_sphere()

        # ── Tooltip ───────────────────────────────────────────────────────
        self._draw_tooltip()

        pygame.display.flip()

    def _draw_sphere(self) -> None:
        """Draw a glowing hollow square frame around the central cover."""
        scale_at_z0 = self._focal_length / (self._z_near + self._focal_length)
        frame_half = float(self._base_thumb_px) * scale_at_z0 / 2.0 + 4.0
        a = 0.35 + 0.25 * self._confidence
        glow = 25.0  # glow spread in pixels
        r, g, b = 0.45, 0.5, 1.0  # blue-white glow color

        cx, cy = self._cx, self._cy
        L = cx - frame_half   # inner left
        R = cx + frame_half   # inner right
        T = cy - frame_half   # inner top
        B = cy + frame_half   # inner bottom

        glDisable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)

        # Top edge glow (inner bright → outer transparent)
        glColor4f(r, g, b, a);   glVertex2f(L, T)
        glColor4f(r, g, b, a);   glVertex2f(R, T)
        glColor4f(r, g, b, 0.0); glVertex2f(R, T - glow)
        glColor4f(r, g, b, 0.0); glVertex2f(L, T - glow)

        # Bottom edge glow
        glColor4f(r, g, b, a);   glVertex2f(L, B)
        glColor4f(r, g, b, a);   glVertex2f(R, B)
        glColor4f(r, g, b, 0.0); glVertex2f(R, B + glow)
        glColor4f(r, g, b, 0.0); glVertex2f(L, B + glow)

        # Left edge glow
        glColor4f(r, g, b, a);   glVertex2f(L, T)
        glColor4f(r, g, b, a);   glVertex2f(L, B)
        glColor4f(r, g, b, 0.0); glVertex2f(L - glow, B)
        glColor4f(r, g, b, 0.0); glVertex2f(L - glow, T)

        # Right edge glow
        glColor4f(r, g, b, a);   glVertex2f(R, T)
        glColor4f(r, g, b, a);   glVertex2f(R, B)
        glColor4f(r, g, b, 0.0); glVertex2f(R + glow, B)
        glColor4f(r, g, b, 0.0); glVertex2f(R + glow, T)

        # Corner glow (triangle fans approximated as quads)
        # Top-left
        glColor4f(r, g, b, a);   glVertex2f(L, T)
        glColor4f(r, g, b, 0.0); glVertex2f(L - glow, T)
        glColor4f(r, g, b, 0.0); glVertex2f(L - glow, T - glow)
        glColor4f(r, g, b, 0.0); glVertex2f(L, T - glow)
        # Top-right
        glColor4f(r, g, b, a);   glVertex2f(R, T)
        glColor4f(r, g, b, 0.0); glVertex2f(R, T - glow)
        glColor4f(r, g, b, 0.0); glVertex2f(R + glow, T - glow)
        glColor4f(r, g, b, 0.0); glVertex2f(R + glow, T)
        # Bottom-left
        glColor4f(r, g, b, a);   glVertex2f(L, B)
        glColor4f(r, g, b, 0.0); glVertex2f(L, B + glow)
        glColor4f(r, g, b, 0.0); glVertex2f(L - glow, B + glow)
        glColor4f(r, g, b, 0.0); glVertex2f(L - glow, B)
        # Bottom-right
        glColor4f(r, g, b, a);   glVertex2f(R, B)
        glColor4f(r, g, b, 0.0); glVertex2f(R + glow, B)
        glColor4f(r, g, b, 0.0); glVertex2f(R + glow, B + glow)
        glColor4f(r, g, b, 0.0); glVertex2f(R, B + glow)

        glEnd()

        # Thin crisp inner frame
        glColor4f(0.7, 0.73, 1.0, a * 0.5)
        self._draw_rect_outline(cx, cy, frame_half, frame_half, 1.0)

        glEnable(GL_TEXTURE_2D)

    def _draw_rect_outline(self, cx: float, cy: float,
                           half_w: float, half_h: float,
                           thickness: float) -> None:
        """Draw a rectangular outline using GL quads."""
        l, r = cx - half_w, cx + half_w
        t, b = cy - half_h, cy + half_h
        glBegin(GL_QUADS)
        # Top edge
        glVertex2f(l - thickness, t - thickness)
        glVertex2f(r + thickness, t - thickness)
        glVertex2f(r + thickness, t)
        glVertex2f(l - thickness, t)
        # Bottom edge
        glVertex2f(l - thickness, b)
        glVertex2f(r + thickness, b)
        glVertex2f(r + thickness, b + thickness)
        glVertex2f(l - thickness, b + thickness)
        # Left edge
        glVertex2f(l - thickness, t)
        glVertex2f(l, t)
        glVertex2f(l, b)
        glVertex2f(l - thickness, b)
        # Right edge
        glVertex2f(r, t)
        glVertex2f(r + thickness, t)
        glVertex2f(r + thickness, b)
        glVertex2f(r, b)
        glEnd()

    def _draw_label(self) -> None:
        if self._label_tex == 0:
            return
        sphere_r = max(8, int(
            self._sphere_base_radius * (0.7 + 0.3 * self._confidence)
        ))
        lx = self._cx
        ly = self._cy + sphere_r + 10 + self._label_tex_h / 2.0
        self._draw_quad(self._label_tex, lx,  ly,
                        self._label_tex_w / 2.0, self._label_tex_h / 2.0)

    def _draw_tooltip(self) -> None:
        if self._hover_alpha <= 0.0 or self._hover_track_id is None:
            return
        # Render tooltip text to a temporary texture
        row = self._db_rows.get(self._hover_track_id)
        if not row:
            return
        lines: list[str] = []
        if row["title"]:
            lines.extend(self._wrap_text(row["title"], 20))
        if row["artist"]:
            lines.extend(self._wrap_text(row["artist"], 20))
        if not lines:
            return

        surfs = [self._font.render(ln, True, (220, 220, 220)) for ln in lines]
        lh = self._font.get_linesize()

        # Album art + text
        art_size = self._tooltip_size
        w = max(art_size, max(s.get_width() for s in surfs) + 8)
        h = art_size + 6 + len(surfs) * lh + 8

        combined = pygame.Surface((w, h), pygame.SRCALPHA)
        combined.fill((0, 0, 0, int(190 * self._hover_alpha)))

        pil_img = self._pil_thumbs.get(self._hover_track_id)
        if pil_img is not None:
            resized = pil_img.resize((art_size, art_size), Image.LANCZOS)
            arr = np.array(resized)
            tip_surf = pygame.image.frombuffer(
                arr.tobytes(), (art_size, art_size), "RGBA"
            )
            combined.blit(tip_surf, ((w - art_size) // 2, 0))

        for j, s in enumerate(surfs):
            combined.blit(s, (4, art_size + 6 + j * lh))

        data = pygame.image.tostring(combined, "RGBA", True)
        tex = self._upload_rgba(data, w, h)

        mx, my = pygame.mouse.get_pos()
        tx = min(float(mx) + 16, self._canvas_w - w - 4)
        ty = max(4.0, float(my) - h - 16)
        self._draw_quad(tex, tx + w / 2.0, ty + h / 2.0,
                        w / 2.0, h / 2.0, self._hover_alpha)
        glDeleteTextures([tex])

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self._hover_track_id is None:
            return
        override_path = DATA_DIR / "dj_override.txt"
        try:
            override_path.write_text(self._hover_track_id)
            row = self._db_rows.get(self._hover_track_id)
            name = (f"{row['title']} — {row['artist']}"
                    if row else self._hover_track_id)
            logger.info("Click → DJ override: %s", name)
        except Exception as e:
            logger.warning("Failed to write DJ override: %s", e)
