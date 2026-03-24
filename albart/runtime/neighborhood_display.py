"""3D perspective neighborhood visualization.

Shows the live audio embedding as a shaded sphere at screen-center, surrounded
by album art thumbnails from the entire library.

Hybrid projection:
  - MDS on K nearest neighbors gives 3-D *directions* (angular structure —
    similar tracks cluster together in the same direction from center).
  - Actual 512-D L2 distances, shifted so the nearest track is at the origin,
    determine *how far* each track is from the sphere.
  - Each track is placed at:  MDS_direction × (L2 - L2_min) / median_shifted
  - Standard |z| perspective projection gives natural cloud depth.
  - Between recomputes, directions are cached (stable angular layout); only
    L2 distances update per cycle → tracks move radially, not angularly.
  - On recompute, Procrustes alignment prevents axis flips.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import faiss
import numpy as np
import pygame
from PIL import Image

from albart.pipeline.database import DB_PATH, get_connection
from albart.pipeline.embedder import FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH
from albart.utils import DATA_DIR

logger = logging.getLogger(__name__)

np.seterr(all="ignore")  # suppress BLAS matmul warnings (values are finite)

EMBEDDINGS_RAW_PATH = DATA_DIR / "embeddings_raw.npy"


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
        self._recompute_threshold = recompute_threshold
        self._recompute_consecutive = recompute_consecutive
        self._brightness_k = brightness_k
        self._brightness_floor = brightness_floor
        self._brightness_power = brightness_power

        self._confidence: float = 0.0
        self._top1_track_id: Optional[str] = None
        self._label_text: str = ""

        # ── Load FAISS index + all embeddings ────────────────────────────
        logger.info("Loading FAISS raw index...")
        self._faiss_index = faiss.read_index(str(FAISS_RAW_INDEX_PATH))
        faiss_ids = np.load(str(FAISS_RAW_IDS_PATH), allow_pickle=True)
        self._id_list: list[str] = [str(t) for t in faiss_ids]
        self._N = len(self._id_list)

        logger.info("Loading all embeddings (%d tracks)...", self._N)
        all_emb = np.load(str(EMBEDDINGS_RAW_PATH)).astype(np.float32)

        # Deduplicate: tracks with identical embeddings (same song on
        # multiple albums) cause z-fighting flicker.  Keep first occurrence.
        _, unique_idx = np.unique(
            all_emb.view(np.uint8).reshape(all_emb.shape[0], -1),
            axis=0, return_index=True,
        )
        unique_idx.sort()  # preserve original order
        n_dupes = self._N - len(unique_idx)

        # Map from original FAISS index → deduplicated index (-1 if removed)
        self._faiss_to_dedup = np.full(len(all_emb), -1, dtype=np.int32)
        for new_i, orig_i in enumerate(unique_idx):
            self._faiss_to_dedup[orig_i] = new_i

        if n_dupes > 0:
            logger.info("Deduplicated %d tracks with identical embeddings "
                        "(%d → %d)", n_dupes, len(all_emb), len(unique_idx))
            self._id_list = [self._id_list[i] for i in unique_idx]
            all_emb = all_emb[unique_idx]
            self._N = len(self._id_list)

        self._all_embeddings = all_emb

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


        # ── MDS / Nyström state (set on recompute, fixed between) ────────
        self._landmarks:     Optional[np.ndarray] = None
        self._lm_norms_sq:   Optional[np.ndarray] = None
        self._V_k:           Optional[np.ndarray] = None
        self._inv_sqrt_lam:  Optional[np.ndarray] = None
        self._col_means:     Optional[np.ndarray] = None
        self._grand_mean:    float = 0.0

        # All N tracks projected into MDS space (fixed between recomputes,
        # Procrustes-aligned; directions computed fresh each cycle from these)
        self._all_proj_mds: Optional[np.ndarray] = None     # (N, 3) float32

        # L2 normalization: shift by min, divide by median of shifted
        self._L2_min:          float = 0.0
        self._L2_median_shift: float = 1.0

        # Fixed ppu (tuned for ~500 on screen at 1080p)
        self._ppu: float = 1695.0

        # Hybrid positions (updated per cycle)
        self._all_proj:        Optional[np.ndarray] = None  # (N, 3) float32
        self._all_proj_smooth: Optional[np.ndarray] = None  # (N, 3) float32

        # ── Recompute gating ─────────────────────────────────────────────
        self._last_basis_emb:  Optional[np.ndarray] = None
        self._consecutive_far: int = 0

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
            "lerp=%.2f  threshold=%.3f  consecutive=%d  N=%d",
            self._canvas_w, self._canvas_h, neighborhood_k,
            lerp_speed, recompute_threshold, recompute_consecutive, self._N,
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

    # ── L2 distance computation ──────────────────────────────────────────

    def _compute_L2_shifted(self, emb: np.ndarray) -> np.ndarray:
        """Shifted + normalized L2 distances: nearest track at 0.

        Returns (N,) float32 = (L2 - L2_min) / median_shifted.
        """
        q64 = np.nan_to_num(
            emb, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float64).ravel()
        q_norm_sq = float(np.dot(q64, q64))
        dots = self._all_embeddings.astype(np.float64) @ q64
        L2_sq = self._all_norms_sq + q_norm_sq - 2.0 * dots
        np.clip(L2_sq, 0.0, None, out=L2_sq)
        L2 = np.sqrt(L2_sq)

        L2_min = float(L2.min())
        shifted = np.maximum(L2 - L2_min, 0.0)
        return (shifted / self._L2_median_shift).astype(np.float32)

    # ── Nyström helpers ──────────────────────────────────────────────────

    def _nystrom_project_one(self, emb: np.ndarray) -> np.ndarray:
        """Nyström-embed a single point into raw MDS space."""
        emb64 = np.nan_to_num(
            emb, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float64).ravel()
        diff = emb64 - self._landmarks
        delta_sq = np.sum(diff ** 2, axis=1)
        row_mean = delta_sq.mean()
        b = -0.5 * (delta_sq - row_mean - self._col_means + self._grand_mean)
        return ((b @ self._V_k) * self._inv_sqrt_lam).astype(np.float32)

    def _nystrom_project_all(self) -> np.ndarray:
        """Nyström-embed all N library tracks into raw MDS space."""
        embs = self._all_embeddings.astype(np.float64)
        dots = embs @ self._landmarks.T
        D2 = (
            self._all_norms_sq[:, None]
            + self._lm_norms_sq[None, :]
            - 2.0 * dots
        )
        np.clip(D2, 0.0, None, out=D2)
        row_means = D2.mean(axis=1, keepdims=True)
        B = -0.5 * (D2 - row_means - self._col_means[None, :] + self._grand_mean)
        return ((B @ self._V_k) * self._inv_sqrt_lam[None, :]).astype(np.float32)

    # ── MDS basis computation ─────────────────────────────────────────────

    def _compute_mds_basis(self, emb_raw: np.ndarray) -> None:
        """MDS on K landmarks → Nyström directions for all N → cache.

        Computes shifted L2 normalization and auto-scales ppu.
        """
        t0 = time.monotonic()

        q = np.nan_to_num(
            emb_raw, nan=0.0, posinf=0.0, neginf=0.0
        ).reshape(1, -1).astype(np.float32)

        # K nearest neighbors + L2 distances
        # Request extra to cover dedup'd tracks, then remap to dedup indices
        n_search = min(self._neighborhood_k + 200, self._faiss_index.ntotal)
        dists_sq, idxs = self._faiss_index.search(q, n_search)

        # Remap FAISS indices → deduplicated indices, skip removed tracks
        neighbor_idxs_dedup = []
        L2_neighbors_list = []
        seen: set[int] = set()
        for dist, idx in zip(dists_sq[0], idxs[0]):
            if idx < 0:
                continue
            dedup_i = int(self._faiss_to_dedup[idx])
            if dedup_i < 0 or dedup_i in seen:
                continue
            seen.add(dedup_i)
            neighbor_idxs_dedup.append(dedup_i)
            L2_neighbors_list.append(float(np.sqrt(max(dist, 0.0))))
            if len(neighbor_idxs_dedup) >= self._neighborhood_k:
                break

        neighbor_idxs = np.array(neighbor_idxs_dedup, dtype=np.int64)
        L2_neighbors = np.array(L2_neighbors_list, dtype=np.float64)

        # L2 shift + normalization
        self._L2_min = float(L2_neighbors[0]) if len(L2_neighbors) > 0 else 0.0
        nb_shifted = L2_neighbors - self._L2_min
        nz = nb_shifted > 1e-8
        self._L2_median_shift = (
            float(np.median(nb_shifted[nz])) if nz.any() else 1.0
        )
        self._L2_median_shift = max(self._L2_median_shift, 1e-8)

        # ── Landmarks: query + K neighbors ────────────────────────────────
        landmarks = np.vstack(
            [q, self._all_embeddings[neighbor_idxs]]
        ).astype(np.float64)
        np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        L = landmarks.shape[0]

        lm_norms_sq = np.sum(landmarks ** 2, axis=1)
        D2 = lm_norms_sq[:, None] + lm_norms_sq[None, :] - 2.0 * (
            landmarks @ landmarks.T
        )
        np.clip(D2, 0.0, None, out=D2)

        # ── Classical MDS ─────────────────────────────────────────────────
        col_means  = D2.mean(axis=0)
        grand_mean = D2.mean()
        H = np.eye(L) - 1.0 / L
        B = -0.5 * H @ D2 @ H

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        lam3 = eigenvalues[-3:][::-1].copy()
        V_k  = eigenvectors[:, -3:][:, ::-1].copy()
        lam3 = np.maximum(lam3, 1e-10)

        self._landmarks    = landmarks
        self._lm_norms_sq  = lm_norms_sq
        self._V_k          = V_k
        self._inv_sqrt_lam = 1.0 / np.sqrt(lam3)
        self._col_means    = col_means
        self._grand_mean   = grand_mean

        # ── Nyström: all N tracks + query ─────────────────────────────────
        all_mds = self._nystrom_project_all()
        q_mds   = self._nystrom_project_one(emb_raw)

        # Build initial hybrid positions (directions × shifted L2)
        rel_mds = all_mds - q_mds
        mds_r   = np.linalg.norm(rel_mds, axis=1, keepdims=True)
        dirs    = (rel_mds / np.maximum(mds_r, 1e-8)).astype(np.float32)

        L2_shifted   = self._compute_L2_shifted(emb_raw)
        new_all_proj = dirs * L2_shifted[:, None]

        # ── Procrustes alignment ──────────────────────────────────────────
        if self._all_proj_smooth is not None:
            old_nb = self._all_proj_smooth[neighbor_idxs].astype(np.float64)
            new_nb = new_all_proj[neighbor_idxs].astype(np.float64)
            old_c = old_nb - old_nb.mean(axis=0)
            new_c = new_nb - new_nb.mean(axis=0)

            M = old_c.T @ new_c
            U_p, _S_p, Vt_p = np.linalg.svd(M)
            d = np.sign(np.linalg.det(Vt_p.T @ U_p.T))
            R = (Vt_p.T @ np.diag([1.0, 1.0, d]) @ U_p.T).astype(np.float32)

            new_all_proj = new_all_proj @ R
            # Rotate the stored MDS projections so per-cycle Nyström
            # queries produce coordinates in the aligned frame
            all_mds = all_mds @ R
            # Also rotate V_k so _nystrom_project_one produces aligned coords
            # nystrom output = (b @ V_k) * inv_sqrt_lam
            # rotated = output @ R = (b @ V_k) * inv_sqrt_lam @ R
            # = b @ (V_k * inv_sqrt_lam) @ R / inv_sqrt_lam ... messy
            # Instead, store a combined projection matrix:
            # _nystrom_mat = V_k @ diag(inv_sqrt_lam) @ R
            # output = b @ _nystrom_mat (already rotated)
            nystrom_mat = (self._V_k * self._inv_sqrt_lam[None, :]).astype(
                np.float64
            ) @ R.astype(np.float64)
            self._V_k = nystrom_mat  # now (L, 3) combined
            self._inv_sqrt_lam = np.ones(3, dtype=np.float64)  # absorbed into V_k

        # ── Store MDS projections for per-cycle direction computation ─────
        self._all_proj_mds = all_mds.astype(np.float32)
        self._all_proj     = new_all_proj


        if self._all_proj_smooth is None:
            self._all_proj_smooth = new_all_proj.copy()

        self._last_basis_emb  = emb_raw.copy()
        self._consecutive_far = 0

        # ── Diagnostic ────────────────────────────────────────────────────
        all_r    = np.linalg.norm(new_all_proj, axis=1)
        top1_idx = int(neighbor_idxs[0])
        top1_id  = self._id_list[top1_idx]
        top1_r   = float(all_r[top1_idx])
        n_closer = int((all_r < top1_r - 1e-6).sum())
        logger.info(
            "MDS hybrid recompute %.1fms  K=%d  top1=%s  r=%.3f  "
            "closer=%d  ppu=%.0f  L2_min=%.4f  med_shift=%.4f",
            (time.monotonic() - t0) * 1000,
            len(neighbor_idxs), top1_id, top1_r, n_closer,
            self._ppu, self._L2_min, self._L2_median_shift,
        )

    # ── Update (called on new UDP embedding, ~1/s) ────────────────────────

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

        # ── Recompute gate ────────────────────────────────────────────────
        if self._last_basis_emb is not None:
            dist_moved = float(np.linalg.norm(emb_clean - self._last_basis_emb))
            if dist_moved > self._recompute_threshold:
                self._consecutive_far += 1
            else:
                self._consecutive_far = 0
            needs_recompute = self._consecutive_far >= self._recompute_consecutive
        else:
            needs_recompute = True

        if needs_recompute:
            self._compute_mds_basis(emb_clean)
        elif self._all_proj_mds is not None:
            # Per-cycle: fresh directions from same MDS model × fresh L2
            q_mds = self._nystrom_project_one(emb_clean)
            rel_mds = self._all_proj_mds - q_mds
            mds_r = np.linalg.norm(rel_mds, axis=1, keepdims=True)
            dirs = rel_mds / np.maximum(mds_r, 1e-8)
            L2_shifted = self._compute_L2_shifted(emb_clean)
            self._all_proj = (dirs * L2_shifted[:, None]).astype(np.float32)


    # ── Per-frame tick ───────────────────────────────────────────────────

    def _tick(self, dt: float) -> None:
        if self._all_proj is not None and self._all_proj_smooth is not None:
            alpha = min(1.0, self._lerp_speed * dt)
            self._all_proj_smooth += alpha * (
                self._all_proj - self._all_proj_smooth
            )

        # Hover detection (clear if mouse is outside the window)
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
        depth = abs_z + self._z_near
        scale = self._focal_length / (depth + self._focal_length)
        sx    = (self._cx + rel[:, 0] * scale * self._ppu).astype(np.int32)
        sy    = (self._cy + rel[:, 1] * scale * self._ppu).astype(np.int32)
        tpx   = np.maximum(1, (self._base_thumb_px * scale + 0.5)).astype(
            np.int32
        )

        # On-screen mask
        half   = tpx // 2
        on_scr = (
            (sx + half >= 0) & (sx - half < self._canvas_w) &
            (sy + half >= 0) & (sy - half < self._canvas_h)
        )

        # Z-sort back to front
        vis   = np.where(on_scr)[0]
        order = vis[np.argsort(-depth[vis])]

        self._frame_vis    = vis
        self._frame_sx     = sx[vis]
        self._frame_sy     = sy[vis]
        self._frame_halfpx = half[vis]

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

    def _draw_cluster_labels(self) -> None:
        """Draw 3D-projected text labels for K-means clusters."""
        if not self._cluster_labels or len(self._cluster_pos_3d) == 0:
            return

        # Cache label font (one size — perspective handled via alpha only)
        if not hasattr(self, "_label_font"):
            self._label_font = pygame.font.SysFont("Arial", 14)

        for ci, label in enumerate(self._cluster_labels):
            if not label:
                continue
            pos = self._cluster_pos_3d[ci]
            abs_z = abs(float(pos[2]))
            depth = abs_z + self._z_near
            scale = self._focal_length / (depth + self._focal_length)
            lx = int(self._cx + pos[0] * scale * self._ppu)
            ly = int(self._cy + pos[1] * scale * self._ppu)

            # Off-screen check
            if lx < -100 or lx > self._canvas_w + 100:
                continue
            if ly < -20 or ly > self._canvas_h + 20:
                continue

            # Semi-transparent text — fades with distance
            alpha = max(30, min(200, int(220 * scale)))
            text_surf = self._label_font.render(label, True, (200, 200, 200))
            text_surf.set_alpha(alpha)
            tw, th = text_surf.get_size()
            self._screen.blit(text_surf, (lx - tw // 2, ly - th // 2))

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
