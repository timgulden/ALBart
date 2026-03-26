"""Test: hybrid MDS direction × shifted L2 distance, ALL 5192 tracks.

MDS on K=100 landmarks for directions, Nyström extends to all N tracks.
Shifted L2 (nearest at 0) for radial distance. |z| perspective.
Many tracks off screen, distant ones shrink to 1px dots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

np.seterr(all="ignore")

import faiss  # noqa: E402
import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from albart.pipeline.database import DB_PATH, get_connection  # noqa: E402
from albart.pipeline.embedder import FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH  # noqa: E402
from albart.utils import DATA_DIR  # noqa: E402

EMBEDDINGS_RAW_PATH = DATA_DIR / "embeddings_raw.npy"
CANVAS_W, CANVAS_H = 1920, 1080
CX, CY = CANVAS_W // 2, CANVAS_H // 2
BASE_THUMB = 80
FOCAL, Z_NEAR = 2.0, 0.5
K = 100


def main() -> None:
    print("Loading data...")
    index = faiss.read_index(str(FAISS_RAW_INDEX_PATH))
    ids_arr = np.load(str(FAISS_RAW_IDS_PATH), allow_pickle=True)
    id_list = [str(t) for t in ids_arr]
    emb = np.load(str(EMBEDDINGS_RAW_PATH)).astype(np.float32)
    N = len(id_list)

    conn = get_connection(DB_PATH)
    rows = conn.execute(
        "SELECT track_id, title, artist, art_path_original, art_path_32 "
        "FROM tracks"
    ).fetchall()
    conn.close()
    db = {r["track_id"]: r for r in rows}

    # ── Query: AC/DC Back In Black + noise ────────────────────────────
    rng = np.random.default_rng(42)
    query_base = emb[42].astype(np.float64)
    query = (query_base + rng.normal(0, 0.03, 512)).astype(np.float32)
    q64 = query.astype(np.float64)
    qid = id_list[42]
    row = db.get(qid)
    qname = f"{row['title']} — {row['artist']}" if row else qid
    print(f"Query: {qname}")

    # ── L2 distances from query to ALL tracks ─────────────────────────
    all_nsq = np.sum(emb.astype(np.float64) ** 2, axis=1)
    q_nsq = float(np.dot(q64, q64))
    dots = emb.astype(np.float64) @ q64
    L2 = np.sqrt(np.clip(all_nsq + q_nsq - 2.0 * dots, 0, None))

    faiss_d, faiss_i = index.search(query.reshape(1, -1), K)
    nb_idxs = faiss_i[0][faiss_i[0] >= 0]
    L2_neighbors = np.sqrt(np.maximum(faiss_d[0][:len(nb_idxs)], 0))
    print(f"FAISS #1: {id_list[nb_idxs[0]]}  L2={L2_neighbors[0]:.4f}")

    # ── MDS on K+1 landmarks ─────────────────────────────────────────
    print("Computing MDS...")
    q32 = query.reshape(1, -1)
    landmarks = np.vstack([q32, emb[nb_idxs]]).astype(np.float64)
    Lc = landmarks.shape[0]
    lm_n = np.sum(landmarks ** 2, axis=1)
    D2 = np.clip(lm_n[:, None] + lm_n[None, :] - 2 * landmarks @ landmarks.T,
                 0, None)
    H = np.eye(Lc) - 1.0 / Lc
    B = -0.5 * H @ D2 @ H
    ev, ec = np.linalg.eigh(B)
    lam3 = np.maximum(ev[-3:][::-1], 1e-10)
    Vk = ec[:, -3:][:, ::-1]
    isl = 1 / np.sqrt(lam3)
    cm = D2.mean(axis=0)
    gm = D2.mean()
    print(f"Variance captured: {lam3.sum() / np.maximum(ev, 0).sum():.1%}")

    # ── Nyström: project ALL N tracks ─────────────────────────────────
    print("Nyström projecting all tracks...")
    e64 = emb.astype(np.float64)
    lm_nsq = np.sum(landmarks ** 2, axis=1)
    a_nsq = np.sum(e64 ** 2, axis=1)
    D2n = np.clip(a_nsq[:, None] + lm_nsq[None, :] - 2 * e64 @ landmarks.T,
                  0, None)
    rm = D2n.mean(axis=1, keepdims=True)
    Bn = -0.5 * (D2n - rm - cm[None, :] + gm)
    all_mds = ((Bn @ Vk) * isl[None, :]).astype(np.float32)

    # Query MDS position
    dq = q64 - landmarks
    dsq = np.sum(dq ** 2, axis=1)
    qm = dsq.mean()
    bq = -0.5 * (dsq - qm - cm + gm)
    q_mds = ((bq @ Vk) * isl).astype(np.float32)

    # Unit directions from query
    rel_mds = all_mds - q_mds
    mds_r = np.linalg.norm(rel_mds, axis=1, keepdims=True)
    dirs = rel_mds / np.maximum(mds_r, 1e-8)

    # ── Hybrid: MDS direction × shifted L2 ────────────────────────────
    L2_min = float(L2.min())
    L2_shifted = L2 - L2_min
    # Normalize: median of K-neighbor shifted distances → 1.0
    nb_shifted = np.sort(L2_shifted)[:K]
    nz = nb_shifted > 1e-8
    L2_med = float(np.median(nb_shifted[nz])) if nz.any() else 1.0
    L2_norm = L2_shifted / max(L2_med, 1e-8)

    hybrid = dirs * L2_norm[:, None].astype(np.float32)
    print(f"L2_min={L2_min:.4f}  L2_med_shifted={L2_med:.4f}")

    # ── |z| perspective projection ────────────────────────────────────
    # Auto-zoom: K-th neighbor at ~400px
    r_Kth = float(np.sort(np.linalg.norm(hybrid, axis=1))[K])
    abs_z_Kth = abs(float(hybrid[np.argsort(np.linalg.norm(hybrid, axis=1))[K], 2]))
    scale_Kth = FOCAL / (abs_z_Kth + Z_NEAR + FOCAL)
    # Estimate typical screen displacement for the K-th track
    ppu = 580.0 / max(r_Kth * scale_Kth * 0.7, 1e-8)  # target ~500 on screen
    print(f"Auto ppu={ppu:.0f}")

    abs_z = np.abs(hybrid[:, 2])
    depth = abs_z + Z_NEAR
    scale = FOCAL / (depth + FOCAL)
    sx = (CX + hybrid[:, 0] * scale * ppu).astype(np.int32)
    sy = (CY + hybrid[:, 1] * scale * ppu).astype(np.int32)
    tpx = np.maximum(1, (BASE_THUMB * scale + 0.5)).astype(np.int32)

    # On-screen mask (include 1px dots)
    on_scr = (
        (sx >= 0) & (sx < CANVAS_W) &
        (sy >= 0) & (sy < CANVAS_H)
    )
    print(f"On screen: {on_scr.sum()} / {N}")

    # ── Render ────────────────────────────────────────────────────────
    print("Rendering...")
    pygame.init()
    screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
    screen.fill((0, 0, 0))

    vis = np.where(on_scr)[0]
    draw_order = vis[np.argsort(-depth[vis])]  # back to front

    # Pre-load PIL images for speed
    pil_cache: dict[str, Image.Image] = {}
    for tid in id_list:
        row = db.get(tid)
        if row:
            path = row["art_path_original"] or row["art_path_32"]
            if path:
                try:
                    pil_cache[tid] = Image.open(DATA_DIR / path).convert("RGB")
                except Exception:
                    pass

    drawn_art = 0
    drawn_dot = 0
    faiss1_global = int(nb_idxs[0])

    for i in draw_order:
        tid = id_list[i]
        sz = int(tpx[i])
        x, y = int(sx[i]), int(sy[i])

        if sz < 2:
            # 1px dot
            screen.set_at((x, y), (120, 120, 120))
            drawn_dot += 1
            continue

        pil_img = pil_cache.get(tid)
        if pil_img is None:
            c = max(40, min(180, sz * 2))
            pygame.draw.circle(screen, (c, c, c), (x, y), max(1, sz // 4))
            drawn_dot += 1
            continue

        sz = max(2, sz)
        try:
            if i == faiss1_global:
                sz = max(sz, 100)
            resized = pil_img.resize((sz, sz), Image.LANCZOS)
            arr = np.array(resized, dtype=np.uint8)
            surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
            if i == faiss1_global:
                pygame.draw.rect(surf, (0, 255, 0), (0, 0, sz, sz), 4)
            h = sz // 2
            screen.blit(surf, (x - h, y - h))
            drawn_art += 1
        except Exception:
            drawn_dot += 1

    # Sphere
    pygame.draw.circle(screen, (180, 180, 210), (CX, CY), 20)

    font = pygame.font.SysFont("Arial", 16)
    screen.blit(font.render(f"Query: {qname}", True, (255, 255, 255)), (10, 10))
    screen.blit(font.render(
        f"Green=FAISS#1 | {drawn_art} covers + {drawn_dot} dots | "
        f"{on_scr.sum()}/{N} on screen",
        True, (200, 200, 200)), (10, 32))

    pygame.display.flip()

    out = DATA_DIR / "test_render_noisy.png"
    pygame.image.save(screen, str(out))
    print(f"\nSaved → {out}")
    print(f"Drew {drawn_art} covers + {drawn_dot} dots")

    # Screen stats for FAISS #1
    sd = np.sqrt((sx[faiss1_global] - CX)**2 + (sy[faiss1_global] - CY)**2)
    print(f"FAISS #1: screen=({sx[faiss1_global]},{sy[faiss1_global]})  "
          f"dist={sd:.0f}px  thumb={tpx[faiss1_global]}px")

    print("\nPress any key to exit...")
    while True:
        for ev in pygame.event.get():
            if ev.type in (pygame.QUIT, pygame.KEYDOWN):
                pygame.quit()
                return


if __name__ == "__main__":
    main()
