"""Diagnostic: test whether MDS + Nyström places the nearest neighbor
closest to the query in 3-D projected space.

Uses the actual FAISS index and embeddings — no pygame, no UDP.
Picks several real embeddings as simulated queries and reports:
  - Where the raw FAISS #1 lands in 3D (should be smallest radius)
  - How many library tracks project closer than the #1
  - Screen coordinates at various zoom levels
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss  # noqa: E402
from albart.pipeline.embedder import FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH
from albart.utils import DATA_DIR

EMBEDDINGS_RAW_PATH = DATA_DIR / "embeddings_raw.npy"

K = 100           # neighborhood size
CANVAS_W = 1920
CANVAS_H = 1080
FOCAL = 2.0
Z_NEAR = 0.5


def load_data():
    print("Loading FAISS index + embeddings...")
    index = faiss.read_index(str(FAISS_RAW_INDEX_PATH))
    ids = np.load(str(FAISS_RAW_IDS_PATH), allow_pickle=True)
    id_list = [str(t) for t in ids]
    embeddings = np.load(str(EMBEDDINGS_RAW_PATH)).astype(np.float32)
    print(f"  {len(id_list)} tracks, embedding dim={embeddings.shape[1]}")
    return index, id_list, embeddings


def mds_nystrom(query: np.ndarray, index, embeddings, id_list):
    """Run MDS + Nyström and return diagnostics."""
    N = len(id_list)
    q = query.reshape(1, -1).astype(np.float32)

    # FAISS search
    dists_faiss, idxs = index.search(q, K)
    neighbor_idxs = idxs[0][idxs[0] >= 0]
    faiss_top1_idx = int(neighbor_idxs[0])
    faiss_top1_id = id_list[faiss_top1_idx]
    faiss_top1_l2 = float(np.sqrt(dists_faiss[0][0]))  # FAISS returns squared L2

    # Landmarks
    landmarks = np.vstack([q, embeddings[neighbor_idxs]]).astype(np.float64)
    L = landmarks.shape[0]

    # Landmark pairwise D²
    lm_norms = np.sum(landmarks ** 2, axis=1)
    D2 = lm_norms[:, None] + lm_norms[None, :] - 2.0 * (landmarks @ landmarks.T)
    np.clip(D2, 0.0, None, out=D2)

    # Classical MDS
    col_means = D2.mean(axis=0)
    grand_mean = D2.mean()
    H = np.eye(L) - 1.0 / L
    B = -0.5 * H @ D2 @ H

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    lam3 = eigenvalues[-3:][::-1].copy()
    V_k = eigenvectors[:, -3:][:, ::-1].copy()
    lam3 = np.maximum(lam3, 1e-10)
    inv_sqrt_lam = 1.0 / np.sqrt(lam3)

    # MDS landmark coords (for normalization)
    X_L = V_k * np.sqrt(lam3)[None, :]
    neighbor_rel = X_L[1:] - X_L[0:1]
    radii_lm = np.linalg.norm(neighbor_rel, axis=1)
    p25 = max(float(np.percentile(radii_lm, 25)), 1e-8)
    proj_scale = 1.0 / p25

    # Verify: MDS distance of landmark #1 (FAISS nearest) from query
    mds_top1_r_unnorm = float(np.linalg.norm(X_L[1] - X_L[0]))
    mds_top1_r_norm = mds_top1_r_unnorm * proj_scale

    print(f"\n  MDS landmark check:")
    print(f"    FAISS #1 L2 dist in 512D:  {faiss_top1_l2:.6f}")
    print(f"    MDS  #1 dist (unnorm):     {mds_top1_r_unnorm:.6f}")
    print(f"    MDS  #1 dist (normalized): {mds_top1_r_norm:.4f}")
    print(f"    p25 (unnorm):              {p25:.6f}")
    print(f"    Eigenvalues (top 3):       {lam3}")
    print(f"    Variance captured:         "
          f"{lam3.sum() / np.maximum(eigenvalues, 0).sum():.1%}")

    # MDS distances of ALL landmarks from query
    lm_radii_norm = radii_lm * proj_scale
    print(f"\n  Landmark radii (normalized): "
          f"min={lm_radii_norm.min():.3f}  "
          f"p25={np.percentile(lm_radii_norm, 25):.3f}  "
          f"median={np.median(lm_radii_norm):.3f}  "
          f"max={lm_radii_norm.max():.3f}")

    # ── Nyström: project query ────────────────────────────────────────
    def nystrom_one(emb):
        emb64 = emb.astype(np.float64).ravel()
        diff = emb64 - landmarks
        delta_sq = np.sum(diff ** 2, axis=1)
        row_mean = delta_sq.mean()
        b = -0.5 * (delta_sq - row_mean - col_means + grand_mean)
        return ((b @ V_k) * inv_sqrt_lam * proj_scale).astype(np.float32)

    q_3d = nystrom_one(query)

    # Verify: Nyström of query should ≈ MDS landmark 0 (scaled)
    q_mds = (X_L[0] * proj_scale).astype(np.float32)
    print(f"\n  Nyström self-consistency:")
    print(f"    query MDS coord:     {q_mds}")
    print(f"    query Nyström coord: {q_3d}")
    print(f"    difference:          {np.linalg.norm(q_3d - q_mds):.6f}")

    # ── Nyström: project ALL tracks ───────────────────────────────────
    embs64 = embeddings.astype(np.float64)
    lm_norms_sq = np.sum(landmarks ** 2, axis=1)
    all_norms_sq = np.sum(embs64 ** 2, axis=1)
    dots = embs64 @ landmarks.T
    D2_new = all_norms_sq[:, None] + lm_norms_sq[None, :] - 2.0 * dots
    np.clip(D2_new, 0.0, None, out=D2_new)
    row_means = D2_new.mean(axis=1, keepdims=True)
    B_new = -0.5 * (D2_new - row_means - col_means[None, :] + grand_mean)
    all_proj = ((B_new @ V_k) * inv_sqrt_lam[None, :] * proj_scale).astype(np.float32)

    # ── HYBRID: MDS direction × actual L2 distance ─────────────────────
    print(f"\n  --- HYBRID projection (MDS direction × L2 distance) ---")

    # Query MDS position
    q_3d = nystrom_one(query)

    # MDS directions
    rel_mds = all_proj - q_3d                                    # (N, 3)
    mds_r = np.linalg.norm(rel_mds, axis=1, keepdims=True)      # (N, 1)
    direction = rel_mds / np.maximum(mds_r, 1e-8)               # (N, 3) unit

    # Actual 512D L2 distances
    q64 = query.astype(np.float64).ravel()
    q_nsq = float(np.dot(q64, q64))
    all_nsq = np.sum(embeddings.astype(np.float64) ** 2, axis=1)
    dots_all = embeddings.astype(np.float64) @ q64
    L2_sq_all = all_nsq + q_nsq - 2.0 * dots_all
    np.clip(L2_sq_all, 0.0, None, out=L2_sq_all)
    L2_all = np.sqrt(L2_sq_all)

    # Normalize by p25 of non-zero FAISS neighbor distances
    L2_faiss = np.sqrt(np.maximum(dists_faiss[0][:len(neighbor_idxs)], 0.0))
    nz = L2_faiss > 1e-8
    L2_p25 = float(np.percentile(L2_faiss[nz], 25)) if nz.any() else 1.0
    L2_p25 = max(L2_p25, 1e-8)
    L2_norm = L2_all / L2_p25

    # Hybrid positions
    hybrid = direction * L2_norm[:, None]                        # (N, 3)
    hybrid_r = np.linalg.norm(hybrid, axis=1)

    top1_hybrid = hybrid[faiss_top1_idx]
    top1_hr = float(hybrid_r[faiss_top1_idx])
    n_closer_h = int((hybrid_r < top1_hr - 1e-6).sum())
    closest_h_idx = int(np.argmin(hybrid_r))

    print(f"    FAISS #1 ({faiss_top1_id}):")
    print(f"      hybrid pos:  [{top1_hybrid[0]:.4f}, {top1_hybrid[1]:.4f}, {top1_hybrid[2]:.4f}]")
    print(f"      hybrid r:    {top1_hr:.4f}")
    print(f"      L2 norm'd:   {float(L2_norm[faiss_top1_idx]):.4f}")
    print(f"    Tracks closer than FAISS #1 in hybrid: {n_closer_h}")

    # Top-10 by hybrid radius
    print(f"\n  Top-10 by hybrid radius:")
    order_h = np.argsort(hybrid_r)[:10]
    print(f"    {'rank':>7} {'track_id':>24} {'hybrid_r':>8} {'512D_L2':>10} {'FAISS_rank':>10}")
    for rank, idx in enumerate(order_h):
        tid = id_list[idx]
        rh = float(hybrid_r[idx])
        l2 = float(L2_all[idx])
        faiss_rank = "not in K"
        for fr, fi in enumerate(neighbor_idxs):
            if fi == idx:
                faiss_rank = str(fr + 1)
                break
        print(f"    {rank+1:>7} {tid:>24} {rh:>8.4f} {l2:>10.6f} {faiss_rank:>10}")

    # Screen coordinates
    print(f"\n  Screen projection of FAISS #1 (hybrid):")
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    for zoom in [1.0, 2.0, 5.0]:
        ppu = min(CANVAS_W, CANVAS_H) / 2.0 * zoom
        abs_z = abs(float(top1_hybrid[2]))
        depth = abs_z + Z_NEAR
        scale_val = FOCAL / (depth + FOCAL)
        sx = cx + float(top1_hybrid[0]) * scale_val * ppu
        sy = cy + float(top1_hybrid[1]) * scale_val * ppu
        tpx = int(80 * scale_val)
        dist_from_center = np.sqrt((sx - cx)**2 + (sy - cy)**2)
        print(f"    zoom={zoom:.0f}: screen=({sx:.0f}, {sy:.0f})  "
              f"dist_from_center={dist_from_center:.0f}px  "
              f"thumb={tpx}px  "
              f"{'ON SCREEN' if 0 <= sx < CANVAS_W and 0 <= sy < CANVAS_H else 'OFF SCREEN'}")


def main():
    index, id_list, embeddings = load_data()
    N = len(id_list)

    # Test with several different "query" embeddings
    rng = np.random.default_rng(42)
    test_indices = rng.choice(N, size=5, replace=False)

    for i, qi in enumerate(test_indices):
        tid = id_list[qi]
        print(f"\n{'='*70}")
        print(f"Test {i+1}: query = track {tid}")
        print(f"{'='*70}")
        mds_nystrom(embeddings[qi], index, embeddings, id_list)

    # Test with simulated "live audio" (library track + noise)
    print(f"\n{'='*70}")
    print("Test 6: SIMULATED LIVE — track 0 + noise (σ=0.03)")
    print(f"{'='*70}")
    base = embeddings[0].astype(np.float64)
    noisy = (base + rng.normal(0, 0.03, size=512)).astype(np.float32)
    mds_nystrom(noisy, index, embeddings, id_list)

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
