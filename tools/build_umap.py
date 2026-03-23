#!/usr/bin/env python3
"""
Build UMAP 2D projection from the raw FAISS index embeddings.

Extracts all 512-dim CLAP vectors from the raw FAISS index, fits a 2D UMAP
model, and saves the projected coordinates and fitted model for runtime use.

Outputs (written to data/):
    umap_2d.npy       — (N, 2) float32 projected coordinates
    umap_ids.npy      — (N,) track ID strings, aligned with umap_2d rows
    umap_model.joblib — fitted UMAP model; call model.transform(emb) at runtime

Usage:
    python tools/build_umap.py
    python tools/build_umap.py --n-neighbors 10 --min-dist 0.05
    python tools/build_umap.py --force   # rebuild even if outputs exist
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from albart.pipeline.embedder import FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH
from albart.utils import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

UMAP_2D_PATH    = DATA_DIR / "umap_2d.npy"
UMAP_IDS_PATH   = DATA_DIR / "umap_ids.npy"
UMAP_MODEL_PATH = DATA_DIR / "umap_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UMAP 2D projection")
    parser.add_argument("--n-neighbors", type=int,   default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist",    type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--metric",      type=str,   default="cosine",
                        help="UMAP metric — cosine or euclidean (default: cosine)")
    parser.add_argument("--force",       action="store_true",
                        help="Rebuild even if outputs already exist")
    args = parser.parse_args()

    if not args.force and UMAP_2D_PATH.exists() and UMAP_MODEL_PATH.exists():
        logger.info("UMAP outputs already exist at %s — use --force to rebuild.", DATA_DIR)
        return

    # ── Load FAISS raw index ──────────────────────────────────────────────────
    import faiss  # lazy — avoids BLAS conflict on macOS

    if not FAISS_RAW_INDEX_PATH.exists():
        logger.error("Raw FAISS index not found: %s", FAISS_RAW_INDEX_PATH)
        logger.error("Run the pipeline first: python -m albart.pipeline.run_pipeline")
        sys.exit(1)

    logger.info("Loading FAISS raw index from %s", FAISS_RAW_INDEX_PATH)
    index = faiss.read_index(str(FAISS_RAW_INDEX_PATH))
    n, d = index.ntotal, index.d
    logger.info("Index: %d vectors, dim=%d", n, d)

    # Extract all vectors from the flat index
    logger.info("Extracting embedding vectors...")
    vectors = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        index.reconstruct(i, vectors[i])

    # Load matching track IDs
    track_ids = np.load(str(FAISS_RAW_IDS_PATH), allow_pickle=True)
    assert len(track_ids) == n, f"ID count mismatch: {len(track_ids)} IDs vs {n} vectors"

    # ── Fit UMAP ─────────────────────────────────────────────────────────────
    import umap as umap_lib

    logger.info(
        "Fitting UMAP (n=%d, n_neighbors=%d, min_dist=%.3f, metric=%s)...",
        n, args.n_neighbors, args.min_dist, args.metric,
    )
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=42,
        verbose=True,
    )
    coords_2d = reducer.fit_transform(vectors).astype(np.float32)

    # ── Save outputs ─────────────────────────────────────────────────────────
    np.save(str(UMAP_2D_PATH),  coords_2d)
    np.save(str(UMAP_IDS_PATH), track_ids)
    joblib.dump(reducer, str(UMAP_MODEL_PATH))

    logger.info("Saved: %s", UMAP_2D_PATH)
    logger.info("Saved: %s", UMAP_IDS_PATH)
    logger.info("Saved: %s", UMAP_MODEL_PATH)
    logger.info(
        "Coordinate ranges: x=[%.3f, %.3f]  y=[%.3f, %.3f]",
        coords_2d[:, 0].min(), coords_2d[:, 0].max(),
        coords_2d[:, 1].min(), coords_2d[:, 1].max(),
    )


if __name__ == "__main__":
    main()
