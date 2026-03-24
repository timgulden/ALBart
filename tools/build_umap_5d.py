"""Compute 5-D UMAP embedding of all library tracks.

Saves to data/umap_5d.npy (N, 5) float32 — positions for the neighborhood
display's hybrid projection.  Also saves data/umap_5d_ids.npy for ID mapping.

Usage:
    python tools/build_umap_5d.py
    python tools/build_umap_5d.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.utils import DATA_DIR, load_config

EMBEDDINGS_PATH = DATA_DIR / "embeddings_norm.npy"
IDS_PATH            = DATA_DIR / "faiss_raw_ids.npy"
OUTPUT_PATH         = DATA_DIR / "umap_5d.npy"
OUTPUT_IDS_PATH     = DATA_DIR / "umap_5d_ids.npy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 5-D UMAP embedding of all library tracks"
    )
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if output exists")
    args = parser.parse_args()

    if OUTPUT_PATH.exists() and not args.force:
        data = np.load(str(OUTPUT_PATH))
        print(f"Already exists: {OUTPUT_PATH} ({data.shape})")
        print("Use --force to rebuild.")
        return

    config = load_config()
    umap_cfg = config.get("umap", {})

    print("Loading embeddings...")
    embeddings = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)
    ids = np.load(str(IDS_PATH), allow_pickle=True)
    print(f"  {embeddings.shape[0]} tracks, {embeddings.shape[1]}D")

    import umap
    print(f"Computing 5-D UMAP (n_neighbors={umap_cfg.get('n_neighbors', 15)}, "
          f"min_dist={umap_cfg.get('min_dist', 0.1)}, "
          f"metric={umap_cfg.get('metric', 'cosine')})...")

    reducer = umap.UMAP(
        n_components=5,
        n_neighbors=int(umap_cfg.get("n_neighbors", 15)),
        min_dist=float(umap_cfg.get("min_dist", 0.1)),
        metric=str(umap_cfg.get("metric", "cosine")),
        random_state=42,
        verbose=True,
    )
    coords_5d = reducer.fit_transform(embeddings).astype(np.float32)
    print(f"Result shape: {coords_5d.shape}")

    np.save(str(OUTPUT_PATH), coords_5d)
    np.save(str(OUTPUT_IDS_PATH), ids)
    print(f"Saved → {OUTPUT_PATH}")
    print(f"Saved → {OUTPUT_IDS_PATH}")

    # Quick stats
    print(f"\nCoordinate ranges:")
    for d in range(5):
        print(f"  dim {d}: [{coords_5d[:, d].min():.3f}, {coords_5d[:, d].max():.3f}]  "
              f"std={coords_5d[:, d].std():.3f}")


if __name__ == "__main__":
    main()
