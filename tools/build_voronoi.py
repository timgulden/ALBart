"""
Build Voronoi cluster regions from UMAP 2D positions.

Steps:
  1. K-means cluster the normalized UMAP 2D positions into N regions
  2. For each cluster, sample representative tracks (weighted by proximity to centroid)
  3. Label each cluster via Claude API (claude-sonnet-4-6)
  4. Compute Voronoi polygons from centroids, clipped to [0, 1]²
  5. Save data/voronoi_regions.json

Usage:
    python tools/build_voronoi.py
    python tools/build_voronoi.py --n-clusters 30
    python tools/build_voronoi.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.pipeline.database import DB_PATH, get_connection
from albart.utils import DATA_DIR

VORONOI_PATH = DATA_DIR / "voronoi_regions.json"
UMAP_2D_PATH = DATA_DIR / "umap_2d.npy"
UMAP_IDS_PATH = DATA_DIR / "umap_ids.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Voronoi cluster map for music map display")
    parser.add_argument("--n-clusters", type=int, default=30,
                        help="Number of Voronoi regions (default: 30)")
    parser.add_argument("--n-sample", type=int, default=10,
                        help="Tracks per cluster sent to Claude for labeling (default: 10)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if output file already exists")
    args = parser.parse_args()

    if VORONOI_PATH.exists() and not args.force:
        print(f"Already exists: {VORONOI_PATH}  (use --force to rebuild)")
        sys.exit(0)

    # Load UMAP coordinates
    umap_2d  = np.load(str(UMAP_2D_PATH))
    umap_ids = np.load(str(UMAP_IDS_PATH), allow_pickle=True)
    print(f"Loaded {len(umap_2d)} tracks from UMAP")

    # Normalize to [0, 1]² (same transform used in map_display.py)
    xy_min   = umap_2d.min(axis=0)
    xy_range = umap_2d.max(axis=0) - xy_min
    umap_norm = (umap_2d - xy_min) / (xy_range + 1e-8)

    # K-means clustering on normalized 2D positions
    from sklearn.cluster import KMeans
    print(f"K-means: {args.n_clusters} clusters...")
    km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(umap_norm)
    centroids = km.cluster_centers_   # shape (n_clusters, 2), normalized

    # Load track metadata from DB
    conn = get_connection(DB_PATH)
    rows = conn.execute("SELECT track_id, title, artist FROM tracks").fetchall()
    conn.close()
    track_meta = {r["track_id"]: (r["title"] or "", r["artist"] or "") for r in rows}

    # Sample representative tracks per cluster (inverse-distance² weighting)
    rng = np.random.default_rng(42)
    cluster_samples: list[list[str]] = []
    for i in range(args.n_clusters):
        mask     = cluster_labels == i
        c_ids    = umap_ids[mask]
        c_pts    = umap_norm[mask]
        centroid = centroids[i]

        dists   = np.linalg.norm(c_pts - centroid, axis=1)
        weights = 1.0 / (dists ** 2 + 0.01)
        weights /= weights.sum()

        n      = min(args.n_sample, len(c_ids))
        chosen = rng.choice(len(c_ids), size=n, replace=False, p=weights)
        tracks = []
        for idx in chosen:
            tid  = str(c_ids[idx])
            meta = track_meta.get(tid, ("Unknown", "Unknown"))
            tracks.append(f"{meta[0]} — {meta[1]}")
        cluster_samples.append(tracks)

    # Label each cluster via Claude API
    print("Labeling clusters via Claude API...")
    label_texts = _label_clusters(cluster_samples)
    print(f"Received {len(label_texts)} labels")

    # Voronoi diagram from centroids, clipped to [0, 1]²
    from shapely.geometry import MultiPoint, Point, box
    from shapely.ops import voronoi_diagram as shapely_voronoi

    print("Computing Voronoi polygons...")
    bounding_box = box(0.0, 0.0, 1.0, 1.0)
    diagram      = shapely_voronoi(MultiPoint([tuple(c) for c in centroids]),
                                   envelope=bounding_box)

    # Match each Voronoi polygon to its centroid (shapely doesn't preserve order)
    regions: list[dict] = []
    matched: set[int]   = set()
    for poly in diagram.geoms:
        clipped = poly.intersection(bounding_box)
        if clipped.is_empty:
            continue
        for i, c in enumerate(centroids):
            if i in matched:
                continue
            if clipped.contains(Point(c[0], c[1])):
                coords = [[float(x), float(y)] for x, y in clipped.exterior.coords]
                regions.append({
                    "cluster_id": i,
                    "label":      label_texts[i] if i < len(label_texts) else f"Region {i+1}",
                    "centroid":   [float(c[0]), float(c[1])],
                    "polygon":    coords,
                    "n_tracks":   int((cluster_labels == i).sum()),
                })
                matched.add(i)
                break

    print(f"Matched {len(regions)} / {args.n_clusters} regions")

    with open(VORONOI_PATH, "w") as f:
        json.dump(regions, f, indent=2)
    print(f"Saved → {VORONOI_PATH}")

    for r in sorted(regions, key=lambda x: x["cluster_id"]):
        print(f"  [{r['cluster_id']:2d}]  {r['n_tracks']:4d} tracks  {r['label']}")


def _label_clusters(cluster_samples: list[list[str]]) -> list[str]:
    """Send all clusters to Claude in one call; parse numbered list response."""
    import anthropic

    lines = [
        "Below are groups of music tracks. For each group, provide a very brief label "
        "(2-4 words, no punctuation, no markdown) describing the musical style, mood, "
        "or genre that unites them. Be specific — differentiate each group from the others.\n"
    ]
    for i, tracks in enumerate(cluster_samples, 1):
        lines.append(f"Group {i}:")
        for t in tracks:
            lines.append(f"  - {t}")
        lines.append("")
    lines.append("Return only a numbered list, one label per group, nothing else.")

    prompt = "\n".join(lines)

    client   = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    matches = re.findall(r"^\d+\.\s*(.+)$", text, re.MULTILINE)
    return [m.strip() for m in matches]


if __name__ == "__main__":
    main()
