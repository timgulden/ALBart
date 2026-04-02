"""Build genre clusters from 25D UMAP embeddings and label them with Claude.

Computes k-means clusters on the 25D embedding space, stores centroids
and radii, then asks Claude to label each cluster based on sample tracks.

The result is saved to data/clusters.json and used by the mood filter
to map mood descriptions to track sets.

Usage:
    python3 tools/build_clusters.py
    python3 tools/build_clusters.py --k 40       # explicit cluster count
    python3 tools/build_clusters.py --force       # rebuild even if exists
    python3 tools/build_clusters.py --no-label    # skip Claude labeling
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.effects.database import DatabaseClient, DatabaseConfig
from albart.utils import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CLUSTERS_PATH = DATA_DIR / "clusters.json"

# Power-law exponent for auto-selecting k
ALPHA = 0.43


def compute_clusters(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run k-means and return (labels, centroids, mean_radii)."""
    from sklearn.cluster import KMeans

    logger.info("Running k-means with k=%d on %d tracks...", k, len(X))
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_.astype(np.float32)

    mean_radii = np.zeros(k, dtype=np.float32)
    for c in range(k):
        members = X[labels == c]
        dists = np.linalg.norm(members - centroids[c], axis=1)
        mean_radii[c] = float(dists.mean())

    return labels, centroids, mean_radii


def label_clusters_with_claude(
    cluster_samples: dict[int, list[str]],
    mean_radii: np.ndarray,
    sizes: list[int],
) -> list[str]:
    """Ask Claude to label each cluster based on sample tracks."""
    import anthropic

    # Build the prompt
    cluster_descriptions = []
    for c in sorted(cluster_samples.keys()):
        samples = cluster_samples[c]
        sample_text = "\n".join(f"    - {s}" for s in samples)
        cluster_descriptions.append(
            f"  Cluster {c} ({sizes[c]} tracks, radius={mean_radii[c]:.3f}):\n{sample_text}"
        )

    prompt = (
        "I have clustered a music library by audio similarity. Each cluster "
        "contains tracks that sound similar. Based on the sample tracks below, "
        "give each cluster a short genre/mood label (2-5 words) that describes "
        "what kind of music it contains.\n\n"
        "Return ONLY a JSON array of strings, one label per cluster, in order. "
        "No commentary, no markdown, just the JSON array.\n\n"
        "Clusters:\n" + "\n".join(cluster_descriptions)
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    labels = json.loads(text)
    if len(labels) != len(cluster_samples):
        logger.warning(
            "Claude returned %d labels for %d clusters — padding/truncating",
            len(labels), len(cluster_samples),
        )
        while len(labels) < len(cluster_samples):
            labels.append(f"Cluster {len(labels)}")
        labels = labels[:len(cluster_samples)]

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build genre clusters from 25D embeddings")
    parser.add_argument("--k", type=int, default=0,
                        help="Number of clusters (0 = auto from N^0.43)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if clusters.json exists")
    parser.add_argument("--no-label", action="store_true",
                        help="Skip Claude labeling (use placeholder labels)")
    args = parser.parse_args()

    if CLUSTERS_PATH.exists() and not args.force:
        data = json.loads(CLUSTERS_PATH.read_text())
        logger.info("Clusters already exist: %d clusters (%s). Use --force to rebuild.",
                     len(data["labels"]), CLUSTERS_PATH)
        return

    # Load 25D embeddings
    db = DatabaseClient(DatabaseConfig())
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT track_id, title, artist, umap_25d "
                "FROM tracks WHERE umap_25d IS NOT NULL"
            )
            rows = cur.fetchall()

    track_ids = [r[0] for r in rows]
    titles = [r[1] for r in rows]
    artists = [r[2] for r in rows]
    X = np.array([np.array(r[3], dtype=np.float32) for r in rows])
    N = len(X)
    logger.info("Loaded %d tracks with 25D embeddings", N)

    # Determine k
    k = args.k if args.k > 0 else max(10, int(N ** ALPHA))
    logger.info("Using k=%d (N=%d, alpha=%.2f)", k, N, ALPHA)

    # Cluster
    labels, centroids, mean_radii = compute_clusters(X, k)
    sizes = [int(np.sum(labels == c)) for c in range(k)]

    # Collect sample tracks per cluster (up to 8 for labeling)
    rng = np.random.RandomState(42)
    cluster_samples: dict[int, list[str]] = {}
    for c in range(k):
        members = np.where(labels == c)[0]
        sample_idx = rng.choice(members, size=min(8, len(members)), replace=False)
        cluster_samples[c] = [
            f"{titles[i]} — {artists[i]}" for i in sample_idx
        ]

    # Label
    if args.no_label:
        cluster_labels = [f"Cluster {c}" for c in range(k)]
    else:
        logger.info("Labeling clusters with Claude...")
        cluster_labels = label_clusters_with_claude(cluster_samples, mean_radii, sizes)

    # Print summary
    print(f"\n{'ID':>3} {'Label':40s} {'Size':>5} {'Radius':>7}")
    print("-" * 60)
    for c in range(k):
        print(f"{c:3d} {cluster_labels[c]:40s} {sizes[c]:5d} {mean_radii[c]:7.4f}")

    # Save
    data = {
        "k": k,
        "n_tracks": N,
        "alpha": ALPHA,
        "labels": cluster_labels,
        "sizes": sizes,
        "centroids": centroids.tolist(),
        "mean_radii": mean_radii.tolist(),
        "samples": cluster_samples,
    }
    CLUSTERS_PATH.write_text(json.dumps(data, indent=2))
    logger.info("Saved → %s", CLUSTERS_PATH)


if __name__ == "__main__":
    main()
