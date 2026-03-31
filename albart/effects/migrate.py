"""One-time migration: SQLite + .npy + FAISS → PostgreSQL + pgvector.

Usage:
    python -m albart.effects.migrate                    # run migration
    python -m albart.effects.migrate --verify           # verify only
    python -m albart.effects.migrate --verify --recall   # verify + recall test

Reads:
    data/db.sqlite              — track metadata (5201 rows)
    data/embeddings_norm.npy    — (5192, 512) L2-normalised CLAP embeddings
    data/faiss_norm_ids.npy     — (5192,) track IDs for embedding rows
    data/umap_5d.npy            — (5192, 5) UMAP 5D projections
    data/umap_5d_ids.npy        — (5192,) track IDs for 5D rows
    data/umap_2d.npy            — (5191, 2) UMAP 2D projections
    data/umap_ids.npy           — (5191,) track IDs for 2D rows

Writes:
    PostgreSQL albart.tracks    — unified table with vector columns
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from albart.effects.database import DatabaseClient, DatabaseConfig
from albart.utils import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("albart.migrate")

# Source file paths
SQLITE_PATH = DATA_DIR / "db.sqlite"
EMBEDDINGS_PATH = DATA_DIR / "embeddings_norm.npy"
EMB_IDS_PATH = DATA_DIR / "faiss_norm_ids.npy"
UMAP_5D_PATH = DATA_DIR / "umap_5d.npy"
UMAP_5D_IDS_PATH = DATA_DIR / "umap_5d_ids.npy"
UMAP_2D_PATH = DATA_DIR / "umap_2d.npy"
UMAP_2D_IDS_PATH = DATA_DIR / "umap_ids.npy"


def _load_sqlite() -> list[dict]:
    """Load all tracks from SQLite."""
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM tracks").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_embeddings() -> dict[str, np.ndarray]:
    """Load 512D embeddings keyed by track_id."""
    embs = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)
    ids = np.load(str(EMB_IDS_PATH), allow_pickle=True)
    return {str(tid): embs[i] for i, tid in enumerate(ids)}


def _load_umap_5d() -> dict[str, np.ndarray]:
    """Load 5D UMAP projections keyed by track_id."""
    if not UMAP_5D_PATH.exists() or not UMAP_5D_IDS_PATH.exists():
        logger.warning("5D UMAP files not found — skipping")
        return {}
    umap = np.load(str(UMAP_5D_PATH)).astype(np.float32)
    ids = np.load(str(UMAP_5D_IDS_PATH), allow_pickle=True)
    return {str(tid): umap[i] for i, tid in enumerate(ids)}


def _load_umap_2d() -> dict[str, np.ndarray]:
    """Load 2D UMAP projections keyed by track_id."""
    if not UMAP_2D_PATH.exists() or not UMAP_2D_IDS_PATH.exists():
        logger.warning("2D UMAP files not found — skipping")
        return {}
    umap = np.load(str(UMAP_2D_PATH)).astype(np.float32)
    ids = np.load(str(UMAP_2D_IDS_PATH), allow_pickle=True)
    return {str(tid): umap[i] for i, tid in enumerate(ids)}


def migrate(db: DatabaseClient) -> None:
    """Run the full migration."""
    t0 = time.time()

    # 1. Create schema
    logger.info("Creating schema...")
    db.create_schema()

    # 2. Load source data
    logger.info("Loading SQLite tracks...")
    sqlite_rows = _load_sqlite()
    logger.info("  %d tracks in SQLite", len(sqlite_rows))

    logger.info("Loading embeddings...")
    emb_map = _load_embeddings()
    logger.info("  %d embeddings loaded", len(emb_map))

    logger.info("Loading UMAP projections...")
    umap5d_map = _load_umap_5d()
    logger.info("  %d 5D projections loaded", len(umap5d_map))
    umap2d_map = _load_umap_2d()
    logger.info("  %d 2D projections loaded", len(umap2d_map))

    # 3. Merge into unified rows
    logger.info("Merging data...")
    merged = []
    for row in sqlite_rows:
        tid = row["track_id"]
        merged.append({
            "track_id": tid,
            "title": row["title"],
            "artist": row["artist"],
            "album": row["album"],
            "preview_url": row.get("preview_url"),
            "preview_path": row.get("preview_path"),
            "art_url": row.get("art_url", ""),
            "art_path_original": row.get("art_path_original"),
            "art_path_32": row.get("art_path_32"),
            "embedding_status": row.get("embedding_status", "pending"),
            "embedding_512": emb_map.get(tid),
            "umap_5d": umap5d_map.get(tid),
            "umap_2d": umap2d_map.get(tid),
        })

    # 4. Bulk insert
    logger.info("Inserting %d tracks into PostgreSQL...", len(merged))
    count = db.bulk_upsert_tracks(merged)
    elapsed = time.time() - t0
    logger.info("Migration complete: %d tracks in %.1fs", count, elapsed)


def verify(db: DatabaseClient, test_recall: bool = False) -> bool:
    """Verify migration integrity."""
    ok = True

    # Count check
    total = db.get_total_tracks()
    logger.info("PostgreSQL tracks with embeddings: %d", total)

    sqlite_rows = _load_sqlite()
    ok_count = sum(1 for r in sqlite_rows if r.get("embedding_status") == "ok")
    if total != ok_count:
        logger.error("Count mismatch: PostgreSQL=%d, SQLite ok=%d", total, ok_count)
        ok = False
    else:
        logger.info("Count matches: %d tracks", total)

    # Spot-check a few tracks
    emb_map = _load_embeddings()
    sample_ids = list(emb_map.keys())[:5]
    for tid in sample_ids:
        pg_emb = db.get_embedding_512(tid)
        if pg_emb is None:
            logger.error("Track %s: missing embedding in PostgreSQL", tid)
            ok = False
            continue
        diff = float(np.linalg.norm(pg_emb - emb_map[tid]))
        if diff > 1e-4:
            logger.error("Track %s: embedding mismatch (diff=%.6f)", tid, diff)
            ok = False
        else:
            logger.info("Track %s: embedding OK (diff=%.6f)", tid, diff)

    # Recall test: compare pgvector HNSW results vs exact numpy search
    if test_recall:
        logger.info("Running recall test (pgvector vs exact numpy)...")
        embs = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)
        ids_arr = np.load(str(EMB_IDS_PATH), allow_pickle=True)
        id_list = [str(t) for t in ids_arr]

        rng = np.random.default_rng(42)
        n_queries = 20
        k = 20
        total_recall = 0.0

        query_indices = rng.choice(len(id_list), size=n_queries, replace=False)
        for qi in query_indices:
            query_emb = embs[qi]
            query_tid = id_list[qi]

            # Exact numpy search
            dists = np.linalg.norm(embs - query_emb, axis=1)
            exact_order = np.argsort(dists)
            exact_top_k = set(id_list[idx] for idx in exact_order[1:k + 1])  # skip self

            # pgvector search
            pg_results = db.find_neighbors_512d(
                query_emb, k=k, exclude_ids=frozenset({query_tid}),
            )
            pg_top_k = set(r[0] for r in pg_results)

            overlap = len(exact_top_k & pg_top_k)
            recall = overlap / k
            total_recall += recall

        avg_recall = total_recall / n_queries
        logger.info(
            "Recall@%d over %d queries: %.1f%%",
            k, n_queries, avg_recall * 100,
        )
        if avg_recall < 0.90:
            logger.warning("Recall below 90%% — consider increasing ef_search")
            ok = False

    if ok:
        logger.info("Verification PASSED")
    else:
        logger.error("Verification FAILED")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Migrate ALBart data to PostgreSQL")
    parser.add_argument("--verify", action="store_true", help="Verify only (skip migration)")
    parser.add_argument("--recall", action="store_true", help="Include recall test in verification")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", default="albart")
    parser.add_argument("--user", default="albart")
    parser.add_argument("--password", default="albart")
    args = parser.parse_args()

    config = DatabaseConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
    )
    db = DatabaseClient(config=config)

    if args.verify:
        success = verify(db, test_recall=args.recall)
        sys.exit(0 if success else 1)
    else:
        migrate(db)
        verify(db, test_recall=args.recall)


if __name__ == "__main__":
    main()
