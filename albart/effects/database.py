"""PostgreSQL + pgvector database client.

Replaces SQLite + FAISS + .npy files with a single unified store.
All vector search, metadata queries, and embedding lookups go through
this module.

The client is a frozen dataclass — configuration is captured at
construction time, connection pooling is managed internally.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from psycopg2 import pool

from albart.core.state import TrackRef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "albart"
    user: str = "albart"
    password: str = "albart"


# ---------------------------------------------------------------------------
# Connection pool (global, keyed by config)
# ---------------------------------------------------------------------------

_pools: dict[str, pool.ThreadedConnectionPool] = {}
_pool_lock = threading.Lock()


def _pool_key(cfg: DatabaseConfig) -> str:
    return f"{cfg.host}:{cfg.port}:{cfg.database}:{cfg.user}"


def _get_pool(cfg: DatabaseConfig) -> pool.ThreadedConnectionPool:
    key = _pool_key(cfg)
    with _pool_lock:
        if key not in _pools:
            _pools[key] = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                host=cfg.host,
                port=cfg.port,
                database=cfg.database,
                user=cfg.user,
                password=cfg.password,
            )
        return _pools[key]


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tracks (
    track_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    artist            TEXT NOT NULL,
    album             TEXT NOT NULL,
    preview_url       TEXT,
    preview_path      TEXT,
    art_url           TEXT NOT NULL DEFAULT '',
    art_path_original TEXT,
    art_path_32       TEXT,
    embedding_status  TEXT NOT NULL DEFAULT 'pending',

    -- Vector columns (replaces .npy files + FAISS index)
    embedding_512     vector(512),
    umap_25d          vector(25),   -- primary navigation space (dwell + transit)
    umap_5d           vector(5),    -- legacy (kept for map display)
    umap_2d           vector(2),    -- legacy (kept for map display)

    -- Denormalised for fast filtered search
    artist_lower      TEXT
);

-- HNSW indexes for approximate nearest-neighbor search at scale.
-- m=16, ef_construction=200 gives >99% recall for top-20 at 1M+ scale.
CREATE INDEX IF NOT EXISTS idx_tracks_embedding_512
    ON tracks USING hnsw (embedding_512 vector_l2_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_tracks_umap_25d
    ON tracks USING hnsw (umap_25d vector_l2_ops)
    WITH (m = 16, ef_construction = 100);

CREATE INDEX IF NOT EXISTS idx_tracks_umap_5d
    ON tracks USING hnsw (umap_5d vector_l2_ops)
    WITH (m = 16, ef_construction = 100);

CREATE INDEX IF NOT EXISTS idx_tracks_artist_lower
    ON tracks (artist_lower);

CREATE INDEX IF NOT EXISTS idx_tracks_embedding_status
    ON tracks (embedding_status);
"""

# Set HNSW search accuracy at session level
SET_EF_SEARCH = "SET hnsw.ef_search = 100;"


# ---------------------------------------------------------------------------
# Database client
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseClient:
    """PostgreSQL + pgvector client for the ALBart DJ engine.

    Thread-safe via connection pooling.  All methods acquire and release
    connections per call.
    """
    config: DatabaseConfig = field(default_factory=DatabaseConfig)

    @contextmanager
    def _conn(self) -> Generator:
        """Acquire a connection, register pgvector, auto-commit/rollback."""
        p = _get_pool(self.config)
        conn = p.getconn()
        try:
            register_vector(conn)
            conn.cursor().execute(SET_EF_SEARCH)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            p.putconn(conn)

    # ── Schema management ────────────────────────────────────────────

    def create_schema(self) -> None:
        """Create tables and indexes (idempotent)."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
        logger.info("Schema created / verified")

    # ── Track CRUD ───────────────────────────────────────────────────

    def upsert_track(
        self,
        track_id: str,
        title: str,
        artist: str,
        album: str,
        *,
        preview_url: Optional[str] = None,
        preview_path: Optional[str] = None,
        art_url: str = "",
        art_path_original: Optional[str] = None,
        art_path_32: Optional[str] = None,
        embedding_status: str = "pending",
        embedding_512: Optional[np.ndarray] = None,
        umap_5d: Optional[np.ndarray] = None,
        umap_2d: Optional[np.ndarray] = None,
    ) -> None:
        """Insert or update a track with all fields."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tracks (
                        track_id, title, artist, album,
                        preview_url, preview_path,
                        art_url, art_path_original, art_path_32,
                        embedding_status, embedding_512, umap_5d, umap_2d,
                        artist_lower
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s
                    )
                    ON CONFLICT (track_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        artist = EXCLUDED.artist,
                        album = EXCLUDED.album,
                        preview_url = EXCLUDED.preview_url,
                        preview_path = EXCLUDED.preview_path,
                        art_url = EXCLUDED.art_url,
                        art_path_original = EXCLUDED.art_path_original,
                        art_path_32 = EXCLUDED.art_path_32,
                        embedding_status = EXCLUDED.embedding_status,
                        embedding_512 = COALESCE(EXCLUDED.embedding_512, tracks.embedding_512),
                        umap_5d = COALESCE(EXCLUDED.umap_5d, tracks.umap_5d),
                        umap_2d = COALESCE(EXCLUDED.umap_2d, tracks.umap_2d),
                        artist_lower = EXCLUDED.artist_lower
                """, (
                    track_id, title, artist, album,
                    preview_url, preview_path,
                    art_url, art_path_original, art_path_32,
                    embedding_status,
                    embedding_512 if embedding_512 is not None else None,
                    umap_5d if umap_5d is not None else None,
                    umap_2d if umap_2d is not None else None,
                    artist.lower() if artist else "",
                ))

    def upsert_embedding(
        self,
        track_id: str,
        embedding_512: np.ndarray,
    ) -> None:
        """Update only the 512D embedding for a track."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE tracks
                    SET embedding_512 = %s, embedding_status = 'ok'
                    WHERE track_id = %s
                """, (embedding_512, track_id))

    def upsert_umap_25d(
        self,
        track_id: str,
        umap_25d: np.ndarray,
    ) -> None:
        """Update only the 25D UMAP projection for a track."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE tracks SET umap_25d = %s WHERE track_id = %s",
                    (umap_25d, track_id),
                )

    def get_track(self, track_id: str) -> Optional[TrackRef]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    "SELECT track_id, title, artist, album FROM tracks WHERE track_id = %s",
                    (track_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return TrackRef(
                    track_id=row["track_id"],
                    title=row["title"],
                    artist=row["artist"],
                    album=row["album"],
                )

    def get_all_tracks(self) -> List[dict]:
        """Return all tracks as dicts (for orbit matching, etc.)."""
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT track_id, title, artist, album FROM tracks")
                return [dict(row) for row in cur.fetchall()]

    def get_track_artist(self, track_id: str) -> Optional[str]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT artist_lower FROM tracks WHERE track_id = %s",
                    (track_id,),
                )
                row = cur.fetchone()
                return row[0] if row else None

    def get_artists_for_tracks(self, track_ids: List[str]) -> dict:
        """Return {track_id: lowercase_artist} for a batch of tracks."""
        if not track_ids:
            return {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT track_id, artist_lower FROM tracks WHERE track_id = ANY(%s)",
                    (list(track_ids),),
                )
                return {row[0]: row[1] or "" for row in cur.fetchall()}

    def get_total_tracks(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM tracks WHERE embedding_status = 'ok'")
                return cur.fetchone()[0]

    def get_track_metadata(self, track_id: str) -> Optional[dict]:
        """Full metadata row as dict (for server status, art paths, etc.)."""
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT * FROM tracks WHERE track_id = %s", (track_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    # ── Embedding lookups ────────────────────────────────────────────

    def get_embedding_512(self, track_id: str) -> Optional[np.ndarray]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT embedding_512 FROM tracks WHERE track_id = %s",
                    (track_id,),
                )
                row = cur.fetchone()
                if row is None or row[0] is None:
                    return None
                return np.array(row[0], dtype=np.float32)

    def get_embedding_5d(self, track_id: str) -> Optional[np.ndarray]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT umap_5d FROM tracks WHERE track_id = %s",
                    (track_id,),
                )
                row = cur.fetchone()
                if row is None or row[0] is None:
                    return None
                return np.array(row[0], dtype=np.float32)

    def get_embedding_25d(self, track_id: str) -> Optional[np.ndarray]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT umap_25d FROM tracks WHERE track_id = %s",
                    (track_id,),
                )
                row = cur.fetchone()
                if row is None or row[0] is None:
                    return None
                return np.array(row[0], dtype=np.float32)

    # ── Vector search (replaces FAISS) ───────────────────────────────

    def find_neighbors_25d(
        self,
        target: np.ndarray,
        k: int,
        exclude_ids: frozenset[str] = frozenset(),
        mood_ids: Optional[frozenset[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Find K nearest tracks in 25D UMAP space (L2 distance)."""
        search_k = k * 3 + len(exclude_ids)

        with self._conn() as conn:
            with conn.cursor() as cur:
                if mood_ids is not None:
                    cur.execute("""
                        SELECT track_id,
                               umap_25d <-> %s::vector AS dist
                        FROM tracks
                        WHERE umap_25d IS NOT NULL
                          AND NOT (track_id = ANY(%s))
                          AND track_id = ANY(%s)
                        ORDER BY umap_25d <-> %s::vector
                        LIMIT %s
                    """, (
                        target, list(exclude_ids), list(mood_ids),
                        target, search_k,
                    ))
                else:
                    cur.execute("""
                        SELECT track_id,
                               umap_25d <-> %s::vector AS dist
                        FROM tracks
                        WHERE umap_25d IS NOT NULL
                          AND NOT (track_id = ANY(%s))
                        ORDER BY umap_25d <-> %s::vector
                        LIMIT %s
                    """, (target, list(exclude_ids), target, search_k))

                results = []
                for row in cur.fetchall():
                    dist = float(row[1])
                    results.append((row[0], dist))
                    if len(results) >= k:
                        break
                return results

    def find_neighbors_512d(
        self,
        target: np.ndarray,
        k: int,
        exclude_ids: frozenset[str] = frozenset(),
        mood_ids: Optional[frozenset[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Find K nearest tracks in 512D embedding space (L2 distance).

        Uses pgvector HNSW index for approximate search.

        Args:
            target: (512,) float32 query vector.
            k: number of results.
            exclude_ids: track IDs to skip (played set).
            mood_ids: if set, only include these track IDs (mood filter).

        Returns:
            List of (track_id, l2_distance) sorted by distance.
        """
        # Search wider to account for exclusions
        search_k = k * 3 + len(exclude_ids)

        with self._conn() as conn:
            with conn.cursor() as cur:
                if mood_ids is not None:
                    cur.execute("""
                        SELECT track_id,
                               embedding_512 <-> %s::vector AS dist
                        FROM tracks
                        WHERE embedding_512 IS NOT NULL
                          AND NOT (track_id = ANY(%s))
                          AND track_id = ANY(%s)
                        ORDER BY embedding_512 <-> %s::vector
                        LIMIT %s
                    """, (
                        target, list(exclude_ids), list(mood_ids),
                        target, search_k,
                    ))
                else:
                    cur.execute("""
                        SELECT track_id,
                               embedding_512 <-> %s::vector AS dist
                        FROM tracks
                        WHERE embedding_512 IS NOT NULL
                          AND NOT (track_id = ANY(%s))
                        ORDER BY embedding_512 <-> %s::vector
                        LIMIT %s
                    """, (target, list(exclude_ids), target, search_k))

                results = []
                for row in cur.fetchall():
                    # pgvector <-> returns squared L2 distance; take sqrt
                    dist = float(row[1])
                    results.append((row[0], dist))
                    if len(results) >= k:
                        break
                return results

    def find_neighbors_5d(
        self,
        target: np.ndarray,
        k: int,
        exclude_ids: frozenset[str] = frozenset(),
        mood_ids: Optional[frozenset[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Find K nearest tracks in 5D UMAP space (L2 distance).

        Uses pgvector HNSW index.
        """
        search_k = k * 3 + len(exclude_ids)

        with self._conn() as conn:
            with conn.cursor() as cur:
                if mood_ids is not None:
                    cur.execute("""
                        SELECT track_id,
                               umap_5d <-> %s::vector AS dist
                        FROM tracks
                        WHERE umap_5d IS NOT NULL
                          AND NOT (track_id = ANY(%s))
                          AND track_id = ANY(%s)
                        ORDER BY umap_5d <-> %s::vector
                        LIMIT %s
                    """, (
                        target, list(exclude_ids), list(mood_ids),
                        target, search_k,
                    ))
                else:
                    cur.execute("""
                        SELECT track_id,
                               umap_5d <-> %s::vector AS dist
                        FROM tracks
                        WHERE umap_5d IS NOT NULL
                          AND NOT (track_id = ANY(%s))
                        ORDER BY umap_5d <-> %s::vector
                        LIMIT %s
                    """, (target, list(exclude_ids), target, search_k))

                results = []
                for row in cur.fetchall():
                    dist = float(row[1])
                    results.append((row[0], dist))
                    if len(results) >= k:
                        break
                return results

    # ── Mood mask computation ────────────────────────────────────────

    def compute_mood_mask(
        self,
        positive_embs: Optional[np.ndarray],
        negative_embs: Optional[np.ndarray],
        threshold: float,
    ) -> Optional[frozenset[str]]:
        """Compute the set of track IDs that pass the mood filter.

        Uses cosine similarity between track embeddings and mood
        descriptor embeddings.  For 5K tracks this is fast enough in
        Python; at 1M+ scale, push to SQL with pgvector <=> operator.

        Args:
            positive_embs: (M, 512) positive mood embeddings, or None.
            negative_embs: (N, 512) negative mood embeddings, or None.
            threshold: cosine similarity threshold (0–1).

        Returns:
            frozenset of track_ids that pass, or None if no mood set.
        """
        if positive_embs is None and negative_embs is None:
            return None

        # Fetch all embeddings (efficient for < 100K tracks)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT track_id, embedding_512 FROM tracks "
                    "WHERE embedding_512 IS NOT NULL"
                )
                rows = cur.fetchall()

        if not rows:
            return frozenset()

        track_ids = [r[0] for r in rows]
        embeddings = np.array([r[1] for r in rows], dtype=np.float32)

        # L2-normalise for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_normed = embeddings / np.maximum(norms, 1e-8)

        mask = np.ones(len(track_ids), dtype=bool)

        if positive_embs is not None:
            pos_sims = emb_normed @ positive_embs.T
            mask &= pos_sims.max(axis=1) > threshold

        if negative_embs is not None:
            neg_threshold = 1.0 - threshold
            neg_sims = emb_normed @ negative_embs.T
            mask &= neg_sims.max(axis=1) <= neg_threshold

        in_mood = frozenset(
            tid for tid, m in zip(track_ids, mask) if m
        )
        logger.info(
            "Mood mask: %d/%d tracks in-mood (%.0f%%)",
            len(in_mood), len(track_ids),
            100 * len(in_mood) / max(len(track_ids), 1),
        )
        return in_mood

    # ── Embedding math (transit / long hop) ──────────────────────────

    def compute_transit_target(
        self,
        current_track_id: str,
        target_track_id: str,
        fraction: float,
    ) -> Optional[np.ndarray]:
        """Interpolate in 25D UMAP space for orbit transit.

        Returns a point ``fraction`` of the way from current to target.
        No normalization needed — UMAP coordinates are Euclidean.
        """
        current = self.get_embedding_25d(current_track_id)
        target = self.get_embedding_25d(target_track_id)
        if current is None or target is None:
            return target if target is not None else current

        point = current.astype(np.float64) + (
            target.astype(np.float64) - current.astype(np.float64)
        ) * fraction
        return point.astype(np.float32)

    def compute_initial_transit_distance(
        self,
        current_track_id: str,
        target_track_id: str,
    ) -> float:
        """L2 distance between two tracks in 25D UMAP space."""
        current = self.get_embedding_25d(current_track_id)
        target = self.get_embedding_25d(target_track_id)
        if current is None or target is None:
            return 1.0
        return float(np.linalg.norm(
            target.astype(np.float64) - current.astype(np.float64)
        ))

    def compute_long_hop_target(
        self,
        prev_track_id: str,
        current_track_id: str,
        hop_multiplier: float,
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        """Extrapolate trajectory for a long hop in 25D UMAP space."""
        emb_prev = self.get_embedding_25d(prev_track_id)
        emb_curr = self.get_embedding_25d(current_track_id)
        if emb_prev is None or emb_curr is None:
            return emb_curr

        direction = emb_curr.astype(np.float64) - emb_prev.astype(np.float64)
        dist = float(np.linalg.norm(direction))
        if dist < 1e-8:
            direction = rng.normal(size=25)
            dist = float(np.linalg.norm(direction))

        direction /= dist
        jump_dist = dist * hop_multiplier
        target = emb_curr.astype(np.float64) + direction * jump_dist
        return target.astype(np.float32)

    # ── Text search (for seed track matching) ────────────────────────

    def search_tracks(self, query: str, limit: int = 10) -> List[TrackRef]:
        """Simple ILIKE search on title and artist."""
        pattern = f"%{query}%"
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT track_id, title, artist, album FROM tracks
                    WHERE title ILIKE %s OR artist ILIKE %s
                    LIMIT %s
                """, (pattern, pattern, limit))
                return [
                    TrackRef(
                        track_id=row["track_id"],
                        title=row["title"],
                        artist=row["artist"],
                        album=row["album"],
                    )
                    for row in cur.fetchall()
                ]

    # ── Deduplication ─────────────────────────────────────────────────

    def find_duplicates(
        self,
        threshold: float = 0.01,
    ) -> List[Tuple[str, str, str, str, str, str, float]]:
        """Find near-duplicate track pairs by embedding distance.

        Returns list of (tid_keep, title_keep, artist_keep,
                         tid_drop, title_drop, artist_drop, distance).

        Keep/drop decision: prefer the track with a preview file,
        then prefer the one that appears first alphabetically by track_id
        (stable, deterministic).
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT a.track_id, a.title, a.artist, a.preview_path,
                           b.track_id, b.title, b.artist, b.preview_path,
                           a.embedding_512 <-> b.embedding_512 AS dist
                    FROM tracks a, tracks b
                    WHERE a.track_id < b.track_id
                      AND a.embedding_512 IS NOT NULL
                      AND b.embedding_512 IS NOT NULL
                      AND a.embedding_512 <-> b.embedding_512 < %s
                    ORDER BY dist, a.track_id
                """, (threshold,))

                results = []
                for row in cur.fetchall():
                    a_tid, a_title, a_artist, a_preview = row[0], row[1], row[2], row[3]
                    b_tid, b_title, b_artist, b_preview = row[4], row[5], row[6], row[7]
                    dist = float(row[8])

                    # Decide which to keep: prefer the one with a preview
                    if a_preview and not b_preview:
                        keep, drop = (a_tid, a_title, a_artist), (b_tid, b_title, b_artist)
                    elif b_preview and not a_preview:
                        keep, drop = (b_tid, b_title, b_artist), (a_tid, a_title, a_artist)
                    else:
                        # Both have or lack previews — keep the first alphabetically
                        keep, drop = (a_tid, a_title, a_artist), (b_tid, b_title, b_artist)

                    results.append((*keep, *drop, dist))
                return results

    def delete_tracks(self, track_ids: List[str]) -> int:
        """Delete tracks by ID. Returns count deleted."""
        if not track_ids:
            return 0
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM tracks WHERE track_id = ANY(%s)",
                    (list(track_ids),),
                )
                return cur.rowcount

    # ── Batch operations (for migration / pipeline) ──────────────────

    def bulk_upsert_tracks(
        self,
        rows: List[dict],
        batch_size: int = 500,
    ) -> int:
        """Bulk upsert tracks.  Each dict should have keys matching
        the tracks table columns.  Returns count of rows upserted."""
        count = 0
        with self._conn() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i + batch_size]
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO tracks (
                            track_id, title, artist, album,
                            preview_url, preview_path,
                            art_url, art_path_original, art_path_32,
                            embedding_status,
                            embedding_512, umap_5d, umap_2d,
                            artist_lower
                        ) VALUES %s
                        ON CONFLICT (track_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            artist = EXCLUDED.artist,
                            album = EXCLUDED.album,
                            preview_url = EXCLUDED.preview_url,
                            preview_path = EXCLUDED.preview_path,
                            art_url = EXCLUDED.art_url,
                            art_path_original = EXCLUDED.art_path_original,
                            art_path_32 = EXCLUDED.art_path_32,
                            embedding_status = EXCLUDED.embedding_status,
                            embedding_512 = COALESCE(EXCLUDED.embedding_512, tracks.embedding_512),
                            umap_5d = COALESCE(EXCLUDED.umap_5d, tracks.umap_5d),
                            umap_2d = COALESCE(EXCLUDED.umap_2d, tracks.umap_2d),
                            artist_lower = EXCLUDED.artist_lower
                        """,
                        [
                            (
                                r["track_id"], r["title"], r["artist"], r["album"],
                                r.get("preview_url"), r.get("preview_path"),
                                r.get("art_url", ""),
                                r.get("art_path_original"), r.get("art_path_32"),
                                r.get("embedding_status", "pending"),
                                r.get("embedding_512"), r.get("umap_5d"),
                                r.get("umap_2d"),
                                (r["artist"] or "").lower(),
                            )
                            for r in batch
                        ],
                        page_size=batch_size,
                    )
                    count += len(batch)
        return count
