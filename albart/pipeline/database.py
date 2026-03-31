"""Database access for the ALBart pipeline.

Thin convenience layer over the shared DatabaseClient (PostgreSQL + pgvector).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import psycopg2.extras

from albart.effects.database import DatabaseClient, DatabaseConfig

logger = logging.getLogger(__name__)

# Module-level DatabaseClient — initialised lazily by get_db()
_db: Optional[DatabaseClient] = None


def get_db() -> DatabaseClient:
    """Get the shared DatabaseClient instance."""
    global _db
    if _db is None:
        _db = DatabaseClient(config=DatabaseConfig())
    return _db


def init_db() -> None:
    """Create the PostgreSQL schema (idempotent)."""
    get_db().create_schema()
    logger.info("PostgreSQL database initialized")


def upsert_track(track: dict) -> None:
    """Insert or update a track record from Spotify pull data."""
    db = get_db()
    db.upsert_track(
        track_id=track["track_id"],
        title=track["title"],
        artist=track["artist"],
        album=track["album"],
        preview_url=track.get("preview_url"),
        preview_path=track.get("preview_path"),
        art_url=track.get("art_url", ""),
        art_path_original=track.get("art_path_original"),
        art_path_32=track.get("art_path_32"),
        embedding_status=track.get("embedding_status", "pending"),
    )


def get_track(track_id: str) -> Optional[dict]:
    """Get a single track as a dict, or None."""
    return get_db().get_track_metadata(track_id)


def get_all_tracks() -> List[dict]:
    """Return all tracks as dicts."""
    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT track_id, title, artist, album, preview_url, "
                "preview_path, art_url, art_path_original, art_path_32, "
                "embedding_status FROM tracks"
            )
            return [dict(row) for row in cur.fetchall()]


def get_tracks_by_status(status: str) -> List[dict]:
    db = get_db()
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                "SELECT track_id, title, artist, album, preview_url, "
                "preview_path, art_url, art_path_original, art_path_32, "
                "embedding_status FROM tracks "
                "WHERE embedding_status = %s",
                (status,),
            )
            return [dict(row) for row in cur.fetchall()]


def update_embedding_status(track_id: str, status: str) -> None:
    db = get_db()
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tracks SET embedding_status = %s WHERE track_id = %s",
                (status, track_id),
            )


def update_track_fields(track_id: str, **fields) -> None:
    """Update arbitrary fields on a track row."""
    if not fields:
        return
    db = get_db()
    set_clause = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [track_id]
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE tracks SET {set_clause} WHERE track_id = %s",
                values,
            )


def upsert_embedding(track_id: str, embedding_512) -> None:
    """Store a 512D embedding for a track."""
    get_db().upsert_embedding(track_id, embedding_512)
