"""SQLite database schema and access for the ALBart pipeline."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from albart.utils import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "db.sqlite"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id         TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    artist           TEXT NOT NULL,
    album            TEXT NOT NULL,
    preview_url      TEXT,
    preview_path     TEXT,
    art_url          TEXT NOT NULL,
    art_path_original TEXT,
    art_path_32      TEXT,
    embedding_status TEXT NOT NULL DEFAULT 'pending'
);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the database and tracks table if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
    logger.info("Database initialized at %s", db_path)


def upsert_track(conn: sqlite3.Connection, track: dict) -> None:
    """Insert or replace a track record."""
    conn.execute(
        """
        INSERT INTO tracks
            (track_id, title, artist, album, preview_url, preview_path,
             art_url, art_path_original, art_path_32, embedding_status)
        VALUES
            (:track_id, :title, :artist, :album, :preview_url, :preview_path,
             :art_url, :art_path_original, :art_path_32, :embedding_status)
        ON CONFLICT(track_id) DO UPDATE SET
            title=excluded.title,
            artist=excluded.artist,
            album=excluded.album,
            art_url=excluded.art_url
        """,
        track,
    )


def get_track(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM tracks WHERE track_id = ?", (track_id,)
    ).fetchone()


def get_all_tracks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM tracks").fetchall()


def get_tracks_by_status(
    conn: sqlite3.Connection, status: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM tracks WHERE embedding_status = ?", (status,)
    ).fetchall()


def update_embedding_status(
    conn: sqlite3.Connection, track_id: str, status: str
) -> None:
    conn.execute(
        "UPDATE tracks SET embedding_status = ? WHERE track_id = ?",
        (status, track_id),
    )
