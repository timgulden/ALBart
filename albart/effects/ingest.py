"""On-the-fly ingestion of a single Spotify track into the database.

When the DJ encounters an unknown track (e.g. user queued something not
in the library), this module downloads the preview, computes embeddings,
and stores everything in PostgreSQL — reusing the existing pipeline code.

Designed to run in a background thread while the track plays.  Typical
wall-clock time: ~15-20 seconds for metadata + preview + CLAP + UMAP.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from albart.effects.database import DatabaseClient
from albart.pipeline import downloader, preprocess
from albart.pipeline.deezer import lookup_preview_url as deezer_lookup
from albart.pipeline.itunes import lookup_preview_url as itunes_lookup
from albart.pipeline.spotify import normalize_track

logger = logging.getLogger(__name__)


def ingest_track(
    track_id: str,
    sp,  # spotipy.Spotify
    db: DatabaseClient,
    projector=None,  # UmapProjector | None
    clap_model=None,
    clap_processor=None,
    clap_device: str = "cpu",
    norm_target: float = 0.12,
) -> bool:
    """Ingest a single unknown Spotify track into the database.

    Fetches metadata, downloads preview + art, computes CLAP embedding
    and 25D UMAP projection, then stores everything in PostgreSQL.

    Returns True on success (embedding stored), False otherwise.
    Thread-safe: uses only thread-safe components (DB connection pool,
    torch.no_grad inference, file I/O to unique paths).
    """
    # Check if already in database with an embedding
    existing = db.get_track_metadata(track_id)
    if existing and existing.get("embedding_status") == "ok":
        logger.info("Track %s already ingested — skipping", track_id)
        return True

    # Step 1: Fetch metadata from Spotify
    try:
        item = sp.track(track_id)
    except Exception as e:
        logger.error("Spotify API error for %s: %s", track_id, e)
        return False

    track = normalize_track(item)
    if track is None:
        logger.error("Could not normalize track %s", track_id)
        return False

    logger.info("Ingesting: %s — %s", track["artist"], track["title"])

    # Step 2: Find preview URL (Spotify → Deezer → iTunes fallback)
    preview_url = item.get("preview_url")
    if not preview_url:
        preview_url = deezer_lookup(track["title"], track["artist"])
    if not preview_url:
        preview_url = itunes_lookup(track["title"], track["artist"])

    # Step 3: Download preview + art
    downloader.ensure_dirs()
    preprocess.ensure_dirs()

    preview_path = None
    if preview_url:
        preview_path = downloader.download_preview(track_id, preview_url)
        track["preview_url"] = preview_url
        if preview_path:
            track["preview_path"] = str(preview_path)

    art_path = None
    if track.get("art_url"):
        art_path = downloader.download_art(track_id, track["art_url"])
        if art_path:
            track["art_path_original"] = str(art_path)
            art_32 = preprocess.downsample_art(track_id, art_path)
            if art_32:
                track["art_path_32"] = str(art_32)

    # Step 4: Upsert metadata (even if we can't embed — at least store the track)
    if not preview_path:
        track["embedding_status"] = "no_preview"
        _upsert_track(db, track)
        logger.warning("No preview available for %s — stored metadata only", track_id)
        return False

    track["embedding_status"] = "pending"
    _upsert_track(db, track)

    # Step 5: Compute CLAP embedding
    if clap_model is None:
        logger.error("No CLAP model provided — cannot embed %s", track_id)
        return False

    try:
        import librosa
        from albart.pipeline.embedder import SAMPLE_RATE, embed_audio
        from albart.utils import DATA_DIR

        audio_path = DATA_DIR / preview_path
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True,
                                dtype="float32")

        # 3-chunk averaging (same as pipeline)
        chunk_samples = 10 * SAMPLE_RATE
        n_chunks = 3
        total = n_chunks * chunk_samples
        if len(audio) < total:
            audio = np.pad(audio, (total - len(audio), 0))
        else:
            audio = audio[-total:]

        chunk_embs = []
        for i in range(n_chunks):
            chunk = audio[i * chunk_samples:(i + 1) * chunk_samples]
            chunk_embs.append(
                embed_audio(chunk, clap_model, clap_processor, clap_device,
                            norm_target=norm_target)
            )

        emb_512 = np.mean(chunk_embs, axis=0).astype(np.float32)
        emb_512 = emb_512 / (np.linalg.norm(emb_512) + 1e-8)

    except Exception as e:
        logger.error("Embedding failed for %s: %s", track_id, e)
        _update_status(db, track_id, "error")
        return False

    # Step 6: Store 512D embedding
    db.upsert_embedding(track_id, emb_512)

    # Step 7: Project to 25D via parametric UMAP (if available)
    if projector is not None:
        try:
            umap_25d = projector.project(emb_512)
            db.upsert_umap_25d(track_id, umap_25d)
            logger.info("Ingestion complete: %s (512D + 25D)", track_id)
        except Exception as e:
            logger.warning("25D projection failed for %s: %s (512D still stored)", track_id, e)
    else:
        logger.info("Ingestion complete: %s (512D only, no projector)", track_id)

    return True


def _upsert_track(db: DatabaseClient, track: dict) -> None:
    """Upsert track metadata via the database client."""
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


def _update_status(db: DatabaseClient, track_id: str, status: str) -> None:
    """Update just the embedding_status field."""
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tracks SET embedding_status = %s WHERE track_id = %s",
                (status, track_id),
            )
