"""Ingest a Spotify playlist (or album) into the ALBart database.

Downloads previews, computes CLAP embeddings and 25D UMAP projections,
and stores everything in PostgreSQL.  Idempotent — tracks already in
the database with embeddings are skipped.

Usage:
    python3 -m albart.pipeline.ingest_playlist "https://open.spotify.com/playlist/..."
    python3 -m albart.pipeline.ingest_playlist "spotify:playlist:37i9dQZF1DX5g856aiKiDS"
    python3 -m albart.pipeline.ingest_playlist "https://open.spotify.com/album/..."
    python3 -m albart.pipeline.ingest_playlist --dry-run "https://open.spotify.com/playlist/..."
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCOPE = (
    "user-read-playback-state,"
    "user-modify-playback-state,"
    "playlist-read-private,"
    "playlist-read-collaborative,"
    "user-library-read,"
    "user-top-read"
)


def _get_client() -> spotipy.Spotify:
    """Authenticated Spotify client with playlist scope.

    Uses a separate cache file (.cache-ingest) so the broader scope
    doesn't interfere with the DJ server's token. Will prompt for
    browser auth on first run.
    """
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=os.environ.get(
            "SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback"
        ),
        scope=SCOPE,
        cache_path=".cache-ingest",
    ))


def _parse_uri(uri: str) -> tuple[str, str]:
    """Parse a Spotify URI or URL into (type, id).

    Returns ("playlist", "abc123") or ("album", "xyz789").
    Strips query parameters (?si=...) from sharing URLs.
    """
    # Strip query params
    uri = uri.split("?")[0]

    # URL format: https://open.spotify.com/playlist/37i9dQZF1DX5g856aiKiDS
    url_match = re.search(r"open\.spotify\.com/(playlist|album)/([a-zA-Z0-9]+)", uri)
    if url_match:
        return url_match.group(1), url_match.group(2)

    # URI format: spotify:playlist:37i9dQZF1DX5g856aiKiDS
    uri_match = re.match(r"spotify:(playlist|album):([a-zA-Z0-9]+)", uri)
    if uri_match:
        return uri_match.group(1), uri_match.group(2)

    # Bare ID — assume playlist
    if re.match(r"^[a-zA-Z0-9]{22}$", uri):
        return "playlist", uri

    raise ValueError(f"Cannot parse Spotify URI: {uri}")


def _fetch_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> list[dict]:
    """Fetch all tracks from a playlist (paginated)."""
    tracks = []
    offset = 0
    while True:
        result = sp.playlist_tracks(playlist_id, offset=offset, limit=100)
        items = result.get("items", [])
        if not items:
            break
        for item in items:
            track = item.get("track")
            if track and track.get("id"):
                tracks.append(track)
        offset += len(items)
        if result.get("next") is None:
            break
    return tracks


def _fetch_album_tracks(sp: spotipy.Spotify, album_id: str) -> list[dict]:
    """Fetch all tracks from an album, enriching with album metadata."""
    album = sp.album(album_id)
    album_info = {
        "name": album.get("name"),
        "images": album.get("images", []),
    }
    tracks = []
    offset = 0
    while True:
        result = sp.album_tracks(album_id, offset=offset, limit=50)
        items = result.get("items", [])
        if not items:
            break
        for item in items:
            # Album track items don't include album info — add it
            item["album"] = album_info
            if item.get("id"):
                tracks.append(item)
        offset += len(items)
        if result.get("next") is None:
            break
    return tracks


def _download_track(
    track_id: str,
    track: dict,
    db,
) -> dict | None:
    """Download preview + art for a single track. Returns normalized dict or None."""
    from albart.pipeline import downloader, preprocess
    from albart.pipeline.deezer import lookup_preview_url as deezer_lookup
    from albart.pipeline.itunes import lookup_preview_url as itunes_lookup
    from albart.pipeline.spotify import normalize_track

    normalized = normalize_track(track)
    if normalized is None:
        return None

    # Find preview URL: Spotify → Deezer → iTunes
    preview_url = track.get("preview_url")
    if not preview_url:
        preview_url = deezer_lookup(normalized["title"], normalized["artist"])
    if not preview_url:
        preview_url = itunes_lookup(normalized["title"], normalized["artist"])

    downloader.ensure_dirs()
    preprocess.ensure_dirs()

    preview_path = None
    if preview_url:
        preview_path = downloader.download_preview(track_id, preview_url)
        normalized["preview_url"] = preview_url
        if preview_path:
            normalized["preview_path"] = str(preview_path)

    if normalized.get("art_url"):
        art_path = downloader.download_art(track_id, normalized["art_url"])
        if art_path:
            normalized["art_path_original"] = str(art_path)
            art_32 = preprocess.downsample_art(track_id, art_path)
            if art_32:
                normalized["art_path_32"] = str(art_32)

    # Store metadata even without preview
    if not preview_path:
        normalized["embedding_status"] = "no_preview"
    else:
        normalized["embedding_status"] = "pending"

    db.upsert_track(
        track_id=normalized["track_id"],
        title=normalized["title"],
        artist=normalized["artist"],
        album=normalized["album"],
        preview_url=normalized.get("preview_url"),
        preview_path=normalized.get("preview_path"),
        art_url=normalized.get("art_url", ""),
        art_path_original=normalized.get("art_path_original"),
        art_path_32=normalized.get("art_path_32"),
        embedding_status=normalized["embedding_status"],
    )

    if not preview_path:
        return None
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a Spotify playlist or album into ALBart"
    )
    parser.add_argument(
        "uri", nargs="?", default=None,
        help="Spotify playlist/album URL or URI",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be ingested without downloading",
    )
    args = parser.parse_args()

    sp = _get_client()

    if args.uri is None:
        parser.error("Provide a Spotify playlist or album URL")

    # Parse URI and fetch tracks
    kind, spotify_id = _parse_uri(args.uri)
    logger.info("Fetching %s %s...", kind, spotify_id)

    try:
        if kind == "playlist":
            info = sp.playlist(spotify_id, fields="name,owner.display_name")
            name = info.get("name", spotify_id)
            owner = info.get("owner", {}).get("display_name", "")
            raw_tracks = _fetch_playlist_tracks(sp, spotify_id)
            logger.info("Playlist: %s (by %s) — %d tracks", name, owner, len(raw_tracks))
        elif kind == "album":
            raw_tracks = _fetch_album_tracks(sp, spotify_id)
            album_name = raw_tracks[0]["album"]["name"] if raw_tracks else spotify_id
            logger.info("Album: %s — %d tracks", album_name, len(raw_tracks))
        else:
            logger.error("Unknown type: %s", kind)
            sys.exit(1)
    except Exception as e:
        logger.error("Failed to fetch %s %s: %s", kind, spotify_id, e)
        sys.exit(1)

    # Check which tracks are already in the database
    from albart.effects.database import DatabaseClient, DatabaseConfig
    db = DatabaseClient(DatabaseConfig())

    track_map = {}  # track_id → spotify track dict
    skip_count = 0
    for track in raw_tracks:
        tid = track["id"]
        existing = db.get_track_metadata(tid)
        if existing and existing.get("embedding_status") == "ok":
            skip_count += 1
            continue
        track_map[tid] = track

    logger.info(
        "%d new tracks to ingest, %d already in database",
        len(track_map), skip_count,
    )

    if not track_map:
        logger.info("Nothing to do.")
        return

    if args.dry_run:
        from albart.pipeline.spotify import normalize_track
        print(f"\nWould ingest {len(track_map)} tracks:")
        for tid, track in track_map.items():
            n = normalize_track(track)
            if n:
                print(f"  {n['artist']:30s}  {n['title']}")
        return

    # Phase 1: Download previews + art (parallelized)
    logger.info("Phase 1: Downloading previews and art (%d workers)...", args.workers)
    downloaded = {}  # track_id → normalized dict (only tracks with previews)
    no_preview = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_track, tid, track, db): tid
            for tid, track in track_map.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Download"):
            tid = futures[future]
            try:
                result = future.result()
                if result:
                    downloaded[tid] = result
                else:
                    no_preview.append(tid)
            except Exception as e:
                logger.error("Download failed for %s: %s", tid, e)
                no_preview.append(tid)

    logger.info(
        "Downloaded: %d with previews, %d without",
        len(downloaded), len(no_preview),
    )

    if not downloaded:
        logger.info("No tracks with previews to embed.")
        return

    # Phase 2: Compute CLAP embeddings (sequential — GPU bottleneck)
    logger.info("Phase 2: Computing CLAP embeddings...")
    from albart.pipeline.embedder import SAMPLE_RATE, embed_audio, load_model
    from albart.utils import DATA_DIR, load_config

    config = load_config()
    norm_target = float(config.get("pipeline", {}).get("norm_target", 0.12))

    model, processor, device = load_model(allow_mps=True)
    logger.info("CLAP model loaded on %s", device)

    chunk_samples = 10 * SAMPLE_RATE
    n_chunks = 3
    total_samples = n_chunks * chunk_samples

    embeddings = {}  # track_id → (512,) float32
    for tid, info in tqdm(downloaded.items(), desc="Embedding"):
        try:
            import librosa
            audio_path = DATA_DIR / info["preview_path"]
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True,
                                    dtype="float32")

            if len(audio) < total_samples:
                audio = np.pad(audio, (total_samples - len(audio), 0))
            else:
                audio = audio[-total_samples:]

            chunk_embs = []
            for i in range(n_chunks):
                chunk = audio[i * chunk_samples:(i + 1) * chunk_samples]
                chunk_embs.append(
                    embed_audio(chunk, model, processor, device,
                                norm_target=norm_target)
                )

            emb = np.mean(chunk_embs, axis=0).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings[tid] = emb
        except Exception as e:
            logger.error("Embedding failed for %s: %s", tid, e)

    logger.info("Computed %d embeddings", len(embeddings))

    # Phase 3: Store embeddings + 25D projections
    logger.info("Phase 3: Storing embeddings and projections...")

    # Load projector
    projector = None
    umap_cfg = config.get("umap_25d", {})
    model_path = umap_cfg.get("model_path", "data/umap_25d_model/model.pt")
    full_path = DATA_DIR.parent / model_path
    if full_path.exists():
        from albart.effects.umap_projector import UmapProjector
        projector = UmapProjector.load(full_path)

    stored = 0
    for tid, emb in tqdm(embeddings.items(), desc="Storing"):
        try:
            db.upsert_embedding(tid, emb)
            if projector is not None:
                umap_25d = projector.project(emb)
                db.upsert_umap_25d(tid, umap_25d)
            stored += 1
        except Exception as e:
            logger.error("Store failed for %s: %s", tid, e)

    # Summary
    total = len(track_map)
    print(f"\n{'=' * 50}")
    print(f"Ingestion complete:")
    print(f"  Total in playlist:   {len(raw_tracks)}")
    print(f"  Already in database: {skip_count}")
    print(f"  Newly ingested:      {stored}")
    print(f"  No preview:          {len(no_preview)}")
    print(f"  Failed:              {total - stored - len(no_preview)}")
    print(f"  Database total:      {db.get_total_tracks()}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
