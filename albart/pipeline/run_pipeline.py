"""CLI entry point for the ALBart offline pipeline.

Usage:
    python -m albart.pipeline.run_pipeline [--force]
"""

import argparse
import logging
import sys

import numpy as np
from tqdm import tqdm

from albart.pipeline import database, downloader, embedder, preprocess, spotify
from albart.utils import DATA_DIR, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run(force: bool = False) -> None:
    config = load_config()
    _ = config  # reserved for future pipeline config keys

    # --- Setup ---
    database.init_db()
    downloader.ensure_dirs()
    preprocess.ensure_dirs()

    # --- Spotify pull ---
    print("Authenticating with Spotify...")
    sp = spotify.get_client()
    tracks = spotify.fetch_top_tracks(sp)
    print(f"Fetched {len(tracks)} tracks from Spotify.")

    # --- Upsert into DB (skipping already-OK tracks unless --force) ---
    conn = database.get_connection()
    with conn:
        for track in tracks:
            existing = database.get_track(conn, track["track_id"])
            if existing and existing["embedding_status"] == "ok" and not force:
                continue
            database.upsert_track(conn, track)

    # --- Download and preprocess ---
    all_tracks = database.get_all_tracks(conn)
    to_process = [
        t for t in all_tracks
        if force or t["embedding_status"] not in ("ok", "no_preview")
    ]
    print(f"Processing {len(to_process)} tracks (skipping already-OK and no-preview).")

    for row in tqdm(to_process, desc="Downloading + preprocessing"):
        track_id = row["track_id"]

        # Download preview
        if row["preview_url"]:
            preview_rel = downloader.download_preview(track_id, row["preview_url"])
            if preview_rel is None:
                with conn:
                    database.update_embedding_status(conn, track_id, "error")
                continue
        else:
            with conn:
                database.update_embedding_status(conn, track_id, "no_preview")
            continue

        # Download art
        art_rel = downloader.download_art(track_id, row["art_url"])
        if art_rel is None:
            with conn:
                database.update_embedding_status(conn, track_id, "error")
            continue

        # Downsample art
        art_32_rel = preprocess.downsample_art(track_id, art_rel)

        # Update paths in DB
        with conn:
            conn.execute(
                """UPDATE tracks SET preview_path=?, art_path_original=?, art_path_32=?
                   WHERE track_id=?""",
                (str(preview_rel), str(art_rel), str(art_32_rel) if art_32_rel else None, track_id),
            )

    # --- Compute embeddings ---
    model, processor, device = embedder.load_model()

    embeddable = database.get_tracks_by_status(conn, "pending")
    # Also re-embed if --force and already ok
    if force:
        embeddable = [t for t in database.get_all_tracks(conn) if t["preview_path"]]

    print(f"Computing embeddings for {len(embeddable)} tracks...")

    all_embeddings = []
    all_ids = []

    # Include already-OK tracks in the final index
    ok_tracks = [t for t in database.get_all_tracks(conn) if t["embedding_status"] == "ok" and not force]

    for row in tqdm(embeddable, desc="Embedding"):
        track_id = row["track_id"]
        preview_path = DATA_DIR / row["preview_path"]

        try:
            import soundfile as sf
            audio, sr = sf.read(str(preview_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Resample to 48kHz if needed — processor handles this
            emb = embedder.embed_audio(audio, model, processor, device)
            all_embeddings.append(emb)
            all_ids.append(track_id)
            with conn:
                database.update_embedding_status(conn, track_id, "ok")
        except Exception as e:
            logger.error("Embedding failed for %s: %s", track_id, e)
            with conn:
                database.update_embedding_status(conn, track_id, "error")

    # Rebuild index including previously-OK tracks (when not forcing full rebuild)
    if ok_tracks and not force:
        print(f"Note: {len(ok_tracks)} previously embedded tracks are not re-embedded.")
        print("Run with --force to rebuild the full index from scratch.")

    if all_ids:
        embeddings_array = np.stack(all_embeddings).astype(np.float32)
        embedder.build_and_save_index(embeddings_array, all_ids)

    # --- Summary ---
    all_final = database.get_all_tracks(conn)
    counts = {}
    for t in all_final:
        counts[t["embedding_status"]] = counts.get(t["embedding_status"], 0) + 1

    print("\n--- Pipeline complete ---")
    print(f"  ok:         {counts.get('ok', 0)}")
    print(f"  no_preview: {counts.get('no_preview', 0)}")
    print(f"  error:      {counts.get('error', 0)}")
    print(f"  pending:    {counts.get('pending', 0)}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="ALBart offline pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all tracks, ignoring existing status",
    )
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
