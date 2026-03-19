"""CLI entry point for the ALBart offline pipeline.

Usage:
    python -m albart.pipeline.run_pipeline [--force]
"""

from __future__ import annotations

import argparse
import logging

import librosa
import numpy as np
from tqdm import tqdm

from albart.pipeline import database, downloader, embedder, itunes, preprocess, spotify
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

    # --- iTunes preview URL lookup for tracks with no Spotify preview ---
    needs_preview = [
        t for t in database.get_all_tracks(conn)
        if t["preview_url"] is None and (force or t["embedding_status"] != "ok")
    ]
    if needs_preview:
        print(f"Looking up iTunes preview URLs for {len(needs_preview)} tracks...")
        found = 0
        for row in tqdm(needs_preview, desc="iTunes lookup"):
            url = itunes.lookup_preview_url(row["title"], row["artist"])
            if url:
                with conn:
                    conn.execute(
                        "UPDATE tracks SET preview_url=?, embedding_status='pending' WHERE track_id=?",
                        (url, row["track_id"]),
                    )
                found += 1
            else:
                with conn:
                    database.update_embedding_status(conn, row["track_id"], "no_preview")
        print(f"Found iTunes previews for {found} of {len(needs_preview)} tracks.")

    # --- Download and preprocess ---
    to_process = [
        t for t in database.get_all_tracks(conn)
        if t["preview_url"] and (force or t["embedding_status"] == "pending")
    ]
    print(f"Downloading previews and art for {len(to_process)} tracks...")

    for row in tqdm(to_process, desc="Downloading + preprocessing"):
        track_id = row["track_id"]

        # Download preview (extension based on URL)
        preview_rel = downloader.download_preview(track_id, row["preview_url"])
        if preview_rel is None:
            with conn:
                database.update_embedding_status(conn, track_id, "error")
            continue

        # Download art
        art_rel = downloader.download_art(track_id, row["art_url"])
        if art_rel is None:
            with conn:
                database.update_embedding_status(conn, track_id, "error")
            continue

        # Downsample art to 32x32
        art_32_rel = preprocess.downsample_art(track_id, art_rel)

        # Update paths in DB
        with conn:
            conn.execute(
                """UPDATE tracks SET preview_path=?, art_path_original=?, art_path_32=?
                   WHERE track_id=?""",
                (
                    str(preview_rel),
                    str(art_rel),
                    str(art_32_rel) if art_32_rel else None,
                    track_id,
                ),
            )

    # --- Compute embeddings ---
    model, processor, device = embedder.load_model()

    embeddable = [
        t for t in database.get_all_tracks(conn)
        if t["preview_path"] and (force or t["embedding_status"] == "pending")
    ]
    print(f"Computing embeddings for {len(embeddable)} tracks...")

    new_embeddings = []
    new_ids = []

    for row in tqdm(embeddable, desc="Embedding"):
        track_id = row["track_id"]
        preview_path = DATA_DIR / row["preview_path"]

        try:
            audio, _ = librosa.load(str(preview_path), sr=48000, mono=True, dtype="float32")
            emb = embedder.embed_audio(audio, model, processor, device)
            new_embeddings.append(emb)
            new_ids.append(track_id)
            with conn:
                database.update_embedding_status(conn, track_id, "ok")
        except Exception as e:
            logger.error("Embedding failed for %s: %s", track_id, e)
            with conn:
                database.update_embedding_status(conn, track_id, "error")

    # --- Rebuild FAISS index from all OK tracks ---
    all_ok = [t for t in database.get_all_tracks(conn) if t["embedding_status"] == "ok"]
    if all_ok:
        print(f"Rebuilding FAISS index from {len(all_ok)} embedded tracks...")
        # Load all embeddings by re-embedding won't work; we need stored vectors.
        # For now, build index only from newly embedded tracks in this run.
        # TODO: persist embeddings to disk for incremental index rebuilds.
        if new_ids:
            embeddings_array = np.stack(new_embeddings).astype(np.float32)
            embedder.build_and_save_index(embeddings_array, new_ids)
    elif new_ids:
        embeddings_array = np.stack(new_embeddings).astype(np.float32)
        embedder.build_and_save_index(embeddings_array, new_ids)

    # --- Summary ---
    all_final = database.get_all_tracks(conn)
    counts: dict[str, int] = {}
    for t in all_final:
        counts[t["embedding_status"]] = counts.get(t["embedding_status"], 0) + 1

    print("\n--- Pipeline complete ---")
    print(f"  ok:         {counts.get('ok', 0)}")
    print(f"  no_preview: {counts.get('no_preview', 0)}")
    print(f"  error:      {counts.get('error', 0)}")
    print(f"  pending:    {counts.get('pending', 0)}")
    conn.close()


def main() -> None:
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
