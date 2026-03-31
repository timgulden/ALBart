"""CLI entry point for the ALBart offline pipeline.

Usage:
    python -m albart.pipeline.run_pipeline [--force]

Design note: Deezer preview URLs contain signed tokens that expire within
~90 minutes of generation. The lookup and download steps are therefore merged
into a single pass — each URL is downloaded immediately after being fetched.
"""

from __future__ import annotations

import argparse
import logging

import librosa
import numpy as np
from tqdm import tqdm

from albart.pipeline import database, deezer, downloader, embedder, itunes, preprocess, spotify
from albart.utils import DATA_DIR, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run(force: bool = False, skip_spotify: bool = False) -> None:
    config = load_config()

    # --- Setup ---
    database.init_db()
    downloader.ensure_dirs()
    preprocess.ensure_dirs()

    if skip_spotify:
        print("Skipping Spotify pull (--skip-spotify).")
    else:
        # --- Spotify pull ---
        print("Authenticating with Spotify...")
        sp = spotify.get_client()
        tracks = spotify.fetch_top_tracks(sp)
        print(f"Fetched {len(tracks)} tracks from Spotify.")

        # --- Upsert into DB (skipping already-OK tracks unless --force) ---
        for track in tracks:
            existing = database.get_track(track["track_id"])
            if existing and existing["embedding_status"] == "ok" and not force:
                continue
            database.upsert_track(track)

    # --- Reset errored tracks that have no preview file (stale/expired URLs) ---
    db = database.get_db()
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tracks SET preview_url = NULL "
                "WHERE embedding_status = 'error' AND preview_path IS NULL"
            )

    # --- Lookup + download + preprocess (atomic per track) ---
    needs_work = [
        t for t in database.get_all_tracks()
        if t["preview_path"] is None
        and (force or t["embedding_status"] not in ("ok", "no_preview"))
    ]
    print(f"Looking up and downloading previews for {len(needs_work)} tracks (Deezer → iTunes)...")

    deezer_found = itunes_found = 0

    for row in tqdm(needs_work, desc="Lookup + download"):
        track_id = row["track_id"]

        # --- Look up preview URL ---
        url = deezer.lookup_preview_url(row["title"], row["artist"])
        if url:
            deezer_found += 1
        else:
            url = itunes.lookup_preview_url(row["title"], row["artist"])
            if url:
                itunes_found += 1

        if not url:
            database.update_embedding_status(track_id, "no_preview")
            continue

        # --- Download preview immediately (before URL expires) ---
        preview_rel = downloader.download_preview(track_id, url)
        if preview_rel is None:
            database.update_embedding_status(track_id, "error")
            continue

        # --- Download art ---
        art_rel = downloader.download_art(track_id, row["art_url"])
        if art_rel is None:
            database.update_embedding_status(track_id, "error")
            continue

        # --- Downsample art to 32x32 ---
        art_32_rel = preprocess.downsample_art(track_id, art_rel)

        # --- Persist URL and paths ---
        database.update_track_fields(
            track_id,
            preview_url=url,
            preview_path=str(preview_rel),
            art_path_original=str(art_rel),
            art_path_32=str(art_32_rel) if art_32_rel else None,
        )

    print(f"Found previews: Deezer={deezer_found}  iTunes={itunes_found}  "
          f"total new={deezer_found + itunes_found}")

    # --- Compute embeddings ---
    model, processor, device = embedder.load_model(allow_mps=True)

    embeddable = [
        t for t in database.get_all_tracks()
        if t["preview_path"] and (force or t["embedding_status"] == "pending")
    ]
    print(f"Computing embeddings for {len(embeddable)} tracks...")

    CHUNK_SAMPLES = 10 * 48000  # 10s at 48kHz
    N_CHUNKS = 3

    # Normalization target for the norm path (dual-index legacy; we store
    # the norm variant in PostgreSQL as the primary embedding)
    norm_target = float(config.get("pipeline", {}).get("norm_target", 0.12))

    new_ids = []
    new_embeddings = []

    for row in tqdm(embeddable, desc="Embedding"):
        track_id = row["track_id"]
        preview_path = DATA_DIR / row["preview_path"]

        try:
            audio, _ = librosa.load(str(preview_path), sr=48000, mono=True, dtype="float32")

            chunk_embs = []
            for i in range(N_CHUNKS):
                start = i * CHUNK_SAMPLES
                chunk = audio[start:start + CHUNK_SAMPLES]
                if len(chunk) == 0:
                    break
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                chunk_embs.append(
                    embedder.embed_audio(chunk, model, processor, device, norm_target=norm_target)
                )

            # Average chunks and re-normalize
            avg = np.mean(chunk_embs, axis=0).astype(np.float32)
            emb = (avg / (np.linalg.norm(avg) + 1e-8)).astype(np.float32)

            new_ids.append(track_id)
            new_embeddings.append(emb)
            database.update_embedding_status(track_id, "ok")
        except Exception as e:
            logger.error("Embedding failed for %s: %s", track_id, e)
            database.update_embedding_status(track_id, "error")

    # --- Store embeddings in PostgreSQL (replaces FAISS index build) ---
    if new_ids:
        embedder.save_embeddings_to_db(new_ids, np.stack(new_embeddings))
        print(f"Stored {len(new_ids)} embeddings in PostgreSQL.")

    # --- Summary ---
    status_counts: dict[str, int] = {}
    for t in database.get_all_tracks():
        s = t["embedding_status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    print("\n--- Pipeline complete ---")
    print(f"  ok:         {status_counts.get('ok', 0)}")
    print(f"  no_preview: {status_counts.get('no_preview', 0)}")
    print(f"  error:      {status_counts.get('error', 0)}")
    print(f"  pending:    {status_counts.get('pending', 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ALBart offline pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all tracks, ignoring existing status",
    )
    parser.add_argument(
        "--skip-spotify",
        action="store_true",
        help="Skip the Spotify pull (use tracks already in DB)",
    )
    args = parser.parse_args()
    run(force=args.force, skip_spotify=args.skip_spotify)


if __name__ == "__main__":
    main()
