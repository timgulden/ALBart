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
    _ = config  # reserved for future pipeline config keys

    # --- Setup ---
    database.init_db()
    downloader.ensure_dirs()
    preprocess.ensure_dirs()

    conn = database.get_connection()

    if skip_spotify:
        print("Skipping Spotify pull (--skip-spotify).")
    else:
        # --- Spotify pull ---
        print("Authenticating with Spotify...")
        sp = spotify.get_client()
        tracks = spotify.fetch_top_tracks(sp)
        print(f"Fetched {len(tracks)} tracks from Spotify.")

        # --- Upsert into DB (skipping already-OK tracks unless --force) ---
        with conn:
            for track in tracks:
                existing = database.get_track(conn, track["track_id"])
                if existing and existing["embedding_status"] == "ok" and not force:
                    continue
                database.upsert_track(conn, track)

    # --- Reset errored tracks that have no preview file (stale/expired URLs) ---
    # Their stored preview_url is either expired or invalid — clear it for re-lookup.
    with conn:
        conn.execute(
            "UPDATE tracks SET preview_url=NULL WHERE embedding_status='error' AND preview_path IS NULL"
        )
    reset_count = conn.execute(
        "SELECT COUNT(*) FROM tracks WHERE embedding_status='error' AND preview_path IS NULL"
    ).fetchone()[0]
    # (count is 0 after the reset above; this just confirms the clear ran)

    # --- Lookup + download + preprocess (atomic per track) ---
    # Deezer URLs expire ~90 minutes after generation, so we download immediately
    # after fetching rather than storing URLs for a later download pass.
    needs_work = [
        t for t in database.get_all_tracks(conn)
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
            with conn:
                database.update_embedding_status(conn, track_id, "no_preview")
            continue

        # --- Download preview immediately (before URL expires) ---
        preview_rel = downloader.download_preview(track_id, url)
        if preview_rel is None:
            with conn:
                database.update_embedding_status(conn, track_id, "error")
            continue

        # --- Download art ---
        art_rel = downloader.download_art(track_id, row["art_url"])
        if art_rel is None:
            with conn:
                database.update_embedding_status(conn, track_id, "error")
            continue

        # --- Downsample art to 32x32 ---
        art_32_rel = preprocess.downsample_art(track_id, art_rel)

        # --- Persist URL and paths ---
        with conn:
            conn.execute(
                """UPDATE tracks
                   SET preview_url=?, preview_path=?, art_path_original=?, art_path_32=?
                   WHERE track_id=?""",
                (
                    url,
                    str(preview_rel),
                    str(art_rel),
                    str(art_32_rel) if art_32_rel else None,
                    track_id,
                ),
            )

    print(f"Found previews: Deezer={deezer_found}  iTunes={itunes_found}  "
          f"total new={deezer_found + itunes_found}")

    # --- Compute embeddings ---
    model, processor, device = embedder.load_model(allow_mps=True)

    embeddable = [
        t for t in database.get_all_tracks(conn)
        if t["preview_path"] and (force or t["embedding_status"] == "pending")
    ]
    print(f"Computing embeddings for {len(embeddable)} tracks...")

    CHUNK_SAMPLES = 10 * 48000  # 10s at 48kHz (processor max_length_s for unfused models)
    N_CHUNKS = 3                 # embed 3 chunks, average into one vector per track

    # Dual-index normalization target (read from config with sensible default)
    norm_target = float(config.get("pipeline", {}).get("norm_target", 0.12))

    new_embeddings_raw  = []
    new_embeddings_norm = []
    new_ids = []

    for row in tqdm(embeddable, desc="Embedding"):
        track_id = row["track_id"]
        preview_path = DATA_DIR / row["preview_path"]

        try:
            audio, _ = librosa.load(str(preview_path), sr=48000, mono=True, dtype="float32")
            # Embed N_CHUNKS × 10s windows at both raw and norm targets in one pass.
            # Average and renormalize (mean of unit vectors is not a unit vector).
            chunk_embs_raw  = []
            chunk_embs_norm = []
            for i in range(N_CHUNKS):
                start = i * CHUNK_SAMPLES
                chunk = audio[start:start + CHUNK_SAMPLES]
                if len(chunk) == 0:
                    break
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                chunk_embs_raw.append(
                    embedder.embed_audio(chunk, model, processor, device, norm_target=0.0)
                )
                chunk_embs_norm.append(
                    embedder.embed_audio(chunk, model, processor, device, norm_target=norm_target)
                )

            def _avg_norm(embs):
                avg = np.mean(embs, axis=0).astype(np.float32)
                return (avg / (np.linalg.norm(avg) + 1e-8)).astype(np.float32)

            new_embeddings_raw.append(_avg_norm(chunk_embs_raw))
            new_embeddings_norm.append(_avg_norm(chunk_embs_norm))
            new_ids.append(track_id)
            with conn:
                database.update_embedding_status(conn, track_id, "ok")
        except Exception as e:
            logger.error("Embedding failed for %s: %s", track_id, e)
            with conn:
                database.update_embedding_status(conn, track_id, "error")

    # --- Rebuild dual FAISS indices (merge new with previously stored) ---
    RAW_EMB_STORE  = DATA_DIR / "embeddings_raw.npy"
    NORM_EMB_STORE = DATA_DIR / "embeddings_norm.npy"
    IDS_STORE      = DATA_DIR / "faiss_raw_ids.npy"  # both indices share the same ID set

    if not force and RAW_EMB_STORE.exists() and IDS_STORE.exists() and new_ids:
        prev_raw  = np.load(str(RAW_EMB_STORE))
        prev_norm = np.load(str(NORM_EMB_STORE))
        prev_ids  = np.load(str(IDS_STORE), allow_pickle=True).tolist()
        new_id_set = set(new_ids)
        keep = [i for i, tid in enumerate(prev_ids) if tid not in new_id_set]
        all_embeddings_raw  = np.concatenate(
            [prev_raw[keep],  np.stack(new_embeddings_raw)],  axis=0
        ).astype(np.float32)
        all_embeddings_norm = np.concatenate(
            [prev_norm[keep], np.stack(new_embeddings_norm)], axis=0
        ).astype(np.float32)
        all_ids = [prev_ids[i] for i in keep] + new_ids
    elif new_ids:
        all_embeddings_raw  = np.stack(new_embeddings_raw).astype(np.float32)
        all_embeddings_norm = np.stack(new_embeddings_norm).astype(np.float32)
        all_ids = new_ids
    else:
        all_embeddings_raw = all_embeddings_norm = None
        all_ids = []

    if all_ids:
        np.save(str(RAW_EMB_STORE),  all_embeddings_raw)
        np.save(str(NORM_EMB_STORE), all_embeddings_norm)
        embedder.build_and_save_index(
            all_embeddings_raw,  all_ids,
            index_path=embedder.FAISS_RAW_INDEX_PATH,
            ids_path=embedder.FAISS_RAW_IDS_PATH,
        )
        embedder.build_and_save_index(
            all_embeddings_norm, all_ids,
            index_path=embedder.FAISS_NORM_INDEX_PATH,
            ids_path=embedder.FAISS_NORM_IDS_PATH,
        )
        print(f"FAISS index built with {len(all_ids)} tracks.")

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
    parser.add_argument(
        "--skip-spotify",
        action="store_true",
        help="Skip the Spotify pull (use tracks already in DB)",
    )
    args = parser.parse_args()
    run(force=args.force, skip_spotify=args.skip_spotify)


if __name__ == "__main__":
    main()
