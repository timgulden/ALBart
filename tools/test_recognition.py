"""
Test embedding recognition using preview files directly.

Loads a track's own preview file, embeds it with the dual-index approach,
and queries both FAISS indices via RRF. A working system should rank the
track #1 in the fused results with a low raw distance.

Usage:
    python tools/test_recognition.py                        # list ok tracks
    python tools/test_recognition.py teardrop               # search by title/artist
    python tools/test_recognition.py "rolling stones"
    python tools/test_recognition.py --id 67Hna13dNDkZvBpTXRIaOJ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np

from albart.pipeline.database import DB_PATH, get_connection
from albart.pipeline.embedder import embed_audio, load_model, SAMPLE_RATE
from albart.runtime.embedder import _avg_normalize
from albart.runtime.lookup import DualTrackLookup
from albart.utils import DATA_DIR, load_config

CHUNK_SECONDS = 10
N_CHUNKS = 3
WINDOW_SAMPLES = N_CHUNKS * CHUNK_SECONDS * SAMPLE_RATE


def list_tracks(conn) -> None:
    rows = conn.execute(
        "SELECT track_id, title, artist FROM tracks "
        "WHERE embedding_status='ok' AND preview_path IS NOT NULL "
        "ORDER BY artist, title"
    ).fetchall()
    print(f"{len(rows)} tracks available for recognition testing:\n")
    for r in rows:
        print(f"  {r['artist'][:35]:<35}  {r['title'][:40]}")


def find_tracks(conn, query: str) -> list:
    q = f"%{query}%"
    return conn.execute(
        "SELECT track_id, title, artist, preview_path FROM tracks "
        "WHERE embedding_status='ok' AND preview_path IS NOT NULL "
        "AND (LOWER(title) LIKE LOWER(?) OR LOWER(artist) LIKE LOWER(?))",
        (q, q),
    ).fetchall()


def test_track(row, conn, model, processor, device, lookup: DualTrackLookup,
               norm_target_raw: float, norm_target_norm: float) -> None:
    track_id = row["track_id"]
    title    = row["title"]
    artist   = row["artist"]
    preview_path = DATA_DIR / row["preview_path"]

    print(f"\nTesting: {title} — {artist}")
    print(f"  File: {preview_path.name}")

    audio, _ = librosa.load(
        str(preview_path), sr=SAMPLE_RATE, mono=True, dtype="float32",
    )
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"  Duration: {len(audio)/SAMPLE_RATE:.1f}s   RMS: {rms:.4f}")

    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))

    # Use the middle 30s window
    mid = max(0, len(audio) // 2 - WINDOW_SAMPLES // 2)
    window = audio[mid:mid + WINDOW_SAMPLES]

    # Embed 3 × 10s chunks at both normalization levels
    chunk_embs_raw  = []
    chunk_embs_norm = []
    for i in range(N_CHUNKS):
        chunk = window[i * CHUNK_SECONDS * SAMPLE_RATE:(i + 1) * CHUNK_SECONDS * SAMPLE_RATE]
        chunk_embs_raw.append(
            embed_audio(chunk, model, processor, device, norm_target=norm_target_raw)
        )
        chunk_embs_norm.append(
            embed_audio(chunk, model, processor, device, norm_target=norm_target_norm)
        )

    emb_raw  = _avg_normalize(chunk_embs_raw)
    emb_norm = _avg_normalize(chunk_embs_norm)

    # Query dual index (same path as the runtime)
    table = lookup.query(
        emb_raw=emb_raw,
        emb_norm=emb_norm,
    )

    found_rank = None
    print(f"\n  d_min_raw={table.d_min_raw:.4f}  brightness={table.brightness:.3f}")
    print(f"\n  {'Rank':<5} {'Weight%':>8}  Track")
    print(f"  {'-'*65}")

    for rank, (tid, w) in enumerate(zip(table.track_ids, table.weights), 1):
        meta = conn.execute(
            "SELECT title, artist FROM tracks WHERE track_id=?", (tid,)
        ).fetchone()
        label = f"{meta['title'][:38]}  —  {meta['artist'][:25]}" if meta else tid
        pct = 100.0 * w / table.weights.sum()
        marker = " ◀ TARGET" if tid == track_id else ""
        print(f"  {rank:<5} {pct:>7.2f}%  {label}{marker}")
        if tid == track_id:
            found_rank = rank

    if found_rank is not None and found_rank <= len(table.track_ids):
        print(f"\n  ✓ Target at fused rank {found_rank}  (d_min_raw={table.d_min_raw:.4f})")
    else:
        print(f"\n  ✗ Target not in fused top-{len(table.track_ids)}")
        # Report where it falls in raw and norm individually
        from albart.pipeline.embedder import (
            FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH,
            FAISS_NORM_INDEX_PATH, FAISS_NORM_IDS_PATH,
            load_index,
        )
        raw_idx, raw_ids = load_index(FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH)
        raw_ids_list = [str(t) for t in raw_ids]
        q = emb_raw.reshape(1, -1)
        dists, idxs = raw_idx.search(q, raw_idx.ntotal)
        for rank, idx in enumerate(idxs[0], 1):
            if raw_ids_list[int(idx)] == track_id:
                print(f"  Raw index rank: {rank}  dist={dists[0][rank-1]:.4f}")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Test dual-index recognition against preview files")
    parser.add_argument("query", nargs="?", help="Title/artist search term")
    parser.add_argument("--id", help="Specific track_id")
    parser.add_argument("--norm-target-raw",  type=float, default=0.0)
    parser.add_argument("--norm-target-norm", type=float, default=0.12)
    args = parser.parse_args()

    config = load_config()
    rt = config["runtime"]
    norm_target_raw  = args.norm_target_raw  or float(rt.get("norm_target_raw",  0.0))
    norm_target_norm = args.norm_target_norm or float(rt.get("norm_target_norm", 0.12))

    conn = get_connection(DB_PATH)

    if not args.query and not args.id:
        list_tracks(conn)
        conn.close()
        return

    if args.id:
        rows = conn.execute(
            "SELECT track_id, title, artist, preview_path FROM tracks WHERE track_id=?",
            (args.id,)
        ).fetchall()
    else:
        rows = find_tracks(conn, args.query)

    if not rows:
        print(f"No ok tracks found matching '{args.query or args.id}'")
        conn.close()
        sys.exit(1)

    print("Loading CLAP model...")
    model, processor, device = load_model()
    print(f"Model on {device}")

    print("Loading dual FAISS indices...")
    lookup = DualTrackLookup()

    for row in rows:
        test_track(row, conn, model, processor, device, lookup,
                   norm_target_raw, norm_target_norm)

    conn.close()


if __name__ == "__main__":
    main()
