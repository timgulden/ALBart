"""
Record a series of back-to-back 10s audio chunks and query the FAISS index
for each one.  Tracks the rank of target tracks across the sweep so you can
see how recognition changes as different parts of a song play.

Usage:
    python tools/query_sweep.py --device BlackHole --chunks 18 --target "Gimme Shelter"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import sounddevice as sd

from albart.pipeline.embedder import embed_audio, load_model
from albart.pipeline.database import DB_PATH, get_connection
from albart.utils import DATA_DIR, load_config

SAMPLE_RATE = 48000


def record_chunk(seconds: int, device=None) -> np.ndarray:
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.flatten()


def query_index(embedding: np.ndarray, index, track_ids, config: dict):
    """Return sorted list of (track_id, distance)."""
    rt = config["runtime"]
    n = index.ntotal
    q = embedding.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(q, n)

    track_min_dist: dict[str, float] = {}
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        tid = str(track_ids[idx])
        if tid not in track_min_dist or dist < track_min_dist[tid]:
            track_min_dist[tid] = float(dist)

    return sorted(track_min_dist.items(), key=lambda x: x[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--chunks", type=int, default=18,
                        help="Number of 10s chunks to record (default: 18 = 3 minutes)")
    parser.add_argument("--target", type=str, default="Gimme Shelter",
                        help="Substring to match against track titles for rank tracking")
    parser.add_argument("--index", type=str, default="faiss",
                        help="Index filename prefix to use (default: faiss → faiss.index / faiss_ids.npy)")
    parser.add_argument("--file", type=str, default=None,
                        help="Use a pre-recorded WAV file instead of live mic")
    parser.add_argument("--countdown", type=int, default=5)
    args = parser.parse_args()

    config = load_config()

    # Resolve device
    device = None
    if args.device is not None:
        try:
            device = int(args.device)
        except ValueError:
            devs = sd.query_devices()
            matches = [i for i, d in enumerate(devs)
                       if args.device.lower() in d["name"].lower()
                       and d["max_input_channels"] > 0]
            if not matches:
                print(f"No input device matching '{args.device}'.")
                sys.exit(1)
            device = matches[0]
            print(f"Using device [{device}]: {sd.query_devices(device)['name']}")

    # Load model and index once
    print("Loading CLAP model...")
    model, processor, device_str = load_model()
    print(f"Model on {device_str}")

    import faiss  # lazy — must come after CLAP model load to avoid BLAS conflict
    index = faiss.read_index(str(DATA_DIR / f"{args.index}.index"))
    track_ids = np.load(str(DATA_DIR / f"{args.index}_ids.npy"), allow_pickle=True)
    n_unique = len(set(str(t) for t in track_ids))
    print(f"Index: {index.ntotal} vectors / {n_unique} unique tracks")

    # Find target track IDs
    conn = get_connection(DB_PATH)
    target_rows = conn.execute(
        "SELECT track_id, title, artist FROM tracks WHERE title LIKE ?",
        (f"%{args.target}%",),
    ).fetchall()
    target_ids = {row[0]: f"{row[1]} — {row[2]}" for row in target_rows}
    conn.close()

    if target_ids:
        print(f"\nTracking {len(target_ids)} target track(s):")
        for tid, label in target_ids.items():
            print(f"  {label}")
    else:
        print(f"\nWarning: no tracks found matching '{args.target}'")

    # Load file or prepare for live recording
    chunk_secs = 10
    file_audio = None
    if args.file:
        import librosa
        print(f"\nLoading {args.file}...")
        file_audio, _ = librosa.load(args.file, sr=SAMPLE_RATE, mono=True, dtype="float32")
        duration = len(file_audio) / SAMPLE_RATE
        available_chunks = int(duration // chunk_secs)
        print(f"Loaded {duration:.1f}s  ({available_chunks} chunks available)")
        if args.chunks > available_chunks:
            print(f"Warning: requested {args.chunks} chunks but only {available_chunks} available — capping.")
            args.chunks = available_chunks
    else:
        # Countdown for live recording
        print(f"\nStarting in ", end="", flush=True)
        for i in range(args.countdown, 0, -1):
            print(f"{i}... ", end="", flush=True)
            time.sleep(1)
        print("GO")

    print()
    results = []  # list of (chunk_num, top1_label, top1_dist, target_ranks)

    print(f"{'#':<4} {'Time':>6}  {'Dist':>6}  {'Rank(s)':>12}  Top match")
    print("-" * 75)

    for chunk_num in range(1, args.chunks + 1):
        t_start = (chunk_num - 1) * chunk_secs
        t_end   = chunk_num * chunk_secs

        if file_audio is not None:
            start_sample = t_start * SAMPLE_RATE
            end_sample   = t_end * SAMPLE_RATE
            audio = file_audio[start_sample:end_sample]
        else:
            audio = record_chunk(chunk_secs, device=device)
        rms = float(np.sqrt(np.mean(audio ** 2)))

        embedding = embed_audio(audio, model, processor, device_str)
        sorted_results = query_index(embedding, index, track_ids, config)

        # Build rank map: track_id -> rank (1-based)
        rank_map = {tid: rank + 1 for rank, (tid, _) in enumerate(sorted_results)}

        top_tid, top_dist = sorted_results[0]
        conn = get_connection(DB_PATH)
        row = conn.execute(
            "SELECT title, artist FROM tracks WHERE track_id=?", (top_tid,)
        ).fetchone()
        conn.close()
        top_label = f"{row['title'][:30]} — {row['artist'][:20]}" if row else top_tid

        # Ranks for all target tracks
        target_rank_strs = []
        for tid, label in target_ids.items():
            rank = rank_map.get(tid, -1)
            target_rank_strs.append(f"#{rank}")
        rank_display = " ".join(target_rank_strs) if target_rank_strs else "n/a"

        print(f"{chunk_num:<4} {t_start:>3}-{t_end:<3}s  {top_dist:>6.4f}  {rank_display:>12}  {top_label}")

        results.append({
            "chunk": chunk_num,
            "t_start": t_start,
            "t_end": t_end,
            "rms": rms,
            "top_dist": top_dist,
            "top_label": top_label,
            "target_ranks": {tid: rank_map.get(tid, -1) for tid in target_ids},
        })

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    for tid, label in target_ids.items():
        ranks = [r["target_ranks"][tid] for r in results if r["target_ranks"].get(tid, -1) > 0]
        if ranks:
            top5_count = sum(1 for r in ranks if r <= 5)
            top10_count = sum(1 for r in ranks if r <= 10)
            print(f"\n{label}")
            print(f"  Best rank:    #{min(ranks)}")
            print(f"  Median rank:  #{sorted(ranks)[len(ranks)//2]}")
            print(f"  In top 5:     {top5_count}/{len(results)} chunks")
            print(f"  In top 10:    {top10_count}/{len(results)} chunks")
            rank_by_chunk = "  ".join(
                f"{r['t_start']:>2}s:#{r['target_ranks'].get(tid,-1)}"
                for r in results
            )
            print(f"  Per chunk:    {rank_by_chunk}")


if __name__ == "__main__":
    main()
