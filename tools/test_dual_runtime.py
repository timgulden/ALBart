#!/usr/bin/env python3
"""Integration test for the dual-index runtime path.

Simulates what the runtime does for a set of sample recordings:
  1. Loads both FAISS indices (raw + norm)
  2. Embeds each sample the same way EmbeddingWorker does (3×10s chunks, averaged)
  3. Calls DualTrackLookup.query() and reports the fused top-10 results
  4. Shows the AliasTable sampling distribution

Usage:
    python3 tools/test_dual_runtime.py
    python3 tools/test_dual_runtime.py --samples data/samples/bts_dynamite.wav
    python3 tools/test_dual_runtime.py --n-draws 200  # simulate 200 alias samples per file
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np

from albart.pipeline.database import get_connection
from albart.pipeline.embedder import SAMPLE_RATE, embed_audio, load_model
from albart.runtime.embedder import _avg_normalize
from albart.runtime.lookup import DualTrackLookup
from albart.utils import load_config

CHUNK_SAMPLES = 10 * SAMPLE_RATE
N_CHUNKS = 3
WINDOW_SAMPLES = N_CHUNKS * CHUNK_SAMPLES


def get_track_label(conn, track_id: str) -> str:
    row = conn.execute(
        "SELECT title, artist FROM tracks WHERE track_id = ?", (track_id,)
    ).fetchone()
    if row:
        return f"{row['title']} — {row['artist']}"
    return track_id


def main():
    parser = argparse.ArgumentParser(description="Dual-runtime integration test")
    parser.add_argument("--samples", nargs="*", default=None,
                        help="WAV files to test (default: all in data/samples/)")
    parser.add_argument("--n-draws", type=int, default=100,
                        help="Alias table samples to simulate per file")
    parser.add_argument("--norm-target-raw",  type=float, default=0.0)
    parser.add_argument("--norm-target-norm", type=float, default=0.12)
    args = parser.parse_args()

    config = load_config()
    rt = config["runtime"]

    samples_dir = Path(__file__).parent.parent / "data" / "samples"
    if args.samples:
        sample_files = [Path(s) for s in args.samples]
    else:
        sample_files = sorted(samples_dir.glob("*.wav"))

    if not sample_files:
        print("No sample files found.")
        return

    print("Loading CLAP model...")
    model, processor, device = load_model()
    print(f"  Model on {device}")

    print("Loading dual FAISS indices...")
    lookup = DualTrackLookup()

    conn = get_connection()

    rank_fusion_k      = rt.get("rank_fusion_k", 60)
    sampling_top_n     = rt.get("sampling_top_n", 20)
    sampling_rank_decay = rt.get("sampling_rank_decay", 0.7)
    dwell_k            = rt["dwell_k"]
    dwell_floor        = rt["dwell_floor"]
    brightness_k       = rt["brightness_k"]
    brightness_floor   = rt["brightness_floor"]
    brightness_power   = rt["brightness_power"]
    min_dwell          = rt["min_dwell_seconds"]
    max_dwell          = rt["max_dwell_seconds"]

    for fpath in sample_files:
        print(f"\n{'='*70}")
        print(f"File: {fpath.name}")

        audio, _ = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True, dtype="float32")
        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"  Duration: {len(audio)/SAMPLE_RATE:.1f}s  RMS: {rms:.4f}")

        if len(audio) < WINDOW_SAMPLES:
            print("  (too short — padding)")
            audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))

        # Use the middle 30s window for a representative embedding
        mid = max(0, len(audio) // 2 - WINDOW_SAMPLES // 2)
        window = audio[mid:mid + WINDOW_SAMPLES]

        # Embed at both normalization levels (mirrors EmbeddingWorker._run)
        chunk_embs_raw  = []
        chunk_embs_norm = []
        for i in range(N_CHUNKS):
            chunk = window[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
            chunk_embs_raw.append(
                embed_audio(chunk, model, processor, device, norm_target=args.norm_target_raw)
            )
            chunk_embs_norm.append(
                embed_audio(chunk, model, processor, device, norm_target=args.norm_target_norm)
            )

        emb_raw  = _avg_normalize(chunk_embs_raw)
        emb_norm = _avg_normalize(chunk_embs_norm)

        # Build alias table exactly as the runtime would
        table = lookup.query(
            emb_raw=emb_raw,
            emb_norm=emb_norm,
            rank_fusion_k=rank_fusion_k,
            sampling_top_n=sampling_top_n,
            sampling_rank_decay=sampling_rank_decay,
            dwell_k=dwell_k,
            dwell_floor=dwell_floor,
            brightness_k=brightness_k,
            brightness_floor=brightness_floor,
            brightness_power=brightness_power,
            min_dwell=min_dwell,
            max_dwell=max_dwell,
        )

        print(f"  Brightness: {table.brightness:.3f}  Dwell: {table.dwell_times[0]:.1f}s")
        print(f"\n  Top-{min(10, len(table.track_ids))} fused results:")
        for i, (tid, w) in enumerate(zip(table.track_ids[:10], table.weights[:10]), 1):
            label = get_track_label(conn, tid)
            pct = 100 * w / table.weights.sum()
            print(f"    {i:2d}. {label[:55]:<55s}  weight={w:.3f} ({pct:.1f}%)")

        # Simulate alias sampling
        draw_counts: Counter = Counter()
        for _ in range(args.n_draws):
            tid, _ = table.sample()
            draw_counts[tid] += 1

        print(f"\n  Alias sampling ({args.n_draws} draws) — top 5:")
        for tid, count in draw_counts.most_common(5):
            label = get_track_label(conn, tid)
            print(f"    {count:4d}/{args.n_draws}  ({100*count/args.n_draws:5.1f}%)  {label[:55]}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
