"""
Record audio, embed it with the dual-index approach, and query both FAISS indices.
Shows fused top-20 results exactly as the runtime would see them.

Usage:
    python tools/query_live.py [--seconds 30] [--silence]
    python tools/query_live.py --seconds 10  # single 10s chunk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import sounddevice as sd

from albart.pipeline.embedder import embed_audio, load_model
from albart.pipeline.database import DB_PATH, get_connection
from albart.runtime.embedder import _avg_normalize
from albart.runtime.lookup import DualTrackLookup
from albart.utils import load_config

SAMPLE_RATE = 48000
CHUNK_SECONDS = 10


def record(seconds: int, countdown: int = 0, device=None) -> np.ndarray:
    if countdown > 0:
        for i in range(countdown, 0, -1):
            print(f"  Starting in {i}...", flush=True)
            import time; time.sleep(1)
    print(f"*** RECORDING {seconds}s NOW ***", flush=True)
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    audio = audio.flatten()
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"Captured {len(audio):,} samples  RMS={rms:.4f}")
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Live dual-index query (mirrors runtime behavior)")
    parser.add_argument("--seconds", type=int, default=30,
                        help="Total recording seconds (must be multiple of 10, default: 30)")
    parser.add_argument("--silence", action="store_true",
                        help="Skip recording, use silence (tests index with zero embedding)")
    parser.add_argument("--countdown", type=int, default=5,
                        help="Seconds to count down before first chunk (default: 5)")
    parser.add_argument("--device", type=str, default=None,
                        help="Input device name or index (default: system default)")
    parser.add_argument("--norm-target-raw",  type=float, default=0.0)
    parser.add_argument("--norm-target-norm", type=float, default=0.12)
    args = parser.parse_args()

    # Resolve device
    audio_device = None
    if args.device is not None:
        try:
            audio_device = int(args.device)
        except ValueError:
            devs = sd.query_devices()
            matches = [i for i, d in enumerate(devs)
                       if args.device.lower() in d["name"].lower() and d["max_input_channels"] > 0]
            if not matches:
                print(f"No input device matching '{args.device}'. Available inputs:")
                for i, d in enumerate(devs):
                    if d["max_input_channels"] > 0:
                        print(f"  [{i}] {d['name']}")
                sys.exit(1)
            audio_device = matches[0]
            print(f"Using device [{audio_device}]: {sd.query_devices(audio_device)['name']}")

    config = load_config()
    rt = config["runtime"]
    norm_target_raw  = args.norm_target_raw  or float(rt.get("norm_target_raw",  0.0))
    norm_target_norm = args.norm_target_norm or float(rt.get("norm_target_norm", 0.12))
    rank_fusion_k       = rt.get("rank_fusion_k", 60)
    sampling_top_n      = rt.get("sampling_top_n", 20)
    sampling_rank_decay = rt.get("sampling_rank_decay", 0.7)
    dwell_k             = rt["dwell_k"]
    dwell_floor         = rt["dwell_floor"]
    brightness_k        = rt["brightness_k"]
    brightness_floor    = rt["brightness_floor"]
    brightness_power    = rt["brightness_power"]
    min_dwell           = rt["min_dwell_seconds"]
    max_dwell           = rt["max_dwell_seconds"]

    n_chunks = max(1, args.seconds // CHUNK_SECONDS)

    print("Loading CLAP model...")
    model, processor, device_str = load_model()
    print(f"Model on {device_str}")

    print("Loading dual FAISS indices...")
    lookup = DualTrackLookup()

    # Record / generate chunks
    if args.silence:
        print(f"Using silence ({n_chunks} chunks of {CHUNK_SECONDS}s)...")
        chunks = [np.zeros(CHUNK_SECONDS * SAMPLE_RATE, dtype=np.float32)] * n_chunks
    else:
        chunks = []
        for i in range(n_chunks):
            label = f"chunk {i+1}/{n_chunks}"
            cd = args.countdown if i == 0 else 0
            print(f"--- Recording {label} ---")
            chunks.append(record(CHUNK_SECONDS, countdown=cd, device=audio_device))

    # Embed each chunk with both norm targets
    chunk_embs_raw  = []
    chunk_embs_norm = []
    for i, chunk in enumerate(chunks):
        emb_raw  = embed_audio(chunk, model, processor, device_str, norm_target=norm_target_raw)
        emb_norm = embed_audio(chunk, model, processor, device_str, norm_target=norm_target_norm)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        print(f"Chunk {i+1}: RMS={rms:.4f}  "
              f"|emb_raw|={np.linalg.norm(emb_raw):.4f}  "
              f"|emb_norm|={np.linalg.norm(emb_norm):.4f}")
        chunk_embs_raw.append(emb_raw)
        chunk_embs_norm.append(emb_norm)

    emb_raw  = _avg_normalize(chunk_embs_raw)
    emb_norm = _avg_normalize(chunk_embs_norm)

    print(f"\nFinal embedding: |raw|={np.linalg.norm(emb_raw):.4f}  |norm|={np.linalg.norm(emb_norm):.4f}")

    # Query dual index (mirrors DisplayLoop._check_embedding_queue)
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

    conn = get_connection(DB_PATH)

    print(f"\nBrightness: {table.brightness:.3f}   d_min_raw: {table.d_min_raw:.4f}")
    print(f"\n{'Rank':<5} {'Weight%':>8} {'Dwell':>6}  Track")
    print("-" * 80)

    for i, (tid, w, dwell) in enumerate(
        zip(table.track_ids[:20], table.weights[:20], table.dwell_times[:20]), 1
    ):
        row = conn.execute(
            "SELECT title, artist FROM tracks WHERE track_id=?", (tid,)
        ).fetchone()
        label = f"{row['title'][:38]}  —  {row['artist'][:25]}" if row else tid
        pct = 100.0 * w / table.weights.sum()
        print(f"{i:<5} {pct:>7.2f}% {dwell:>5.1f}s  {label}")

    conn.close()

    total = table.weights.sum()
    print(f"\nTop-1 share: {100*table.weights[0]/total:.1f}%  "
          f"Top-5: {100*table.weights[:5].sum()/total:.1f}%  "
          f"Top-10: {100*table.weights[:10].sum()/total:.1f}%")
    print(f"\nConfig: rank_fusion_k={rank_fusion_k}  "
          f"sampling_top_n={sampling_top_n}  "
          f"sampling_rank_decay={sampling_rank_decay}")


if __name__ == "__main__":
    main()
