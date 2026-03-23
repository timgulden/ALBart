#!/usr/bin/env python3
"""Sweep RMS normalization targets on query-side audio to find optimal recognition.

The FAISS index was built WITHOUT normalization (raw Deezer previews).
This script tests whether normalizing room recordings to various RMS targets
improves recognition accuracy against that raw-audio index.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Set up imports before anything else
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa

from albart.pipeline.embedder import embed_audio, load_model, EMBEDDING_DIM
from albart.pipeline.database import DB_PATH, get_connection
from albart.utils import DATA_DIR

SAMPLE_RATE = 48000
CHUNK_SAMPLES = 10 * SAMPLE_RATE
N_CHUNKS = 3
WINDOW_SAMPLES = N_CHUNKS * CHUNK_SAMPLES
TOP_K = 20
ALPHA = 0.5

NORM_TARGETS = [None, 0.05, 0.08, 0.1, 0.12, 0.15, 0.167, 0.2, 0.25, 0.3]

SAMPLES = [
    ("BTS", "data/samples/bts_dynamite.wav", "Dynamite"),
    ("RRH", "data/samples/red_right_hand.wav", "Red Right Hand"),
    ("movo", "data/samples/rrh_movo.wav", "Red Right Hand"),
]


def lookup_target_ids(search_term: str) -> set[str]:
    """Find track IDs matching a search term in the database."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT track_id FROM tracks WHERE title LIKE ? OR artist LIKE ?",
        (f"%{search_term}%", f"%{search_term}%"),
    ).fetchall()
    conn.close()
    ids = {r["track_id"] for r in rows}
    print(f"  Found {len(ids)} target track(s) for '{search_term}': {ids}")
    return ids


def rms_normalize(audio: np.ndarray, target: float) -> np.ndarray:
    """RMS-normalize audio to a target level."""
    rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
    return (audio * (target / rms)).astype(np.float32)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def sweep_file(
    audio: np.ndarray,
    target_ids: set[str],
    norm_target: float | None,
    model,
    processor,
    device,
    index,
    track_ids: np.ndarray,
) -> dict:
    """Run sliding-window sweep on one file with one norm target.

    Returns dict with hit counts for top-1, top-5, top-10.
    """
    n_windows = (len(audio) - WINDOW_SAMPLES) // SAMPLE_RATE + 1
    if n_windows <= 0:
        return {"n": 0, "top1": 0, "top5": 0, "top10": 0}

    hits = {"n": 0, "top1": 0, "top5": 0, "top10": 0}
    ema_vec = None

    for w in range(n_windows):
        start = w * SAMPLE_RATE
        window = audio[start : start + WINDOW_SAMPLES]

        # Split into 3 chunks and embed each
        chunk_vecs = []
        for c in range(N_CHUNKS):
            chunk = window[c * CHUNK_SAMPLES : (c + 1) * CHUNK_SAMPLES]

            # Apply RMS normalization if target is set
            if norm_target is not None:
                chunk = rms_normalize(chunk, norm_target)

            vec = embed_audio(chunk, model, processor, device)
            chunk_vecs.append(vec)

        # Average chunks and L2-normalize
        avg_vec = l2_normalize(np.mean(chunk_vecs, axis=0))

        # EMA smoothing
        if ema_vec is None:
            ema_vec = avg_vec.copy()
        else:
            ema_vec = ALPHA * avg_vec + (1.0 - ALPHA) * ema_vec
            ema_vec = l2_normalize(ema_vec)

        # Query FAISS
        query = ema_vec.reshape(1, -1).astype(np.float32)
        distances, indices = index.search(query, TOP_K)

        result_ids = [track_ids[i] for i in indices[0] if i >= 0]

        hits["n"] += 1
        if len(result_ids) > 0 and result_ids[0] in target_ids:
            hits["top1"] += 1
        if any(rid in target_ids for rid in result_ids[:5]):
            hits["top5"] += 1
        if any(rid in target_ids for rid in result_ids[:10]):
            hits["top10"] += 1

    return hits


def main():
    import faiss

    print("=" * 70)
    print("ALBart Normalization Target Sweep")
    print("=" * 70)

    # Load model once
    print("\nLoading CLAP model...")
    model, processor, device = load_model()
    print(f"  Model loaded on {device}")

    # Load FAISS index once
    print("Loading FAISS index...")
    index = faiss.read_index(str(DATA_DIR / "faiss.index"))
    track_ids = np.load(str(DATA_DIR / "faiss_ids.npy"), allow_pickle=True)
    print(f"  Index has {index.ntotal} vectors")

    # Load audio files and look up targets
    audio_data = []
    for label, rel_path, search_term in SAMPLES:
        path = Path(__file__).parent.parent / rel_path
        print(f"\nLoading {label}: {path}")
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, dtype="float32")
        rms = float(np.sqrt(np.mean(audio ** 2)))
        n_windows = (len(audio) - WINDOW_SAMPLES) // SAMPLE_RATE + 1
        print(f"  Duration: {len(audio)/SAMPLE_RATE:.1f}s, RMS: {rms:.4f}, Windows: {n_windows}")
        target_ids = lookup_target_ids(search_term)
        audio_data.append((label, audio, target_ids))

    # Results: {label: {norm_target: hits_dict}}
    results = {label: {} for label, _, _ in audio_data}

    # Run sweep
    print("\n" + "=" * 70)
    print("Running sweep...")
    print("=" * 70)

    for norm_target in NORM_TARGETS:
        tgt_str = "None" if norm_target is None else f"{norm_target}"
        print(f"\n--- Norm target: {tgt_str} ---")

        for label, audio, target_ids in audio_data:
            hits = sweep_file(
                audio, target_ids, norm_target,
                model, processor, device,
                index, track_ids,
            )
            results[label][norm_target] = hits
            n = hits["n"]
            if n > 0:
                print(
                    f"  {label:5s}: #1={hits['top1']:3d}/{n} ({100*hits['top1']/n:5.1f}%)  "
                    f"top5={hits['top5']:3d}/{n} ({100*hits['top5']/n:5.1f}%)  "
                    f"top10={hits['top10']:3d}/{n} ({100*hits['top10']/n:5.1f}%)"
                )
            else:
                print(f"  {label:5s}: no windows")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Header
    col_w = 7
    header_targets = [("None" if t is None else str(t)) for t in NORM_TARGETS]
    header = f"{'Target':<12s}" + "".join(f"{t:>{col_w}s}" for t in header_targets)
    print(header)
    print("-" * len(header))

    # Rows
    for label, _, _ in audio_data:
        for metric, key in [("#1", "top1"), ("top5", "top5"), ("top10", "top10")]:
            row_label = f"{label} {metric}"
            cells = []
            for norm_target in NORM_TARGETS:
                h = results[label].get(norm_target, {"n": 0, key: 0})
                n = h["n"]
                if n > 0:
                    pct = 100 * h[key] / n
                    cells.append(f"{pct:.0f}%")
                else:
                    cells.append("-")
            print(f"{row_label:<12s}" + "".join(f"{c:>{col_w}s}" for c in cells))
        print()  # blank line between samples

    print("Done.")


if __name__ == "__main__":
    main()
