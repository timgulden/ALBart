#!/usr/bin/env python3
"""Test dual-index approach: query both a raw (no-norm) and normalized index.

For each query window, embed the audio twice:
  1. Raw (no normalization) → query the raw index
  2. RMS-normalized to a target → query the normalized index

Merge results by track_id (min distance wins). This should combine the
strengths of both approaches: raw path for rrh_movo, norm path for BTS.

Usage:
    # First, build both indices:
    #   CLAP_NORM_TARGET=0 python3 -m albart.pipeline.run_pipeline --skip-spotify --force
    #   cp data/faiss.index data/faiss_raw.index && cp data/faiss_ids.npy data/faiss_raw_ids.npy
    #   CLAP_NORM_TARGET=0.12 python3 -m albart.pipeline.run_pipeline --skip-spotify --force
    #   cp data/faiss.index data/faiss_norm.index && cp data/faiss_ids.npy data/faiss_norm_ids.npy
    #
    # Then run:
    #   python3 tools/dual_index_sweep.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable embed_audio's internal normalization — we handle it ourselves.
os.environ["CLAP_NORM_TARGET"] = "0"

import numpy as np
import librosa

from albart.pipeline.embedder import embed_audio, load_model, EMBEDDING_DIM
from albart.pipeline.database import get_connection
from albart.utils import DATA_DIR

SAMPLE_RATE = 48000
CHUNK_SAMPLES = 10 * SAMPLE_RATE
N_CHUNKS = 3
WINDOW_SAMPLES = N_CHUNKS * CHUNK_SAMPLES

GENRE_LABELS = {
    "bts_dynamite": "K-pop",
    "red_right_hand": "Post-punk / Gothic",
    "rrh_movo": "Post-punk / Gothic",
    "gimme_shelter": "Classic Rock",
    "alberto_balsalm": "Electronic / IDM",
    "mad_blunted_jazz": "Lo-fi / Hip-hop",
}

SEARCH_TERMS = {
    "bts_dynamite": "Dynamite",
    "red_right_hand": "Red Right Hand",
    "rrh_movo": "Red Right Hand",
    "gimme_shelter": "Gimme Shelter",
    "alberto_balsalm": "Alberto Balsalm",
    "mad_blunted_jazz": "Mad Blunted Jazz",
}


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def rms_normalize(audio: np.ndarray, target: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
    return (audio * (target / rms)).astype(np.float32)


def find_target_ids(conn, term: str) -> set[str]:
    rows = conn.execute(
        "SELECT track_id FROM tracks WHERE title LIKE ? OR artist LIKE ?",
        (f"%{term}%", f"%{term}%"),
    ).fetchall()
    return {r["track_id"] for r in rows}


def dual_sweep(
    audio: np.ndarray,
    model,
    processor,
    device: str,
    index_raw,
    ids_raw: np.ndarray,
    index_norm,
    ids_norm: np.ndarray,
    norm_target: float,
    target_ids: set[str],
    top_k: int,
    alpha: float,
) -> dict:
    """Sweep with dual-index querying."""
    step_samples = 1 * SAMPLE_RATE
    starts = list(range(0, len(audio) - WINDOW_SAMPLES + 1, step_samples))
    if not starts:
        return {"n": 0, "hit_counts": {}, "median_rank": None}

    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0, top_k: 0}
    target_ranks: list[int] = []
    ema_raw: np.ndarray | None = None
    ema_norm: np.ndarray | None = None

    for start in starts:
        window = audio[start : start + WINDOW_SAMPLES].astype(np.float32)

        # Embed 3 chunks — raw and normalized versions
        raw_embs = []
        norm_embs = []
        for i in range(N_CHUNKS):
            chunk = window[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
            # Raw embedding (CLAP_NORM_TARGET=0 already set)
            raw_embs.append(embed_audio(chunk, model, processor, device))
            # Normalized embedding — normalize audio, then embed
            chunk_norm = rms_normalize(chunk, norm_target)
            norm_embs.append(embed_audio(chunk_norm, model, processor, device))

        # Average chunks
        emb_raw = normalize(np.mean(raw_embs, axis=0))
        emb_norm = normalize(np.mean(norm_embs, axis=0))

        # EMA smoothing (separate for each path)
        if ema_raw is None or alpha >= 1.0:
            ema_raw = emb_raw
            ema_norm = emb_norm
        else:
            ema_raw = normalize(alpha * emb_raw + (1.0 - alpha) * ema_raw)
            ema_norm = normalize(alpha * emb_norm + (1.0 - alpha) * ema_norm)

        # Query both indices — use RANK fusion (not distance fusion,
        # because L2 distances from differently-normalized indices aren't
        # on the same scale).

        # Raw query → raw index: build rank map
        q_raw = ema_raw.reshape(1, -1).astype(np.float32)
        dists_r, idxs_r = index_raw.search(q_raw, top_k)
        raw_ranks: dict[str, int] = {}
        rank = 0
        for idx in idxs_r[0]:
            if idx < 0:
                continue
            tid = ids_raw[idx]
            rank += 1
            if tid not in raw_ranks:
                raw_ranks[tid] = rank

        # Norm query → norm index: build rank map
        q_norm = ema_norm.reshape(1, -1).astype(np.float32)
        dists_n, idxs_n = index_norm.search(q_norm, top_k)
        norm_ranks: dict[str, int] = {}
        rank = 0
        for idx in idxs_n[0]:
            if idx < 0:
                continue
            tid = ids_norm[idx]
            rank += 1
            if tid not in norm_ranks:
                norm_ranks[tid] = rank

        # Rank fusion: each track's score = best rank across both indices
        all_tids = set(raw_ranks) | set(norm_ranks)
        fused: dict[str, int] = {}
        for tid in all_tids:
            r = raw_ranks.get(tid, top_k + 1)
            n = norm_ranks.get(tid, top_k + 1)
            fused[tid] = min(r, n)

        sorted_tracks = sorted(fused.items(), key=lambda x: x[1])

        for rank, (tid, _) in enumerate(sorted_tracks, 1):
            if tid in target_ids:
                target_ranks.append(rank)
                for k in hit_counts:
                    if rank <= k:
                        hit_counts[k] += 1
                break

    n = len(starts)
    return {
        "n": n,
        "hit_counts": hit_counts,
        "target_ranks": target_ranks,
        "median_rank": sorted(target_ranks)[len(target_ranks) // 2] if target_ranks else None,
    }


def main():
    import faiss

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--norm-target", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    # Locate sample files
    samples_dir = Path(__file__).parent.parent / "data" / "samples"
    if args.samples:
        sample_files = [Path(s) for s in args.samples]
    else:
        sample_files = sorted(samples_dir.glob("*.wav"))

    if not sample_files:
        print("No sample files found.")
        return

    # Load indices
    raw_path = DATA_DIR / "faiss_raw.index"
    norm_path = DATA_DIR / "faiss_norm.index"
    if not raw_path.exists() or not norm_path.exists():
        print(f"Missing indices. Need both:\n  {raw_path}\n  {norm_path}")
        print("\nBuild them with:")
        print("  CLAP_NORM_TARGET=0 python3 -m albart.pipeline.run_pipeline --skip-spotify --force")
        print(f"  cp data/faiss.index {raw_path} && cp data/faiss_ids.npy data/faiss_raw_ids.npy")
        print(f"  CLAP_NORM_TARGET={args.norm_target} python3 -m albart.pipeline.run_pipeline --skip-spotify --force")
        print(f"  cp data/faiss.index {norm_path} && cp data/faiss_ids.npy data/faiss_norm_ids.npy")
        return

    print("Loading CLAP model...")
    model, processor, device = load_model()
    print(f"Model on {device}")

    print("Loading FAISS indices...")
    index_raw = faiss.read_index(str(raw_path))
    ids_raw = np.load(str(DATA_DIR / "faiss_raw_ids.npy"), allow_pickle=True)
    index_norm = faiss.read_index(str(norm_path))
    ids_norm = np.load(str(DATA_DIR / "faiss_norm_ids.npy"), allow_pickle=True)
    print(f"  Raw index: {index_raw.ntotal} vectors")
    print(f"  Norm index: {index_norm.ntotal} vectors")

    conn = get_connection()

    print(f"\nDual-index sweep: raw + norm_target={args.norm_target}  alpha={args.alpha}")
    print("=" * 90)

    fmt_header = f"{'Genre':<23s}{'Track':<36s}{'#1':>4s}{'Top3':>5s}{'Top5':>5s}{'Top10':>5s}{'Top20':>6s}{'Median':>8s}"
    print(fmt_header)
    print("-" * 90)

    results = []

    for fpath in sample_files:
        stem = fpath.stem
        genre = GENRE_LABELS.get(stem, "Unknown")
        search_term = SEARCH_TERMS.get(stem, stem.replace("_", " ").title())
        target_ids = find_target_ids(conn, search_term)

        audio, _ = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True, dtype="float32")
        label = f"{search_term} — {stem.replace('_', ' ').title()}"
        if len(label) > 35:
            label = label[:32] + "..."

        res = dual_sweep(
            audio, model, processor, device,
            index_raw, ids_raw,
            index_norm, ids_norm,
            args.norm_target, target_ids,
            args.top_k, args.alpha,
        )

        n = res["n"]
        hc = res["hit_counts"]
        median = res["median_rank"]
        median_str = f"#{median}" if median else "—"

        if n > 0:
            print(
                f"  {genre:<21s}{label:<35s}"
                f"{100*hc[1]/n:4.0f}%{100*hc[3]/n:5.0f}%{100*hc[5]/n:5.0f}%"
                f"{100*hc[10]/n:5.0f}%{100*hc[args.top_k]/n:6.0f}%"
                f"{median_str:>8s}"
            )
        else:
            print(f"  {genre:<21s}{label:<35s}  (too short)")

        results.append((genre, label, res))

    conn.close()

    # Summary
    print("\n" + "=" * 90)
    print(f"DUAL-INDEX SUMMARY  (raw + norm={args.norm_target}  alpha={args.alpha}  top_k={args.top_k})")
    print("=" * 90)
    for genre, label, res in results:
        n = res["n"]
        hc = res["hit_counts"]
        median = res["median_rank"]
        median_str = f"#{median}" if median else "—"
        if n > 0:
            print(
                f"  {genre:<23s}{label:<35s}"
                f"#1={100*hc[1]/n:4.0f}%  top5={100*hc[5]/n:4.0f}%  "
                f"top10={100*hc[10]/n:4.0f}%  median={median_str}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
