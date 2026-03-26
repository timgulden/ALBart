"""
tools/chunk_sweep.py

Simulate the runtime EMA embedding loop over a WAV file.

Takes a sliding 10s window at --step-second intervals (default: 1s, matching
the runtime), applies EMA with --alpha, queries the top-K tracks at each step,
and reports what the display would show vs. the target track rank.

Usage:
    python tools/chunk_sweep.py \
        --recording data/samples/rrh_movo.wav \
        --target "Red Right Hand"

    # Test a specific alpha:
    python tools/chunk_sweep.py \
        --recording data/samples/rrh_movo.wav \
        --target "Red Right Hand" \
        --alpha 0.5

    # No EMA (raw, as before):
    python tools/chunk_sweep.py \
        --recording data/samples/rrh_movo.wav \
        --target "Red Right Hand" \
        --alpha 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np

from albart.pipeline.database import DB_PATH, get_connection
from albart.pipeline.embedder import embed_audio, load_model
from albart.utils import DATA_DIR

SAMPLE_RATE = 48000
CHUNK_SAMPLES = 10 * SAMPLE_RATE   # 10s per chunk (matches runtime and pipeline)
N_CHUNKS = 3                        # 3 chunks per window (matches runtime and pipeline)
WINDOW_SAMPLES = N_CHUNKS * CHUNK_SAMPLES  # 30s total window
TOP_K = 20  # default; overridden by --top-k arg


def find_target_ids(conn, term: str) -> set[str]:
    rows = conn.execute(
        "SELECT track_id FROM tracks WHERE title LIKE ? OR artist LIKE ?",
        (f"%{term}%", f"%{term}%"),
    ).fetchall()
    return {r["track_id"] for r in rows}


def track_label(conn, track_id: str, width: int = 40) -> str:
    row = conn.execute(
        "SELECT title, artist FROM tracks WHERE track_id=?", (track_id,)
    ).fetchone()
    if row:
        s = f"{row['title'][:25]}  —  {row['artist'][:20]}"
        return s[:width]
    return track_id[:width]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True, help="WAV file to sweep")
    parser.add_argument("--target", required=True, help="Target track title/artist substring")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="EMA weight for current embedding (default: 0.5; 1.0 = no EMA)")
    parser.add_argument("--step", type=int, default=1,
                        help="Seconds between query windows (default: 1, matching runtime)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Number of candidates to retrieve per query (default: {TOP_K})")
    parser.add_argument("--index", type=str, default="faiss",
                        help="Index filename prefix (default: faiss → faiss.index / faiss_ids.npy)")
    parser.add_argument("--multi-probe", action="store_true",
                        help="Query each 10s chunk separately and take min distance per track "
                             "(instead of averaging embeddings first)")
    args = parser.parse_args()
    top_k = args.top_k

    conn = get_connection(DB_PATH)
    target_ids = find_target_ids(conn, args.target)
    if not target_ids:
        print(f"No tracks found matching '{args.target}'")
        sys.exit(1)
    print(f"Target ({len(target_ids)} versions):", ", ".join(
        track_label(conn, tid) for tid in target_ids
    ))

    print(f"\nLoading recording: {args.recording}")
    audio, _ = librosa.load(args.recording, sr=SAMPLE_RATE, mono=True, dtype="float32")
    duration = len(audio) / SAMPLE_RATE
    step_samples = args.step * SAMPLE_RATE
    # All valid start positions for a full 10s window
    starts = list(range(0, len(audio) - WINDOW_SAMPLES + 1, step_samples))
    print(f"  {duration:.1f}s recording  →  {len(starts)} windows "
          f"(10s window, {args.step}s step)")

    print("\nLoading CLAP model...")
    model, processor, device = load_model()
    print(f"Model on {device}")

    print("\nLoading FAISS index...")
    import faiss  # lazy — must come after CLAP model load to avoid BLAS conflict
    index = faiss.read_index(str(DATA_DIR / f"{args.index}.index"))
    track_ids_map = np.load(str(DATA_DIR / f"{args.index}_ids.npy"), allow_pickle=True)
    n_tracks = len(set(str(t) for t in track_ids_map))
    print(f"  {index.ntotal} vectors / {n_tracks} unique tracks")

    print(f"\nalpha={args.alpha}  step={args.step}s  top_k={top_k}\n")

    header = f"{'Time':>6}  {'RRH':>6}  {'d_ema':>7}  {'#1 track shown'}"
    print(header)
    print("-" * 75)

    ema: np.ndarray | None = None
    target_ranks: list[int] = []
    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0, top_k: 0}
    display_track_counts: dict[str, int] = {}

    for start in starts:
        t_sec = start / SAMPLE_RATE
        window = audio[start : start + WINDOW_SAMPLES].astype(np.float32)

        # Embed N_CHUNKS × 10s sub-windows
        chunk_embs = []
        for i in range(N_CHUNKS):
            sub = window[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
            chunk_embs.append(embed_audio(sub, model, processor, device))

        def _normalize(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            return (v / (n + 1e-8)).astype(np.float32)

        if args.multi_probe:
            # Query each chunk separately; per-track score = min distance across queries.
            # EMA over the renormalized average embedding (for display consistency).
            avg_emb = _normalize(np.mean(chunk_embs, axis=0))
            if ema is None or args.alpha >= 1.0:
                ema = avg_emb
            else:
                ema = _normalize(args.alpha * avg_emb + (1.0 - args.alpha) * ema)

            track_min: dict[str, float] = {}
            for ce in chunk_embs:
                q = ce.reshape(1, -1).astype(np.float32)
                dists, idxs = index.search(q, top_k)
                for dist, idx in zip(dists[0], idxs[0]):
                    if idx < 0:
                        continue
                    tid = str(track_ids_map[int(idx)])
                    if tid not in track_min or dist < track_min[tid]:
                        track_min[tid] = float(dist)
        else:
            # Average embeddings, renormalize, EMA, renormalize — mirrors runtime EmbeddingWorker
            emb = _normalize(np.mean(chunk_embs, axis=0))
            if ema is None or args.alpha >= 1.0:
                ema = emb
            else:
                ema = _normalize(args.alpha * emb + (1.0 - args.alpha) * ema)

            q = ema.reshape(1, -1).astype(np.float32)
            dists, idxs = index.search(q, top_k)

            track_min: dict[str, float] = {}
            for dist, idx in zip(dists[0], idxs[0]):
                if idx < 0:
                    continue
                tid = str(track_ids_map[int(idx)])
                if tid not in track_min or dist < track_min[tid]:
                    track_min[tid] = float(dist)

        sorted_tracks = sorted(track_min.items(), key=lambda x: x[1])

        # Find target rank within top-K
        target_rank = None
        target_dist = None
        for rank, (tid, dist) in enumerate(sorted_tracks, 1):
            if tid in target_ids:
                target_rank = rank
                target_dist = dist
                break

        # What the display actually shows (#1 track)
        top1_id, top1_dist = sorted_tracks[0]
        top1_label = track_label(conn, top1_id)
        display_track_counts[top1_label] = display_track_counts.get(top1_label, 0) + 1

        rank_str = f"#{target_rank}" if target_rank else f">{top_k}"
        dist_str = f"{target_dist:.4f}" if target_dist is not None else "  —  "
        is_hit = target_rank == 1
        hit_marker = " ◄" if is_hit else ""

        print(f"{t_sec:>5.0f}s  {rank_str:>6}  {dist_str:>7}  "
              f"{top1_label:<40}{hit_marker}")

        if target_rank:
            target_ranks.append(target_rank)
            for k in hit_counts:
                if target_rank <= k:
                    hit_counts[k] += 1

    n = len(starts)
    print("\n" + "=" * 75)
    print(f"SUMMARY  (alpha={args.alpha}, {n} windows)")
    print("=" * 75)
    print(f"  Rank #1:   {hit_counts[1]:>3}/{n}  ({100*hit_counts[1]/n:.0f}%)")
    print(f"  Top-3:     {hit_counts[3]:>3}/{n}  ({100*hit_counts[3]/n:.0f}%)")
    print(f"  Top-5:     {hit_counts[5]:>3}/{n}  ({100*hit_counts[5]/n:.0f}%)")
    print(f"  Top-10:    {hit_counts[10]:>3}/{n}  ({100*hit_counts[10]/n:.0f}%)")
    if top_k > 10:
        print(f"  Top-{top_k}:  {hit_counts[top_k]:>3}/{n}  ({100*hit_counts[top_k]/n:.0f}%)")
    if target_ranks:
        print(f"  Median rank: {sorted(target_ranks)[len(target_ranks)//2]}")
        print(f"  Mean rank:   {sum(target_ranks)/len(target_ranks):.1f}")

    print(f"\nTop tracks shown on display:")
    for label, count in sorted(display_track_counts.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / n
        marker = "  ← TARGET" if any(
            track_label(conn, tid) == label for tid in target_ids
        ) else ""
        print(f"  {count:>4}x ({pct:>4.1f}%)  {label}{marker}")

    conn.close()


if __name__ == "__main__":
    main()
