"""
tools/batch_sweep.py

Run the recognition benchmark across all genre samples and produce a
summary table showing rank distribution for each track.

Usage:
    python tools/batch_sweep.py
    python tools/batch_sweep.py --alpha 0.5 --top-k 20
    python tools/batch_sweep.py --samples data/samples/gimme_shelter.wav data/samples/bts_dynamite.wav

The script infers the target search term from the WAV filename by looking
up that name in GENRE_TRACKS (from record_genre_samples.py), falling back
to using the filename stem as the search term.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np

from albart.pipeline.embedder import embed_audio, load_model, EMBEDDING_DIM
from albart.pipeline.database import DB_PATH, get_connection
from albart.utils import DATA_DIR

SAMPLE_RATE = 48000
CHUNK_SAMPLES = 10 * SAMPLE_RATE
N_CHUNKS = 3
WINDOW_SAMPLES = N_CHUNKS * CHUNK_SAMPLES

# Map output_name → (genre, display_label, target_search_term)
# target_search_term is matched against track title and artist in the DB
SAMPLE_META = {
    "gimme_shelter":     ("Classic Rock",       "Gimme Shelter",            "Gimme Shelter"),
    "sentimental_mood":  ("Jazz",               "In A Sentimental Mood",    "Sentimental Mood"),
    "smokestack":        ("Blues",              "Smokestack Lightnin'",     "Smokestack"),
    "chopin_nocturne":   ("Classical Piano",    "Chopin Nocturne No. 1",    "Nocturne No. 1"),
    "bach_kreuzstab":    ("Classical Choral",   "Bach Kreuzstab",           "Kreuzstab"),
    "bts_dynamite":      ("K-pop",              "Dynamite — BTS",           "Dynamite"),
    "mad_blunted_jazz":  ("Lo-fi / Hip-hop",    "Mad Blunted Jazz",         "Mad Blunted Jazz"),
    "red_right_hand":    ("Post-punk / Gothic", "Red Right Hand",           "Red Right Hand"),
    "alberto_balsalm":   ("Electronic / IDM",   "Alberto Balsalm",         "Alberto Balsalm"),
    # Legacy samples kept for comparison
    "rrh_movo":          ("Post-punk / Gothic", "Red Right Hand (Movo)",    "Red Right Hand"),
    "roads_pdp":         ("Trip-hop",           "Roads (PDP)",              "Roads"),
    "roads_movo":        ("Trip-hop",           "Roads (Movo)",             "Roads"),
}


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / (n + 1e-8)).astype(np.float32)


def find_target_ids(conn, term: str) -> set[str]:
    rows = conn.execute(
        "SELECT track_id FROM tracks WHERE title LIKE ? OR artist LIKE ?",
        (f"%{term}%", f"%{term}%"),
    ).fetchall()
    return {r["track_id"] for r in rows}


def sweep(
    audio: np.ndarray,
    model,
    processor,
    device_str: str,
    index,
    track_ids: list[str],
    target_ids: set[str],
    top_k: int,
    alpha: float,
) -> dict:
    """Run chunk_sweep logic over the full audio. Returns stats dict."""
    step_samples = 1 * SAMPLE_RATE
    starts = list(range(0, len(audio) - WINDOW_SAMPLES + 1, step_samples))
    if not starts:
        return {"n": 0, "hit_counts": {}, "median_rank": None}

    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0, top_k: 0}
    target_ranks: list[int] = []
    ema: np.ndarray | None = None

    for start in starts:
        window = audio[start : start + WINDOW_SAMPLES].astype(np.float32)
        chunk_embs = []
        for i in range(N_CHUNKS):
            sub = window[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
            chunk_embs.append(embed_audio(sub, model, processor, device_str))

        emb = normalize(np.mean(chunk_embs, axis=0))
        if ema is None or alpha >= 1.0:
            ema = emb
        else:
            ema = normalize(alpha * emb + (1.0 - alpha) * ema)

        q = ema.reshape(1, -1).astype(np.float32)
        dists, idxs = index.search(q, top_k)

        track_min: dict[str, float] = {}
        for d_val, idx in zip(dists[0], idxs[0]):
            if idx < 0:
                continue
            tid = track_ids[idx]
            if tid not in track_min or d_val < track_min[tid]:
                track_min[tid] = float(d_val)

        sorted_tracks = sorted(track_min.items(), key=lambda x: x[1])

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
        "mean_rank": sum(target_ranks) / len(target_ranks) if target_ranks else None,
        "pct_found": 100 * len(target_ranks) / n if n else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch recognition sweep across genre samples")
    parser.add_argument("--samples", nargs="*", default=None,
                        help="WAV files to sweep (default: all .wav in data/samples/)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="EMA alpha (default: 0.5)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="FAISS top-K to retrieve per query (default: 20)")
    parser.add_argument("--index", type=str, default="faiss",
                        help="Index filename prefix (default: faiss)")
    args = parser.parse_args()

    samples_dir = Path(__file__).parent.parent / "data" / "samples"

    if args.samples:
        wav_files = [Path(p) for p in args.samples]
    else:
        wav_files = sorted(samples_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {samples_dir}")
        sys.exit(1)

    print(f"Loading CLAP model...")
    model, processor, device_str = load_model()
    print(f"Model on {device_str}")

    print(f"Loading FAISS index ({args.index})...")
    import faiss as faiss_lib
    index = faiss_lib.read_index(str(DATA_DIR / f"{args.index}.index"))
    track_ids = [str(t) for t in np.load(str(DATA_DIR / f"{args.index}_ids.npy"), allow_pickle=True)]
    print(f"  {index.ntotal} vectors\n")

    conn = get_connection(DB_PATH)

    # Header
    top_k = args.top_k
    print(f"{'Genre':<22} {'Track':<28} {'#1':>5} {'Top3':>5} {'Top5':>5} {'Top10':>5} {f'Top{top_k}':>6}  Median")
    print("-" * 90)

    results = []
    for wav in wav_files:
        stem = wav.stem
        meta = SAMPLE_META.get(stem)
        if meta:
            genre, label, search_term = meta
        else:
            genre, label, search_term = "Unknown", stem, stem.replace("_", " ")

        target_ids = find_target_ids(conn, search_term)
        if not target_ids:
            print(f"  WARNING: no DB tracks for search '{search_term}' — skipping {wav.name}")
            continue

        print(f"  {genre:<20} {label:<28} ...", end="", flush=True)
        audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True, dtype="float32")
        stats = sweep(audio, model, processor, device_str, index, track_ids, target_ids, top_k, args.alpha)

        n = stats["n"]
        hc = stats["hit_counts"]
        med = stats["median_rank"]

        def pct(k): return f"{100*hc.get(k,0)/n:.0f}%" if n else "—"

        med_str = f"#{med}" if med is not None else "—"
        print(f"\r  {genre:<20} {label:<28} "
              f"{pct(1):>5} {pct(3):>5} {pct(5):>5} {pct(10):>5} {pct(top_k):>6}  {med_str}")

        results.append({
            "genre": genre,
            "label": label,
            "wav": wav.name,
            "stats": stats,
        })

    conn.close()

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 90)
        print(f"SUMMARY  (alpha={args.alpha}  top_k={top_k}  index={args.index})")
        print("=" * 90)
        for r in results:
            n = r["stats"]["n"]
            hc = r["stats"]["hit_counts"]
            med = r["stats"]["median_rank"]
            def pct(k): return f"{100*hc.get(k,0)/n:.0f}%" if n else "—"
            med_str = f"#{med}" if med is not None else "—"
            print(f"  {r['genre']:<22} {r['label']:<28} "
                  f"#1={pct(1):>4}  top5={pct(5):>4}  top10={pct(10):>4}  median={med_str}")


if __name__ == "__main__":
    main()
