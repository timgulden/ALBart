"""
tools/preprocess_sweep.py

Preprocessing strategy sweep for laion/clap-htsat-unfused.

For each strategy, builds a FlatL2 FAISS index from all preview files
(up to 3 × 10s chunks per preview, stored as individual vectors) and
evaluates recognition accuracy on WAV files in data/samples/ using
non-overlapping 10s windows.

Both index (preview) and query (recording) audio are preprocessed with
the same strategy — ensuring the embedding space stays consistent.

Note: embed_audio() in production always applies preprocess_audio()
internally for clap-htsat-unfused (hidden_size=768).  This sweep
bypasses that via a direct processor call so strategies can be tested
independently.  "hard_limit" corresponds to the production code path.

Usage:
    # Sequential
    python tools/preprocess_sweep.py

    # Parallel — all 8 strategies simultaneously (~8 min on 14-core Mac)
    python tools/preprocess_sweep.py --parallel

    # Subset of strategies
    python tools/preprocess_sweep.py --strategies none rms0.12 hard_limit

    # Quick sanity check with 300 tracks
    python tools/preprocess_sweep.py --max-tracks 300 --parallel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np
import scipy.signal
import torch
from tqdm import tqdm

from albart.pipeline.database import DB_PATH, get_connection
from albart.pipeline.embedder import load_model, EMBEDDING_DIM, SAMPLE_RATE
from albart.utils import DATA_DIR, compress_and_normalize, preprocess_audio

CHUNK_SAMPLES = 10 * SAMPLE_RATE   # 10s per chunk
N_CHUNKS      = 3                   # max chunks taken per preview

# PyTorch threads per worker in parallel mode.
# In practice each worker uses ~1.24 cores (Accelerate handles its own threading),
# so 1 CPU reserved per worker lets all 8 run simultaneously on a 14-core machine.
THREADS_PER_WORKER = 1


# ── Preprocessing strategies ──────────────────────────────────────────────────
# Applied identically to preview (index) and recording (query) audio.

def _rms(a: np.ndarray, target: float = 0.1) -> np.ndarray:
    rms = float(np.sqrt(np.mean(a ** 2))) + 1e-8
    return (a * (target / rms)).astype(np.float32)


def _lp(a: np.ndarray, cutoff: int) -> np.ndarray:
    sos = scipy.signal.butter(8, cutoff, btype="low", fs=SAMPLE_RATE, output="sos")
    return scipy.signal.sosfilt(sos, a).astype(np.float32)


STRATEGIES: dict[str, callable] = {
    "none":           lambda a: a.astype(np.float32),
    "rms0.12":        lambda a: _rms(a, 0.12),
    "compress":       compress_and_normalize,
    "rms+lp4k":       lambda a: _lp(_rms(a), 4000),
    "rms+lp8k":       lambda a: _lp(_rms(a), 8000),
    "compress+lp4k":  lambda a: _lp(compress_and_normalize(a), 4000),
    "compress+lp8k":  lambda a: _lp(compress_and_normalize(a), 8000),
    "hard_limit":     lambda a: preprocess_audio(a, sr=SAMPLE_RATE),
}

# Maps sample filename stem → (title_substring, artist_substring)
# Both fields must match (AND logic); None = skip that field.
SAMPLE_TARGETS: dict[str, tuple[str | None, str | None]] = {
    "alberto_balsalm":     ("Alberto Balsalm", None),
    "bach_kreuzstab":      ("Kreuzstab",        None),
    "bts_dynamite":        ("Dynamite",          "BTS"),
    "chopin_nocturne":     ("Nocturne",          "Chopin"),
    "gimme_shelter":       ("Gimme Shelter",     "Rolling Stones"),
    "mad_blunted_jazz":    ("Mad Blunted Jazz",  None),
    "red_right_hand":      ("Red Right Hand",    "Nick Cave"),
    "red_right_hand_room": ("Red Right Hand",    "Nick Cave"),
    "roads_movo":          ("Roads",             "Portishead"),
    "roads_pdp":           ("Roads",             "Portishead"),
    "rrh_live_test":       ("Red Right Hand",    "Nick Cave"),
    "rrh_movo":            ("Red Right Hand",    "Nick Cave"),
    "sentimental_mood":    ("Sentimental Mood",  None),
    "smokestack":          ("Smokestack",        None),
}

# Default evaluation set: samples recorded 2026-03-22 with AGC-controlled levels.
# Earlier recordings used manual level setting and are excluded by default.
# Use --all-samples to include everything.
AGC_SAMPLES = {
    "alberto_balsalm", "bach_kreuzstab", "bts_dynamite", "chopin_nocturne",
    "gimme_shelter", "mad_blunted_jazz", "red_right_hand", "sentimental_mood",
    "smokestack",
}


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed(audio: np.ndarray, model, processor, device: str) -> np.ndarray:
    """Embed pre-processed audio without any internal preprocessing."""
    inputs = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) if v.dtype != torch.bool else v
              for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    vec = emb.cpu().numpy().squeeze().astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / (norm + 1e-8)


# ── Database helpers ──────────────────────────────────────────────────────────

def find_target_ids(conn, title_term: str | None, artist_term: str | None) -> set[str]:
    parts, params = [], []
    if title_term:
        parts.append("LOWER(title) LIKE LOWER(?)")
        params.append(f"%{title_term}%")
    if artist_term:
        parts.append("LOWER(artist) LIKE LOWER(?)")
        params.append(f"%{artist_term}%")
    if not parts:
        return set()
    sql = "SELECT track_id FROM tracks WHERE " + " AND ".join(parts)
    return {r["track_id"] for r in conn.execute(sql, params).fetchall()}


def track_label(conn, track_id: str) -> str:
    row = conn.execute(
        "SELECT title, artist FROM tracks WHERE track_id=?", (track_id,)
    ).fetchone()
    return f"{row['title'][:30]} — {row['artist'][:20]}" if row else track_id


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_sample(
    audio: np.ndarray,
    target_ids: set[str],
    index,
    ids_list: list[str],
    strategy_fn: callable,
    model, processor, device: str,
) -> dict:
    """Slide 10s windows through recording; return per-window target ranks."""
    n_windows = len(audio) // CHUNK_SAMPLES
    ranks = []

    for i in range(n_windows):
        chunk = audio[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES].astype(np.float32)
        emb = _embed(strategy_fn(chunk), model, processor, device)
        dists, idxs = index.search(emb.reshape(1, -1), index.ntotal)

        track_min: dict[str, float] = {}
        for d, j in zip(dists[0], idxs[0]):
            if j < 0:
                continue
            tid = ids_list[j]
            if tid not in track_min or d < track_min[tid]:
                track_min[tid] = float(d)

        sorted_tracks = sorted(track_min.items(), key=lambda x: x[1])
        for rank, (tid, _) in enumerate(sorted_tracks, 1):
            if tid in target_ids:
                ranks.append(rank)
                break
        else:
            ranks.append(len(sorted_tracks) + 1)

    return {"n": n_windows, "ranks": ranks}


# ── Ray worker ────────────────────────────────────────────────────────────────

def _ray_worker(
    strategy_name: str,
    audio_store: dict,         # {track_id: int16 array} — pre-loaded, shared via object store
    track_ids: list,           # ordered list of track_ids to embed
    sample_tuples: list,       # [(stem, [target_id, ...], audio_array), ...]
    n_threads: int,
) -> dict:
    """
    Run one preprocessing strategy end-to-end in an isolated process:
    build index from pre-loaded audio, evaluate all samples, return results dict.

    Audio is received as int16 arrays from Ray's shared object store (zero-copy),
    converted to float32 per track — no disk I/O during the compute phase.

    Self-contained: all imports and strategy definitions are local so this
    function serializes cleanly across Ray worker processes.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import numpy as np
    import scipy.signal
    import torch
    import faiss
    from albart.pipeline.embedder import load_model, EMBEDDING_DIM, SAMPLE_RATE
    from albart.utils import compress_and_normalize, preprocess_audio

    torch.set_num_threads(n_threads)
    chunk_samples = 10 * SAMPLE_RATE
    n_chunks_max = 3

    # Reconstruct strategy functions locally (lambdas aren't picklable)
    def _rms_local(a, target=0.1):
        rms = float(np.sqrt(np.mean(a ** 2))) + 1e-8
        return (a * (target / rms)).astype(np.float32)

    def _lp_local(a, cutoff):
        sos = scipy.signal.butter(8, cutoff, btype="low", fs=SAMPLE_RATE, output="sos")
        return scipy.signal.sosfilt(sos, a).astype(np.float32)

    _strats = {
        "none":           lambda a: a.astype(np.float32),
        "rms0.12":        lambda a: _rms_local(a, 0.12),
        "compress":       compress_and_normalize,
        "rms+lp4k":       lambda a: _lp_local(_rms_local(a), 4000),
        "rms+lp8k":       lambda a: _lp_local(_rms_local(a), 8000),
        "compress+lp4k":  lambda a: _lp_local(compress_and_normalize(a), 4000),
        "compress+lp8k":  lambda a: _lp_local(compress_and_normalize(a), 8000),
        "hard_limit":     lambda a: preprocess_audio(a, sr=SAMPLE_RATE),
    }
    fn = _strats[strategy_name]

    def _embed_local(audio):
        inputs = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(device) if v.dtype != torch.bool else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            emb = model.get_audio_features(**inputs)
        vec = emb.cpu().numpy().squeeze().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / (norm + 1e-8)

    print(f"[{strategy_name}] Loading CLAP model...", flush=True)
    model, processor, device = load_model(allow_mps=True)

    # Build index from pre-loaded audio (no disk I/O)
    print(f"[{strategy_name}] Building index from {len(track_ids)} previews...", flush=True)
    embs, ids = [], []
    for i, track_id in enumerate(track_ids):
        audio_i16 = audio_store.get(track_id)
        if audio_i16 is None:
            continue
        audio = audio_i16.astype(np.float32) / 32767.0
        n_ch = min(n_chunks_max, len(audio) // chunk_samples)
        for c in range(n_ch):
            chunk = audio[c * chunk_samples:(c + 1) * chunk_samples].astype(np.float32)
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            embs.append(_embed_local(fn(chunk)))
            ids.append(track_id)
        if (i + 1) % 500 == 0:
            print(f"[{strategy_name}] {i + 1}/{len(track_ids)} previews", flush=True)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(np.array(embs, dtype=np.float32))
    print(f"[{strategy_name}] Index built: {index.ntotal} vectors", flush=True)

    # Evaluate samples
    results = {}
    for stem, target_ids_list, rec_audio in sample_tuples:
        target_ids = set(target_ids_list)
        n_windows = len(rec_audio) // chunk_samples
        ranks = []
        for i in range(n_windows):
            chunk = rec_audio[i * chunk_samples:(i + 1) * chunk_samples].astype(np.float32)
            emb = _embed_local(fn(chunk))
            dists, idxs = index.search(emb.reshape(1, -1), index.ntotal)
            track_min: dict[str, float] = {}
            for d, j in zip(dists[0], idxs[0]):
                if j >= 0:
                    tid = ids[j]
                    if tid not in track_min or d < track_min[tid]:
                        track_min[tid] = float(d)
            sorted_tracks = sorted(track_min.items(), key=lambda x: x[1])
            for rank, (tid, _) in enumerate(sorted_tracks, 1):
                if tid in target_ids:
                    ranks.append(rank)
                    break
            else:
                ranks.append(len(sorted_tracks) + 1)
        results[stem] = {"n": n_windows, "ranks": ranks}
        top1 = sum(1 for x in ranks if x == 1)
        print(f"[{strategy_name}] {stem}: top1={top1}/{n_windows}", flush=True)

    print(f"[{strategy_name}] Done.", flush=True)
    return results


# ── Table printing ────────────────────────────────────────────────────────────

def _print_tables(results: dict, sample_stems: list, strat_names: list) -> None:
    col_w = max(max(len(n) for n in strat_names), 7) + 2

    def _pct(r: dict, k: int) -> str:
        if not r.get("n"):
            return "—"
        hits = sum(1 for x in r["ranks"] if x <= k)
        return f"{100 * hits / r['n']:.0f}%"

    def _med(r: dict) -> str:
        if not r.get("ranks"):
            return "—"
        return str(sorted(r["ranks"])[len(r["ranks"]) // 2])

    def _print_table(title: str, cell_fn: callable, agg_k: int | None = None) -> None:
        width = 24 + col_w * len(strat_names)
        print("\n" + "=" * width)
        print(title)
        print("=" * width)
        header = f"{'Sample':<24}" + "".join(f"{n:>{col_w}}" for n in strat_names)
        print(header)
        print("-" * len(header))

        agg: dict[str, list] = {n: [] for n in strat_names}
        for stem in sample_stems:
            row = f"{stem:<24}"
            for name in strat_names:
                r = results[name].get(stem, {})
                row += f"{cell_fn(r):>{col_w}}"
                if r.get("n"):
                    agg[name].append(r)
            print(row)

        if agg_k is not None:
            print("-" * len(header))
            agg_row = f"{'AGGREGATE':<24}"
            for name in strat_names:
                rs = agg[name]
                if rs:
                    total_n = sum(r["n"] for r in rs)
                    total_hits = sum(
                        sum(1 for x in r["ranks"] if x <= agg_k) for r in rs
                    )
                    agg_row += f"{100*total_hits/total_n:.0f}%".rjust(col_w)
                else:
                    agg_row += "—".rjust(col_w)
            print(agg_row)

    _print_table("Top-1 hit rate (%)",  lambda r: _pct(r, 1),  agg_k=1)
    _print_table("Top-5 hit rate (%)",  lambda r: _pct(r, 5),  agg_k=5)
    _print_table("Top-10 hit rate (%)", lambda r: _pct(r, 10), agg_k=10)
    _print_table("Median rank",         _med)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing strategy sweep")
    parser.add_argument(
        "--strategies", nargs="+", choices=list(STRATEGIES), default=list(STRATEGIES),
        metavar="S", help="Which strategies to test (default: all 8)",
    )
    parser.add_argument(
        "--max-tracks", type=int, default=None,
        help="Limit index to first N tracks (for quick sanity tests)",
    )
    parser.add_argument(
        "--samples-dir", type=Path, default=DATA_DIR / "samples",
        help="Directory containing sample WAV files (default: data/samples/)",
    )
    parser.add_argument(
        "--all-samples", action="store_true",
        help="Include all samples; default uses only Mar-22 AGC-recorded set",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run all strategies simultaneously via Ray (requires: pip install ray)",
    )
    args = parser.parse_args()

    strat_names = [k for k in STRATEGIES if k in args.strategies]
    strategies  = {k: STRATEGIES[k] for k in strat_names}

    print("=" * 72)
    print(f"Preprocessing sweep: {len(strategies)} strategies"
          + (" [PARALLEL]" if args.parallel else " [sequential]"))
    print(f"  {', '.join(strat_names)}")
    print("=" * 72)

    # Fetch tracks from DB
    conn = get_connection(DB_PATH)
    all_tracks = conn.execute(
        "SELECT track_id, preview_path FROM tracks "
        "WHERE embedding_status='ok' AND preview_path IS NOT NULL "
        "ORDER BY track_id"
    ).fetchall()
    if args.max_tracks:
        all_tracks = all_tracks[:args.max_tracks]

    track_tuples = [(r["track_id"], r["preview_path"]) for r in all_tracks]
    n_vectors = len(track_tuples) * N_CHUNKS
    print(f"\nIndex: {len(track_tuples)} tracks × {N_CHUNKS} chunks = {n_vectors} vectors")

    # Load sample WAVs
    samples_dir = args.samples_dir
    sample_wavs = sorted(samples_dir.glob("*.wav"))
    allowed = None if args.all_samples else AGC_SAMPLES
    if allowed is not None:
        sample_wavs = [p for p in sample_wavs if p.stem in allowed]
        print(f"\nLoading {len(sample_wavs)} AGC-recorded sample WAVs "
              f"(use --all-samples to include earlier recordings)")
    else:
        print(f"\nLoading {len(sample_wavs)} sample WAVs from {samples_dir}/")

    # [(stem, target_ids_set, audio_array), ...]
    samples: list[tuple[str, set[str], np.ndarray]] = []
    for wav_path in sample_wavs:
        stem = wav_path.stem
        if stem not in SAMPLE_TARGETS:
            print(f"  {stem}: no target mapping — skipping")
            continue
        title_term, artist_term = SAMPLE_TARGETS[stem]
        target_ids = find_target_ids(conn, title_term, artist_term)
        if not target_ids:
            print(f"  {stem}: no DB matches — skipping")
            continue
        rec_audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True, dtype="float32")
        n_win = len(rec_audio) // CHUNK_SAMPLES
        rms = float(np.sqrt(np.mean(rec_audio ** 2)))
        label = track_label(conn, next(iter(target_ids)))
        print(f"  {stem}: {n_win} windows  RMS={rms:.3f}  target: {label}")
        samples.append((stem, target_ids, rec_audio))

    conn.close()

    if not samples:
        print("\nNo samples with known targets — exiting.")
        sys.exit(1)

    sample_stems = [s[0] for s in samples]

    # ── Parallel path ──────────────────────────────────────────────────────────
    if args.parallel:
        import ray

        # Pre-load all preview audio as int16 in the main process.
        # Workers receive a zero-copy view from Ray's shared object store —
        # no disk I/O during the compute phase, eliminating SSD contention.
        print(f"\nPre-loading {len(track_tuples)} previews into shared memory (int16)...")
        audio_store: dict[str, np.ndarray] = {}
        n_load_errors = 0
        for track_id, preview_path in tqdm(track_tuples, desc="Loading audio", unit="track"):
            try:
                audio, _ = librosa.load(
                    str(DATA_DIR / preview_path), sr=SAMPLE_RATE, mono=True, dtype="float32"
                )
                audio_store[track_id] = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            except Exception:
                n_load_errors += 1
        total_mb = sum(a.nbytes for a in audio_store.values()) / 1024 ** 2
        print(f"  {len(audio_store)} tracks loaded  ({total_mb:.0f} MB)"
              + (f"  ({n_load_errors} errors)" if n_load_errors else ""))

        # Ordered list of track_ids that loaded successfully
        loaded_track_ids = [t for t, _ in track_tuples if t in audio_store]

        # Serialize samples: sets → lists for Ray serialization
        sample_tuples = [(stem, list(tids), audio)
                         for stem, tids, audio in samples]

        ray.init()
        try:
            # Put shared data in the object store once; all workers get
            # a zero-copy memory-mapped reference.
            audio_ref      = ray.put(audio_store)
            track_ids_ref  = ray.put(loaded_track_ids)
            sample_ref     = ray.put(sample_tuples)

            # Wrap _ray_worker as a Ray remote function here to avoid
            # module-level decoration interfering with sequential mode
            remote_fn = ray.remote(num_cpus=THREADS_PER_WORKER)(_ray_worker)

            print(f"\nLaunching {len(strat_names)} Ray workers "
                  f"({THREADS_PER_WORKER} CPU each, no disk I/O)...")
            futures = {
                name: remote_fn.remote(
                    name, audio_ref, track_ids_ref, sample_ref, THREADS_PER_WORKER
                )
                for name in strat_names
            }

            print("Waiting for workers to complete...")
            results: dict[str, dict] = {}
            done, remaining = set(), set(futures.keys())
            while remaining:
                ready_refs, _ = ray.wait(
                    [futures[n] for n in remaining], num_returns=1, timeout=30
                )
                for ref in ready_refs:
                    # Find which strategy this ref belongs to
                    for name in remaining:
                        if futures[name] == ref:
                            results[name] = ray.get(ref)
                            print(f"  ✓ {name} complete")
                            done.add(name)
                            break
                remaining -= done
        finally:
            ray.shutdown()

    # ── Sequential path ────────────────────────────────────────────────────────
    else:
        n_total_embs = n_vectors * len(strategies)
        est_min = n_total_embs // 38 // 60 + 1
        print(f"  {n_total_embs:,} total embeddings, ~{est_min} min estimated")

        print("\nLoading CLAP model...")
        model, processor, device = load_model(allow_mps=True)
        print(f"  Loaded on {device}")

        strat_embs: dict[str, list] = {n: [] for n in strat_names}
        strat_ids:  dict[str, list] = {n: [] for n in strat_names}

        print(f"\nBuilding indices (single pass through {len(all_tracks)} previews)...")
        n_skipped = 0
        for row in tqdm(all_tracks, desc="Previews", unit="track"):
            track_id = row["track_id"]
            preview_path = DATA_DIR / row["preview_path"]
            try:
                audio, _ = librosa.load(
                    str(preview_path), sr=SAMPLE_RATE, mono=True, dtype="float32"
                )
            except Exception:
                n_skipped += 1
                continue
            n_chunks = min(N_CHUNKS, len(audio) // CHUNK_SAMPLES)
            for i in range(n_chunks):
                chunk = audio[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES].astype(np.float32)
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                for name, fn in strategies.items():
                    emb = _embed(fn(chunk), model, processor, device)
                    strat_embs[name].append(emb)
                    strat_ids[name].append(track_id)
        if n_skipped:
            print(f"  ({n_skipped} tracks skipped)")

        import faiss
        print("\nBuilding in-memory FAISS indices...")
        indices: dict[str, tuple] = {}
        for name in strat_names:
            embs = np.array(strat_embs[name], dtype=np.float32)
            idx = faiss.IndexFlatL2(EMBEDDING_DIM)
            idx.add(embs)
            indices[name] = (idx, strat_ids[name])
            print(f"  {name:<16}: {idx.ntotal} vectors")

        print(f"\nEvaluating {len(samples)} samples × {len(strategies)} strategies...")
        results: dict[str, dict] = {n: {} for n in strat_names}
        for stem, target_ids, rec_audio in samples:
            n_win = len(rec_audio) // CHUNK_SAMPLES
            print(f"\n  {stem}  ({n_win} windows)")
            for name, fn in strategies.items():
                idx, ids_list = indices[name]
                r = evaluate_sample(rec_audio, target_ids, idx, ids_list, fn,
                                    model, processor, device)
                results[name][stem] = r
                top1  = sum(1 for x in r["ranks"] if x == 1)
                top5  = sum(1 for x in r["ranks"] if x <= 5)
                top10 = sum(1 for x in r["ranks"] if x <= 10)
                med   = sorted(r["ranks"])[len(r["ranks"]) // 2] if r["ranks"] else 0
                print(f"    {name:<16}  top1={top1:>2}/{n_win}  "
                      f"top5={top5:>2}/{n_win}  top10={top10:>2}/{n_win}  med={med}")

    # ── Summary tables (both paths) ────────────────────────────────────────────
    _print_tables(results, sample_stems, strat_names)
    print("\nDone.")


if __name__ == "__main__":
    main()
