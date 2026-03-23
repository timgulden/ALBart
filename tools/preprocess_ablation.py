"""
tools/preprocess_ablation.py

Test multiple audio preprocessing variants to find the optimal configuration
for the Movo mic. For each variant, embeds the query recording AND the
reference previews with the SAME preprocessing, then measures:

  d_target  — mean min-distance from query chunks to target track chunks
  d_compare — mean min-distance from query chunks to comparison track chunks
  ratio     — d_target / d_compare  (lower = better matching + discrimination)

Usage:
    python tools/preprocess_ablation.py \\
        --recording data/samples/rrh_movo.wav \\
        --target "Red Right Hand" \\
        --compare "Willow Weep" "Nina Simone" "Kruder" "Sunshine" "High Noon"
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

from albart.pipeline.database import DB_PATH, get_connection
from albart.utils import DATA_DIR

SAMPLE_RATE = 48000
CHUNK_SAMPLES = 10 * SAMPLE_RATE
MODEL_ID = "laion/clap-htsat-unfused"


# ── Preprocessing building blocks ────────────────────────────────────────────

def rms_norm(a: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(a ** 2))) + 1e-8
    return (a * (0.1 / rms)).astype(np.float32)


def hard_limit(a: np.ndarray, level: float) -> np.ndarray:
    return np.clip(a, -level, level).astype(np.float32)


def lp(a: np.ndarray, cutoff: int) -> np.ndarray:
    sos = scipy.signal.butter(8, cutoff, btype="low", fs=SAMPLE_RATE, output="sos")
    return scipy.signal.sosfilt(sos, a).astype(np.float32)


VARIANTS: dict[str, callable] = {
    "none":              lambda a: a.astype(np.float32),
    "rms":               lambda a: rms_norm(a),
    "rms+lp4k":          lambda a: lp(rms_norm(a), 4000),
    "rms+lp8k":          lambda a: lp(rms_norm(a), 8000),
    "rms+lim.02":        lambda a: hard_limit(rms_norm(a), 0.02),
    "rms+lim.05":        lambda a: hard_limit(rms_norm(a), 0.05),
    "rms+lim.02+lp4k":  lambda a: lp(hard_limit(rms_norm(a), 0.02), 4000),  # CURRENT
    "rms+lim.02+lp8k":  lambda a: lp(hard_limit(rms_norm(a), 0.02), 8000),
    "rms+lim.05+lp4k":  lambda a: lp(hard_limit(rms_norm(a), 0.05), 4000),
    "rms+lim.05+lp8k":  lambda a: lp(hard_limit(rms_norm(a), 0.05), 8000),
    "rms+lp4k+lim.02":  lambda a: hard_limit(lp(rms_norm(a), 4000), 0.02),  # LP before limit
}

CURRENT_VARIANT = "rms+lim.02+lp4k"


# ── CLAP helpers ─────────────────────────────────────────────────────────────

def load_clap():
    from transformers import ClapModel, ClapProcessor
    from albart.utils import get_device
    device = get_device()
    print(f"Loading CLAP model on {device}...")
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return model, processor, device


def embed_preprocessed(audio: np.ndarray, model, processor, device) -> np.ndarray:
    """Embed audio that has already been preprocessed — no internal preprocessing."""
    inputs = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    return emb.cpu().numpy().squeeze().astype(np.float32)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def split_chunks(audio: np.ndarray, max_chunks: int | None = None) -> list[np.ndarray]:
    n = len(audio) // CHUNK_SAMPLES
    if max_chunks is not None:
        n = min(n, max_chunks)
    chunks = []
    for i in range(n):
        start = i * CHUNK_SAMPLES
        chunk = audio[start:start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunks.append(chunk.astype(np.float32))
    return chunks


def min_l2(query_embs: list[np.ndarray], ref_embs: list[np.ndarray]) -> float:
    """Minimum squared L2 distance between any query and any reference embedding."""
    best = float("inf")
    for q in query_embs:
        for r in ref_embs:
            d = float(np.sum((q - r) ** 2))
            if d < best:
                best = d
    return best


# ── Database helpers ──────────────────────────────────────────────────────────

def find_tracks(terms: list[str]) -> dict[str, str]:
    """Return {track_id: 'title — artist'} for tracks matching any term."""
    conn = get_connection(DB_PATH)
    results = {}
    for term in terms:
        rows = conn.execute(
            "SELECT track_id, title, artist FROM tracks "
            "WHERE title LIKE ? OR artist LIKE ? LIMIT 5",
            (f"%{term}%", f"%{term}%"),
        ).fetchall()
        for row in rows:
            results[row["track_id"]] = f"{row['title']} — {row['artist']}"
    conn.close()
    return results


def load_preview(track_id: str) -> np.ndarray | None:
    conn = get_connection(DB_PATH)
    row = conn.execute(
        "SELECT preview_path FROM tracks WHERE track_id=?", (track_id,)
    ).fetchone()
    conn.close()
    if not row or not row["preview_path"]:
        return None
    path = DATA_DIR / row["preview_path"]
    if not path.exists():
        return None
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, dtype="float32")
    return audio


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True, help="Query WAV file (Movo recording)")
    parser.add_argument("--target", required=True, help="Target track title/artist substring")
    parser.add_argument("--compare", nargs="+", required=True,
                        help="Comparison track substrings (false-positive candidates)")
    parser.add_argument("--chunks", type=int, default=18,
                        help="Max query chunks to use from recording (default: 18)")
    args = parser.parse_args()

    model, processor, device = load_clap()

    # Load query recording
    print(f"\nLoading recording: {args.recording}")
    rec, _ = librosa.load(args.recording, sr=SAMPLE_RATE, mono=True, dtype="float32")
    query_chunks_raw = split_chunks(rec, max_chunks=args.chunks)
    print(f"  {len(query_chunks_raw)} query chunks  RMS={float(np.sqrt(np.mean(rec**2))):.4f}")

    # Find target and comparison tracks
    target_tracks = find_tracks([args.target])
    compare_tracks = find_tracks(args.compare)
    for tid in list(compare_tracks.keys()):
        if tid in target_tracks:
            del compare_tracks[tid]

    print(f"\nTarget tracks ({len(target_tracks)}):")
    for tid, label in target_tracks.items():
        print(f"  {label}")

    print(f"\nComparison tracks ({len(compare_tracks)}):")
    for tid, label in compare_tracks.items():
        print(f"  {label}")

    # Load preview audio
    track_audio: dict[str, np.ndarray] = {}
    all_tracks = {**target_tracks, **compare_tracks}
    print()
    for tid, label in all_tracks.items():
        audio = load_preview(tid)
        if audio is not None:
            track_audio[tid] = audio
        else:
            print(f"  WARNING: no preview for {label}")

    target_ids  = [tid for tid in target_tracks  if tid in track_audio]
    compare_ids = [tid for tid in compare_tracks if tid in track_audio]

    if not target_ids or not compare_ids:
        print("Need at least one target and one comparison track with previews.")
        sys.exit(1)

    # ── Run ablation ─────────────────────────────────────────────────────────
    print(f"\nRunning {len(VARIANTS)} variants × "
          f"({len(query_chunks_raw)} query + "
          f"{sum(min(3, len(split_chunks(track_audio[t]))) for t in track_audio)} ref) chunks...\n")

    print(f"{'Variant':<22} {'d_target':>8} {'d_compare':>10} {'ratio':>7}")
    print("-" * 55)

    results = []

    for name, preprocess in VARIANTS.items():
        # Preprocess and embed query chunks
        q_embs = [embed_preprocessed(preprocess(c), model, processor, device)
                  for c in query_chunks_raw]

        # Preprocess and embed reference chunks (same variant)
        ref_embs: dict[str, list[np.ndarray]] = {}
        for tid, audio in track_audio.items():
            chunks = split_chunks(audio, max_chunks=3)
            ref_embs[tid] = [embed_preprocessed(preprocess(c), model, processor, device)
                             for c in chunks]

        # Compute distances
        target_dists  = [min_l2(q_embs, ref_embs[tid]) for tid in target_ids]
        compare_dists = [min_l2(q_embs, ref_embs[tid]) for tid in compare_ids]

        d_t = float(np.mean(target_dists))
        d_c = float(np.mean(compare_dists))
        ratio = d_t / d_c if d_c > 0 else float("inf")
        marker = " ◄ CURRENT" if name == CURRENT_VARIANT else ""

        print(f"{name:<22} {d_t:>8.4f} {d_c:>10.4f} {ratio:>7.4f}{marker}")
        results.append((ratio, name, d_t, d_c))

    print("\n" + "=" * 55)
    print("RANKED BY RATIO (best = lowest d_target/d_compare):")
    print("=" * 55)
    for i, (ratio, name, d_t, d_c) in enumerate(sorted(results), 1):
        marker = " ◄ CURRENT" if name == CURRENT_VARIANT else ""
        print(f"  {i:>2}. {ratio:.4f}  {name:<22}  d_target={d_t:.4f}  d_compare={d_c:.4f}{marker}")


if __name__ == "__main__":
    main()
