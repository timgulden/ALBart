"""Test transit strategies between anchor tracks in 512D space.

Simulates 10-step transits between all consecutive anchor pairs
and reports distance reduction at each step.

Usage:
    python -m tools.test_transit "Ray Charles — I've Got a Woman" "Ray Charles — Hit the Road Jack"
    python -m tools.test_transit --all-pairs "Ray Charles — I've Got a Woman" "Ray Charles — What'd I Say, Pt. 1 & 2" "Ray Charles — Georgia on My Mind"
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from albart.orbit import _find_track, _normalize_search
from albart.pipeline.database import DB_PATH, get_connection
from albart.pipeline.embedder import FAISS_NORM_INDEX_PATH, FAISS_NORM_IDS_PATH


def load_data():
    import faiss
    index = faiss.read_index(str(FAISS_NORM_INDEX_PATH))
    ids_arr = np.load(str(FAISS_NORM_IDS_PATH), allow_pickle=True)
    id_list = [str(t) for t in ids_arr]
    id_to_idx = {tid: i for i, tid in enumerate(id_list)}
    embeddings = np.load("data/embeddings_norm.npy").astype(np.float32)

    conn = get_connection(DB_PATH)
    rows = conn.execute("SELECT track_id, title, artist FROM tracks").fetchall()
    conn.close()
    db = {r["track_id"]: dict(r) for r in rows}

    return index, id_list, id_to_idx, embeddings, db


def find_track_idx(desc: str, id_list, db, id_to_idx, used=None):
    used = used or set()
    idx = _find_track(desc, id_list, db, used, id_to_idx)
    if idx is None:
        print(f"  !! No match for '{desc}'")
        return None
    tid = id_list[idx]
    r = db.get(tid, {})
    print(f"  '{desc}' → {r.get('title', '?')} — {r.get('artist', '?')}")
    return idx


def find_nearest_unplayed(embeddings, target_emb, played_set, id_list):
    """Find nearest track to target in 512D, skipping played."""
    dists = np.linalg.norm(embeddings - target_emb.astype(np.float64), axis=1)
    order = np.argsort(dists)
    for idx in order:
        if id_list[idx] not in played_set:
            return idx, float(dists[idx])
    return None, None


def simulate_transit(embeddings, id_list, db, from_idx, to_idx, strategy="fractional"):
    """Simulate a 10-step transit and report distances."""
    STEPS = 10
    played = set()
    played.add(id_list[from_idx])

    from_emb = embeddings[from_idx].astype(np.float64)
    to_emb = embeddings[to_idx].astype(np.float64)

    initial_dist = float(np.linalg.norm(to_emb - from_emb))

    current_emb = from_emb.copy()
    current_idx = from_idx

    print(f"\n  Strategy: {strategy}")
    print(f"  Initial distance: {initial_dist:.4f}")
    print(f"  {'Step':>4}  {'Frac':>6}  {'Dist→Target':>12}  {'Reduction':>10}  {'Track'}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*40}")

    prev_dist = initial_dist

    for step in range(1, STEPS + 1):
        remaining = STEPS - step + 1

        if strategy == "fractional":
            # Original: 1/N of remaining distance
            fraction = 1.0 / remaining
        elif strategy == "half":
            # Always cover half the remaining distance
            fraction = 0.5
        elif strategy == "aggressive":
            # Cover more early: 1/sqrt(remaining)
            fraction = 1.0 / (remaining ** 0.5)
        else:
            fraction = 1.0 / remaining

        # Compute target point
        direction = to_emb - current_emb
        target_point = current_emb + direction * fraction
        # Re-normalize
        norm = np.linalg.norm(target_point)
        if norm > 1e-8:
            target_point /= norm

        # Find nearest unplayed
        pick_idx, pick_dist = find_nearest_unplayed(
            embeddings, target_point.astype(np.float32), played, id_list
        )
        if pick_idx is None:
            print(f"  {step:>4}  No unplayed tracks found!")
            break

        played.add(id_list[pick_idx])
        pick_emb = embeddings[pick_idx].astype(np.float64)
        dist_to_target = float(np.linalg.norm(to_emb - pick_emb))
        reduction = prev_dist - dist_to_target

        r = db.get(id_list[pick_idx], {})
        name = f"{r.get('title', '?')} — {r.get('artist', '?')}"

        print(f"  {step:>4}  {fraction:>5.2f}  {dist_to_target:>12.4f}  "
              f"{reduction:>+9.4f}  {name[:50]}")

        current_emb = pick_emb
        current_idx = pick_idx
        prev_dist = dist_to_target

    final_dist = prev_dist
    print(f"\n  Final distance: {final_dist:.4f}  "
          f"(reduced {initial_dist - final_dist:.4f} = "
          f"{100 * (initial_dist - final_dist) / initial_dist:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Test transit strategies")
    parser.add_argument("tracks", nargs="+", help="Track descriptions (Artist — Title)")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Test all consecutive pairs")
    parser.add_argument("--strategies", default="fractional,half,aggressive",
                        help="Comma-separated strategies to test")
    args = parser.parse_args()

    print("Loading data...")
    index, id_list, id_to_idx, embeddings, db = load_data()

    # Resolve track names
    print("\nResolving tracks:")
    used = set()
    indices = []
    for desc in args.tracks:
        idx = find_track_idx(desc, id_list, db, id_to_idx, used)
        if idx is None:
            sys.exit(1)
        used.add(idx)
        indices.append(idx)

    strategies = args.strategies.split(",")

    # Build pairs
    if args.all_pairs or len(indices) > 2:
        pairs = [(indices[i], indices[(i + 1) % len(indices)])
                 for i in range(len(indices))]
    else:
        pairs = [(indices[0], indices[1])]

    for from_idx, to_idx in pairs:
        from_r = db.get(id_list[from_idx], {})
        to_r = db.get(id_list[to_idx], {})
        print(f"\n{'='*70}")
        print(f"Transit: {from_r.get('title', '?')} → {to_r.get('title', '?')}")
        print(f"{'='*70}")

        for strategy in strategies:
            simulate_transit(embeddings, id_list, db, from_idx, to_idx, strategy)


if __name__ == "__main__":
    main()
