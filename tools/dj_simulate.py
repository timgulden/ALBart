"""Simulate DJ trajectories without playing anything.

Prints 50 tracks: 10 per set with long hops between sets.
Uses the same hop logic as ALBart DJ (via DJ class).

Usage:
    python tools/dj_simulate.py
    python tools/dj_simulate.py --seed "paint it black"
    python tools/dj_simulate.py --sets 3 --per-set 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.dj import DJ, find_seed_track  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate DJ trajectories")
    parser.add_argument("--seed", type=str, default=None,
                        help="Seed track (fuzzy match)")
    parser.add_argument("--sets", type=int, default=5,
                        help="Number of sets (default: 5)")
    parser.add_argument("--per-set", type=int, default=10,
                        help="Tracks per set (default: 10)")
    parser.add_argument("--hop-multiplier", type=float, default=5.0,
                        help="Long hop distance multiplier (default: 5×)")
    args = parser.parse_args()

    # Create DJ but don't connect to Spotify — we only need the hop logic
    dj = DJ.__new__(DJ)
    # Manually init just the data parts (skip Spotify + UDP)
    import logging
    import numpy as np
    import faiss
    from albart.pipeline.database import DB_PATH, get_connection
    from albart.pipeline.embedder import FAISS_NORM_INDEX_PATH, FAISS_NORM_IDS_PATH
    from albart.utils import DATA_DIR

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.ERROR)
    logging.getLogger("dj_mode").setLevel(logging.WARNING)

    print("Loading data...")
    dj._index = faiss.read_index(str(FAISS_NORM_INDEX_PATH))
    ids_arr = np.load(str(FAISS_NORM_IDS_PATH), allow_pickle=True)
    dj._id_list = [str(t) for t in ids_arr]
    dj._N = len(dj._id_list)
    dj._id_to_idx = {tid: i for i, tid in enumerate(dj._id_list)}
    dj._embeddings = np.load(str(DATA_DIR / "embeddings_norm.npy")).astype(np.float32)

    conn = get_connection(DB_PATH)
    rows = conn.execute("SELECT track_id, title, artist FROM tracks").fetchall()
    conn.close()
    dj._db = {r["track_id"]: r for r in rows}

    dj.hop_multiplier = args.hop_multiplier
    dj._played = set()
    dj._history = []
    dj._rng = np.random.default_rng(42)
    dj._mode = "exact"
    dj._live_emb = None
    dj._temperature = 0.5
    dj._mood_embs = None

    # Find seed
    seed_tid = None
    if args.seed:
        seed_tid = find_seed_track(args.seed, dj._db)
        if seed_tid and seed_tid in dj._id_to_idx:
            print(f"Seed: {dj._track_name(seed_tid)}")
        else:
            print(f"Could not match '{args.seed}'")
            return
    else:
        seed_tid = dj._id_list[dj._rng.integers(0, dj._N)]
        print(f"Random seed: {dj._track_name(seed_tid)}")

    dj._played.add(seed_tid)
    dj._history.append(seed_tid)

    total = args.sets * args.per_set
    track_num = 0

    print(f"\n{'─' * 80}")
    print(f"  {'#':>3}  {'L2':>7}  Track")
    print(f"{'─' * 80}")

    for s in range(args.sets):
        if s > 0:
            # Long hop
            next_tid = dj._pick_long_hop()
            if next_tid:
                d = float(np.linalg.norm(
                    dj._get_embedding(next_tid).astype(np.float64) -
                    dj._get_embedding(dj._history[-1]).astype(np.float64)
                ))
                dj._played.add(next_tid)
                dj._history.append(next_tid)
                track_num += 1
                print(f"\n  {'═' * 76}")
                print(f"  LONG HOP (×{dj.hop_multiplier:.0f} trajectory, L2={d:.4f})")
                print(f"  {'═' * 76}\n")
                print(f"  {track_num:>3}  {d:>7.4f}  {dj._track_name(next_tid)}")
                start = 1
            else:
                start = 0
        else:
            print(f"  {1:>3}  {'seed':>7}  {dj._track_name(seed_tid)}")
            track_num = 1
            start = 1

        for i in range(start, args.per_set):
            emb = dj._get_embedding(dj._history[-1])
            next_tid = dj._pick_normal_hop(emb)
            if not next_tid:
                print("  ... no more unplayed tracks nearby")
                break
            d = float(np.linalg.norm(
                dj._get_embedding(next_tid).astype(np.float64) -
                emb.astype(np.float64)
            ))
            dj._played.add(next_tid)
            dj._history.append(next_tid)
            track_num += 1
            print(f"  {track_num:>3}  {d:>7.4f}  {dj._track_name(next_tid)}")

    print(f"\n{'─' * 80}")
    print(f"Total: {len(dj._played)} tracks played")


if __name__ == "__main__":
    main()
