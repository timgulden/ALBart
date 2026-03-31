"""Simulate DJ trajectories without playing anything.

Uses the exact same pure logic as the live DJ (core/navigation.py),
querying PostgreSQL for neighbors.  No Spotify connection needed.

Usage:
    python3 tools/dj_simulate.py
    python3 tools/dj_simulate.py --seed "paint it black"
    python3 tools/dj_simulate.py --sets 3 --per-set 15
    python3 tools/dj_simulate.py --song-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.core.navigation import on_neighbors_found, on_track_played  # noqa: E402
from albart.core.sampling import get_recent_artists, select_from_candidates  # noqa: E402
from albart.core.state import DJState  # noqa: E402
from albart.effects.database import DatabaseClient, DatabaseConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate DJ trajectories")
    parser.add_argument("--seed", type=str, default=None,
                        help="Seed track (fuzzy match)")
    parser.add_argument("--sets", type=int, default=5,
                        help="Number of sets (default: 5)")
    parser.add_argument("--per-set", type=int, default=10,
                        help="Tracks per set (default: 10)")
    parser.add_argument("--hop-multiplier", type=float, default=5.0,
                        help="Long hop distance multiplier (default: 5x)")
    parser.add_argument("--song-k", type=int, default=10,
                        help="Candidate pool size (default: 10)")
    parser.add_argument("--seed-rng", type=int, default=42,
                        help="RNG seed for reproducibility")
    args = parser.parse_args()

    db = DatabaseClient(config=DatabaseConfig())
    rng = np.random.default_rng(args.seed_rng)
    total_tracks = db.get_total_tracks()

    print(f"Library: {total_tracks} tracks")

    # Resolve seed
    if args.seed:
        results = db.search_tracks(args.seed, limit=1)
        if not results:
            print(f"Could not match '{args.seed}'")
            return
        seed_tid = results[0].track_id
        seed_name = f"{results[0].title} — {results[0].artist}"
    else:
        # Random seed
        import psycopg2.extras
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT track_id FROM tracks WHERE embedding_512 IS NOT NULL "
                    "ORDER BY random() LIMIT 1"
                )
                seed_tid = cur.fetchone()[0]
        track = db.get_track(seed_tid)
        seed_name = f"{track.title} — {track.artist}" if track else seed_tid

    # Initialize state
    state = DJState(
        song_k=args.song_k,
        hop_multiplier=args.hop_multiplier,
        total_tracks=total_tracks,
        history=(seed_tid,),
        played=frozenset({seed_tid}),
    )

    print(f"\n{'─' * 80}")
    print(f"  {'#':>3}  {'L2':>7}  Track")
    print(f"{'─' * 80}")
    print(f"  {1:>3}  {'seed':>7}  {seed_name}")

    track_num = 1

    def _pick_normal(st: DJState) -> tuple[DJState, str | None, float]:
        """Pick the next normal hop. Returns (new_state, track_id, distance)."""
        last_tid = st.history[-1]
        emb = db.get_embedding_512(last_tid)
        if emb is None:
            return st, None, 0.0

        candidates = db.find_neighbors_512d(emb, st.song_k, st.played)
        if not candidates:
            return st, None, 0.0

        track_artists = db.get_artists_for_tracks([c[0] for c in candidates])
        result = on_neighbors_found(st, candidates, track_artists, "normal", 0.0, rng)
        next_tid = result.state.next_pick
        if next_tid is None:
            return result.state, None, 0.0

        # Compute distance for display
        next_emb = db.get_embedding_512(next_tid)
        dist = float(np.linalg.norm(
            next_emb.astype(np.float64) - emb.astype(np.float64)
        )) if next_emb is not None else 0.0

        # Record as played
        result2 = on_track_played(result.state, next_tid, 0.0)
        return result2.state, next_tid, dist

    def _pick_long(st: DJState) -> tuple[DJState, str | None, float]:
        """Pick a long hop. Returns (new_state, track_id, distance)."""
        if len(st.history) < 2:
            return _pick_normal(st)

        target = db.compute_long_hop_target(
            st.history[-2], st.history[-1], st.hop_multiplier, rng,
        )
        if target is None:
            return _pick_normal(st)

        candidates = db.find_neighbors_512d(target, 20, st.played)
        if not candidates:
            return _pick_normal(st)

        track_artists = db.get_artists_for_tracks([c[0] for c in candidates])

        # Use pending_hop_type to mark as long hop
        st = st.model_copy(update={"pending_hop_type": "LONG"})
        result = on_neighbors_found(st, candidates, track_artists, "LONG", 0.0, rng)
        next_tid = result.state.next_pick
        if next_tid is None:
            return result.state, None, 0.0

        # Compute distance from current track
        curr_emb = db.get_embedding_512(st.history[-1])
        next_emb = db.get_embedding_512(next_tid)
        dist = float(np.linalg.norm(
            next_emb.astype(np.float64) - curr_emb.astype(np.float64)
        )) if curr_emb is not None and next_emb is not None else 0.0

        result2 = on_track_played(result.state, next_tid, 0.0)
        return result2.state, next_tid, dist

    def _track_name(tid: str) -> str:
        t = db.get_track(tid)
        return f"{t.title} — {t.artist}" if t else tid

    for s in range(args.sets):
        if s > 0:
            # Long hop between sets
            state, next_tid, dist = _pick_long(state)
            if next_tid:
                track_num += 1
                print(f"\n  {'═' * 76}")
                print(f"  LONG HOP (x{args.hop_multiplier:.0f} trajectory, L2={dist:.4f})")
                print(f"  {'═' * 76}\n")
                print(f"  {track_num:>3}  {dist:>7.4f}  {_track_name(next_tid)}")
                start = 1
            else:
                print("  ... could not find long hop target")
                start = 0
        else:
            start = 1

        for i in range(start, args.per_set):
            state, next_tid, dist = _pick_normal(state)
            if not next_tid:
                print("  ... no more unplayed tracks nearby")
                break
            track_num += 1
            print(f"  {track_num:>3}  {dist:>7.4f}  {_track_name(next_tid)}")

    print(f"\n{'─' * 80}")
    print(f"Total: {len(state.played)} tracks played")

    # Stats
    if len(state.history) >= 2:
        dists = []
        for i in range(1, len(state.history)):
            e1 = db.get_embedding_512(state.history[i - 1])
            e2 = db.get_embedding_512(state.history[i])
            if e1 is not None and e2 is not None:
                dists.append(float(np.linalg.norm(
                    e2.astype(np.float64) - e1.astype(np.float64)
                )))
        if dists:
            print(f"Hop distances: min={min(dists):.4f}  mean={np.mean(dists):.4f}  "
                  f"max={max(dists):.4f}")

    # Artist diversity
    artists = set()
    for tid in state.history:
        t = db.get_track(tid)
        if t:
            artists.add(t.artist)
    print(f"Unique artists: {len(artists)} / {len(state.history)} tracks")


if __name__ == "__main__":
    main()
