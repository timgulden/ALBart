"""Simulate an orbit journey without playing anything.

Uses 25D UMAP for all navigation — same logic as the live engine.
Pass anchor tracks as arguments, get a full orbit printout.

Usage:
    python3 tools/orbit_simulate.py "Ray Charles — I've Got a Woman" "The Stooges — Fun House" "NewJeans — Super Shy"
    python3 tools/orbit_simulate.py --dwell 5 --transit 10 "Artist — Title" "Artist — Title" ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.core.orbit_logic import find_track, normalize_search  # noqa: E402
from albart.core.sampling import select_from_candidates, get_recent_artists  # noqa: E402
from albart.effects.database import DatabaseClient, DatabaseConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate an orbit journey through anchor tracks"
    )
    parser.add_argument("anchors", nargs="+",
                        help='Anchor tracks: "Artist — Title" or partial match')
    parser.add_argument("--dwell", type=int, default=5,
                        help="Tracks per dwell phase (default: 5)")
    parser.add_argument("--transit", type=int, default=10,
                        help="Steps per transit phase (default: 10)")
    parser.add_argument("--song-k", type=int, default=5,
                        help="Candidate pool size (default: 5)")
    parser.add_argument("--seed-rng", type=int, default=42,
                        help="RNG seed for reproducibility")
    args = parser.parse_args()

    db = DatabaseClient(config=DatabaseConfig())
    rng = np.random.default_rng(args.seed_rng)

    # Load all 25D embeddings
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT track_id, umap_25d FROM tracks WHERE umap_25d IS NOT NULL")
            rows = cur.fetchall()

    tids = [r[0] for r in rows]
    coords = np.array([r[1] for r in rows], dtype=np.float32)
    tid_to_idx = {t: i for i, t in enumerate(tids)}

    # Resolve anchors
    all_tracks = db.get_all_tracks()
    used_ids: set[str] = set()
    anchors: list[dict] = []

    print("Resolving anchors:")
    for desc in args.anchors:
        tid = find_track(desc, all_tracks, used_ids)
        if tid is None:
            print("  MISS: %s" % desc)
            continue
        used_ids.add(tid)
        t = db.get_track(tid)
        idx = tid_to_idx.get(tid)
        if idx is None:
            print("  NO EMBEDDING: %s — %s" % (t.title, t.artist))
            continue
        anchors.append({"tid": tid, "title": t.title, "artist": t.artist, "idx": idx})
        print("  [%d] %s — %s" % (len(anchors) - 1, t.title, t.artist))

    if len(anchors) < 2:
        print("\nNeed at least 2 anchors.")
        return

    anchor_ids = frozenset(a["tid"] for a in anchors)
    n_anchors = len(anchors)

    print("\nOrbit: %d anchors, %d dwell tracks, %d transit steps, K=%d" % (
        n_anchors, args.dwell, args.transit, args.song_k))

    played: set[str] = set(anchor_ids)
    history: tuple[str, ...] = ()
    track_num = 0

    def _track_name(tid: str) -> str:
        t = db.get_track(tid)
        return "%s — %s" % (t.title[:35], t.artist[:25]) if t else tid

    def _pick_nearest(target_25d, exclude, k):
        dists = np.linalg.norm(coords - target_25d, axis=1)
        order = np.argsort(dists)
        candidates = []
        for j in order:
            if tids[j] not in exclude:
                candidates.append((tids[j], float(dists[j])))
                if len(candidates) >= k:
                    break
        if not candidates:
            return None, 0.0
        artists = db.get_artists_for_tracks([c[0] for c in candidates])
        recent = get_recent_artists(history, artists)
        pick = select_from_candidates(
            candidates, recent_artists=recent, artist_penalty=0.01,
            allow_same_artist=False, song_k=k, track_artist_map=artists, rng=rng,
        )
        dist = 0.0
        if pick:
            dist = next((d for t, d in candidates if t == pick), 0.0)
        return pick, dist

    # Full orbit cycle
    for ai in range(n_anchors):
        anchor = anchors[ai]
        next_ai = (ai + 1) % n_anchors
        next_anchor = anchors[next_ai]

        # ── DWELL ──
        print("\n  %s" % ("═" * 70))
        print("  DWELL at [%d]: %s — %s" % (ai, anchor["title"], anchor["artist"]))
        print("  %s" % ("═" * 70))

        anchor_25d = coords[anchor["idx"]]
        for di in range(args.dwell):
            pick, dist = _pick_nearest(anchor_25d, played | anchor_ids, args.song_k)
            if pick is None:
                print("    ... no more candidates")
                break
            played.add(pick)
            history = (*history, pick)
            track_num += 1
            print("    %3d. %-45s d=%.4f" % (track_num, _track_name(pick), dist))

        # ── TRANSIT ──
        start_25d = coords[anchor["idx"]]
        end_25d = coords[next_anchor["idx"]]
        total_dist = float(np.linalg.norm(end_25d - start_25d))

        from albart.core.orbit_logic import compute_transit_steps
        transit_steps = compute_transit_steps(total_dist)

        print("\n  %s" % ("─" * 70))
        print("  TRANSIT [%d] → [%d]: %s → %s  (dist=%.4f, %d steps)" % (
            ai, next_ai, anchor["artist"][:20], next_anchor["artist"][:20],
            total_dist, transit_steps))
        print("  %s" % ("─" * 70))

        for si in range(1, transit_steps + 1):
            frac = si / transit_steps
            target = start_25d + (end_25d - start_25d) * frac
            pick, dist = _pick_nearest(target, played | anchor_ids, args.song_k)
            if pick is None:
                print("    ... no more candidates")
                break
            played.add(pick)
            history = (*history, pick)
            track_num += 1

            pick_25d = coords[tid_to_idx[pick]]
            dist_to_end = float(np.linalg.norm(pick_25d - end_25d))
            pct = (1.0 - dist_to_end / total_dist) * 100

            print("    %3d. %2d/%-2d (%5.1f%%) %-40s d=%.4f" % (
                track_num, si, transit_steps, pct, _track_name(pick), dist))

    print("\n" + "=" * 75)
    print("Total: %d tracks, %d unique artists" % (track_num, len(set(
        db.get_track(tid).artist for tid in history if db.get_track(tid)
    ))))


if __name__ == "__main__":
    main()
