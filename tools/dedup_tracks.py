"""Find and remove near-duplicate tracks from the database.

Compares all track pairs by 512D embedding distance. Tracks within
the threshold are considered duplicates of the same recording (remasters,
reissues, compilation variants).

Usage:
    python3 tools/dedup_tracks.py                 # report only
    python3 tools/dedup_tracks.py --apply          # delete duplicates
    python3 tools/dedup_tracks.py --threshold 0.05 # wider threshold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.effects.database import DatabaseClient, DatabaseConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and remove near-duplicate tracks")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="L2 distance threshold (default: 0.01)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete duplicates (default: report only)")
    args = parser.parse_args()

    db = DatabaseClient(config=DatabaseConfig())
    total = db.get_total_tracks()
    print(f"Library: {total} tracks with embeddings")
    print(f"Threshold: {args.threshold}")
    print()

    dupes = db.find_duplicates(threshold=args.threshold)
    if not dupes:
        print("No duplicates found.")
        return

    # A track may appear in multiple pairs — collect unique drops
    to_drop: dict[str, tuple[str, str, str, float]] = {}  # tid -> (title, artist, kept_by, dist)
    for keep_tid, keep_title, keep_artist, drop_tid, drop_title, drop_artist, dist in dupes:
        if drop_tid not in to_drop:
            to_drop[drop_tid] = (drop_title, drop_artist, f"{keep_title} — {keep_artist}", dist)

    # Report
    print(f"{'─' * 90}")
    print(f"  {'Keep':40s}  {'Drop':40s}  {'L2':>6}")
    print(f"{'─' * 90}")
    for keep_tid, keep_title, keep_artist, drop_tid, drop_title, drop_artist, dist in dupes:
        keep_name = f"{keep_title[:18]} — {keep_artist[:18]}"
        drop_name = f"{drop_title[:18]} — {drop_artist[:18]}"
        print(f"  {keep_name:40s}  {drop_name:40s}  {dist:.4f}")

    print(f"{'─' * 90}")
    print(f"\n{len(dupes)} duplicate pairs found")
    print(f"{len(to_drop)} unique tracks to remove")
    print(f"{total - len(to_drop)} tracks would remain")

    if not args.apply:
        print("\nDry run — no changes made. Use --apply to delete duplicates.")
        return

    # Apply
    drop_ids = list(to_drop.keys())
    deleted = db.delete_tracks(drop_ids)
    remaining = db.get_total_tracks()
    print(f"\nDeleted {deleted} duplicate tracks. {remaining} tracks remain.")


if __name__ == "__main__":
    main()
