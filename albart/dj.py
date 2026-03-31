"""ALBart DJ — automated music exploration through embedding space.

Navigates through a Spotify library using CLAP audio embeddings:
  - Normal hops: pick from nearest unplayed tracks (weighted random)
  - Long hops: extrapolate trajectory to jump between genre regions
  - Orbit mode: cycle through anchor tracks with dwell/transit phases
  - Mood filtering: CLAP text embeddings constrain the candidate pool

Usage:
    python3 -m albart.dj
    python3 -m albart.dj --seed "track name"
    python3 -m albart.dj --mode listen
    python3 -m albart.dj --song-k 5 --hop-interval 20
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("albart.dj")


def main() -> None:
    """CLI entry point using the Engine architecture."""
    parser = argparse.ArgumentParser(description="ALBart DJ Mode")
    parser.add_argument(
        "--seed", type=str, default=None,
        help="Seed track name (fuzzy matched) or Spotify track ID",
    )
    parser.add_argument(
        "--hop-interval", type=float, default=30.0,
        help="Minutes between long hops (default: 30)",
    )
    parser.add_argument(
        "--hop-multiplier", type=float, default=5.0,
        help="Long hop distance multiplier (default: 5x)",
    )
    parser.add_argument(
        "--mode", type=str, default="exact", choices=["exact", "listen"],
        help="exact: hop from stored preview embedding (default). "
             "listen: hop from live audio embedding.",
    )
    parser.add_argument(
        "--port", type=int, default=57002,
        help="UDP port to receive live embeddings (listen mode)",
    )
    parser.add_argument(
        "--song-k", type=int, default=10,
        help="Number of candidate tracks to consider (1-50, default: 10)",
    )
    parser.add_argument(
        "--mood", type=str, default=None,
        help='Mood description, e.g. "chill dinner party, jazz, '
             'downtempo electronic, no opera".',
    )
    args = parser.parse_args()

    from albart.effects.broadcast import BroadcastClient
    from albart.effects.database import DatabaseClient, DatabaseConfig
    from albart.effects.spotify import SpotifyClient
    from albart.effects.udp_listener import UDPListener
    from albart.engine import Engine

    db = DatabaseClient(config=DatabaseConfig())
    spotify = SpotifyClient.create()
    broadcast = BroadcastClient()
    udp = UDPListener(port=args.port)

    engine = Engine(
        db=db,
        spotify=spotify,
        broadcast=broadcast,
        udp_listener=udp,
        mode=args.mode,
        song_k=max(1, min(50, args.song_k)),
        hop_multiplier=args.hop_multiplier,
        hop_interval_minutes=args.hop_interval,
    )

    # Resolve seed
    seed = None
    if args.seed:
        track = db.get_track(args.seed)
        if track:
            seed = args.seed
        else:
            results = db.search_tracks(args.seed, limit=1)
            if results:
                seed = results[0].track_id
                logger.info("Matched seed: %s — %s", results[0].title, results[0].artist)
            else:
                logger.warning("Could not match seed '%s'", args.seed)

    # Handle mood setup
    if args.mood:
        from albart.core.mood import update_mood
        result = update_mood(engine.get_snapshot(), mood_text=args.mood, descriptors=[])
        engine.update_param("mood", result.state.mood)

    engine.run(seed_track_id=seed)


if __name__ == "__main__":
    main()
