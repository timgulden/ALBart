"""ALBart DJ Mode — automated music exploration through embedding space.

Plays tracks from your Spotify library, navigating through embedding space:
  - Normal hops: pick from the ~5 nearest unplayed tracks (weighted random)
  - Long hops (every ~30 min): extrapolate the recent trajectory 5× further,
    landing in a related but distinct genre region
  - Override: change the track in Spotify or click a track in the map display

Two modes:
  - exact:  hop from the stored preview embedding of the track just played.
            Controlled, repeatable — the preview is what the index was built from.
  - listen: hop from the main engine's latest live audio embedding.
            Reflects what the song actually sounds like right now, but may
            drift at the tail end of a song.

Usage:
    python tools/dj_mode.py                     # exact mode (default)
    python tools/dj_mode.py --mode listen       # anchor to live audio
    python tools/dj_mode.py --seed "track name"
    python tools/dj_mode.py --hop-interval 20   # long hop every 20 min
"""

from __future__ import annotations

import argparse
import logging
import pickle
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth

sys.path.insert(0, str(Path(__file__).parent.parent))

from albart.pipeline.database import DB_PATH, get_connection
from albart.pipeline.embedder import FAISS_RAW_INDEX_PATH, FAISS_RAW_IDS_PATH
from albart.utils import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("dj_mode")

EMBEDDINGS_PATH = DATA_DIR / "embeddings_raw.npy"
OVERRIDE_PATH   = DATA_DIR / "dj_override.txt"

# How many nearby tracks to consider for normal hops (wide enough to find
# different artists even in dense same-artist clusters)
NORMAL_HOP_K = 20


class DJ:
    def __init__(
        self,
        hop_interval_minutes: float = 30.0,
        hop_multiplier: float = 5.0,
        mode: str = "exact",
        udp_port: int = 57001,
    ) -> None:
        self.hop_interval = hop_interval_minutes * 60.0  # seconds
        self.hop_multiplier = hop_multiplier
        self._mode = mode  # "exact" or "listen"

        # ── Load embeddings + metadata ────────────────────────────────────
        logger.info("Loading embeddings and metadata...")
        import faiss
        self._index = faiss.read_index(str(FAISS_RAW_INDEX_PATH))
        ids_arr = np.load(str(FAISS_RAW_IDS_PATH), allow_pickle=True)
        self._id_list = [str(t) for t in ids_arr]
        self._N = len(self._id_list)
        self._id_to_idx = {tid: i for i, tid in enumerate(self._id_list)}

        self._embeddings = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)

        conn = get_connection(DB_PATH)
        rows = conn.execute(
            "SELECT track_id, title, artist FROM tracks"
        ).fetchall()
        conn.close()
        self._db = {r["track_id"]: r for r in rows}

        # ── Spotify client ────────────────────────────────────────────────
        logger.info("Connecting to Spotify...")
        self._sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            scope="user-read-playback-state,user-modify-playback-state",
        ))
        # Verify connection
        user = self._sp.current_user()
        logger.info("Logged in as: %s", user["display_name"])

        # ── Live audio embedding listener (coupled mode) ─────────────────
        self._live_emb: np.ndarray | None = None
        self._live_top1: str | None = None
        if self._mode == "listen":
            self._start_udp_listener(udp_port)
            logger.info("Listen mode: hopping from live audio embedding (UDP %d)",
                        udp_port)
        else:
            logger.info("Exact mode: hopping from stored preview embeddings")

        # ── DJ state ─────────────────────────────────────────────────────
        self._played: set[str] = set()
        self._history: list[str] = []  # ordered list of played track IDs
        self._last_hop_time: float = time.monotonic()
        self._queued_next: str | None = None
        self._pending_hop_type: str | None = None
        self._rng = np.random.default_rng()

    def _start_udp_listener(self, port: int) -> None:
        """Background thread that receives embeddings from the main engine."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass
        sock.bind(("127.0.0.1", port))
        sock.settimeout(1.0)

        def _listen():
            while True:
                try:
                    data, _ = sock.recvfrom(65536)
                    payload = pickle.loads(data)
                    self._live_emb = payload.get("raw")
                    self._live_top1 = payload.get("top1")
                except socket.timeout:
                    continue
                except Exception:
                    continue

        t = threading.Thread(target=_listen, daemon=True, name="DJ-UDP")
        t.start()

    def _track_name(self, tid: str) -> str:
        r = self._db.get(tid)
        if r:
            return f"{r['title']} — {r['artist']}"
        return tid

    def _get_embedding(self, tid: str) -> np.ndarray | None:
        idx = self._id_to_idx.get(tid)
        if idx is None:
            return None
        return self._embeddings[idx]

    def _find_nearest_unplayed(
        self, emb: np.ndarray, k: int = 10
    ) -> list[tuple[str, float]]:
        """Find K nearest unplayed tracks by L2 distance.

        Skips near-duplicates (L2 < 0.01) of any already-played track.
        """
        search_k = min(k * 5 + len(self._played), self._N)
        q = emb.reshape(1, -1).astype(np.float32)
        dists, idxs = self._index.search(q, search_k)

        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < 0:
                continue
            tid = self._id_list[idx]
            if tid in self._played:
                continue
            l2 = float(np.sqrt(max(dist, 0)))
            # Skip near-duplicates of played tracks (same recording, different remaster)
            candidate_emb = self._embeddings[self._id_to_idx[tid]]
            is_dup = False
            for played_tid in self._history[-20:]:  # check recent history
                played_idx = self._id_to_idx.get(played_tid)
                if played_idx is not None:
                    d = float(np.linalg.norm(
                        candidate_emb.astype(np.float64) -
                        self._embeddings[played_idx].astype(np.float64)
                    ))
                    if d < 0.01:
                        is_dup = True
                        break
            if is_dup:
                continue
            results.append((tid, l2))
            if len(results) >= k:
                break
        return results

    def _recent_artists(self, n: int = 3) -> set[str]:
        """Artists from the last N played tracks."""
        artists: set[str] = set()
        for tid in self._history[-n:]:
            r = self._db.get(tid)
            if r and r["artist"]:
                artists.add(r["artist"].lower())
        return artists

    def _pick_normal_hop(self, current_emb: np.ndarray) -> str | None:
        """Pick from the nearest unplayed tracks (weighted by closeness).

        Penalizes tracks by recently-played artists (0.1× weight).
        """
        candidates = self._find_nearest_unplayed(current_emb, k=NORMAL_HOP_K)
        if not candidates:
            logger.warning("No unplayed tracks found nearby!")
            return None

        recent_artists = self._recent_artists(3)

        tids = [c[0] for c in candidates]
        dists = np.array([c[1] for c in candidates])
        weights = 1.0 / np.maximum(dists, 1e-8)

        # Penalize same artist as recent tracks
        for i, tid in enumerate(tids):
            r = self._db.get(tid)
            if r and r["artist"] and r["artist"].lower() in recent_artists:
                weights[i] *= 0.1

        weights /= weights.sum()

        chosen = self._rng.choice(len(tids), p=weights)
        return tids[chosen]

    def _pick_long_hop(self) -> str | None:
        """Extrapolate the recent trajectory and jump further along it."""
        if len(self._history) < 2:
            return self._pick_normal_hop(
                self._get_embedding(self._history[-1])
            )

        # Direction from second-to-last to last track
        emb_prev = self._get_embedding(self._history[-2])
        emb_curr = self._get_embedding(self._history[-1])
        if emb_prev is None or emb_curr is None:
            return self._pick_normal_hop(emb_curr or emb_prev)

        direction = emb_curr.astype(np.float64) - emb_prev.astype(np.float64)
        dist = float(np.linalg.norm(direction))
        if dist < 1e-8:
            # Same track or very similar — pick random direction
            direction = self._rng.normal(size=512)
            dist = float(np.linalg.norm(direction))

        direction /= dist  # unit vector

        # Extrapolate: jump multiplier × distance in the same direction
        jump_dist = dist * self.hop_multiplier
        target = emb_curr.astype(np.float64) + direction * jump_dist
        target = (target / (np.linalg.norm(target) + 1e-8)).astype(
            np.float32
        )  # re-normalize (CLAP embeddings are L2-normalized)

        # Find nearest unplayed to the target point
        candidates = self._find_nearest_unplayed(target, k=NORMAL_HOP_K)
        if not candidates:
            return self._pick_normal_hop(emb_curr)

        # Pick from candidates (slight randomness)
        tids = [c[0] for c in candidates]
        dists = np.array([c[1] for c in candidates])
        weights = 1.0 / np.maximum(dists, 1e-8)
        weights /= weights.sum()
        chosen = self._rng.choice(len(tids), p=weights)

        logger.info(
            "LONG HOP: trajectory %s → %s, extrapolated %.2f × %.4f = %.4f",
            self._track_name(self._history[-2])[:30],
            self._track_name(self._history[-1])[:30],
            self.hop_multiplier, dist, jump_dist,
        )
        return tids[chosen]

    def _play_track(self, tid: str, via_queue: bool = False) -> bool:
        """Start playing a track on Spotify. Returns True on success."""
        if not via_queue:
            uri = f"spotify:track:{tid}"
            try:
                self._sp.start_playback(uris=[uri])
            except Exception as e:
                logger.error("Playback failed for %s: %s", tid, e)
                return False
        self._played.add(tid)
        self._history.append(tid)
        hop_label = ""
        if hasattr(self, "_pending_hop_type") and self._pending_hop_type == "LONG":
            hop_label = " ═══ LONG HOP"
            self._pending_hop_type = None
        logger.info(
            "▶  %s  [played: %d / %d]%s",
            self._track_name(tid), len(self._played), self._N, hop_label,
        )
        return True

    def _get_current_spotify_track(self) -> str | None:
        """Get the currently playing Spotify track ID (None if nothing)."""
        try:
            pb = self._sp.current_playback()
            if pb and pb.get("is_playing") and pb.get("item"):
                return pb["item"]["id"]
            else:
                logger.info("Spotify: nothing playing or paused")
        except Exception as e:
            logger.warning("Could not get Spotify playback: %s", e)
        return None

    def _get_remaining_ms(self) -> int:
        """Get remaining playback time in ms. Returns 0 if unknown."""
        try:
            pb = self._sp.current_playback()
            if pb and pb.get("item"):
                duration = pb["item"]["duration_ms"]
                progress = pb.get("progress_ms", 0)
                return max(0, duration - progress)
        except Exception:
            pass
        return 0

    def _check_override(self) -> str | None:
        """Check if the map display sent a click-to-play override."""
        if OVERRIDE_PATH.exists():
            try:
                tid = OVERRIDE_PATH.read_text().strip()
                OVERRIDE_PATH.unlink()
                if tid and tid in self._id_to_idx:
                    logger.info("Override from map: %s", self._track_name(tid))
                    return tid
            except Exception:
                pass
        return None

    def run(self, seed_track_id: str | None = None) -> None:
        """Main DJ loop."""
        # Seed from argument, current Spotify track, or random
        if seed_track_id is None:
            seed_track_id = self._get_current_spotify_track()
        if seed_track_id is None or seed_track_id not in self._id_to_idx:
            if seed_track_id:
                logger.warning("Track %s not in library — picking random start",
                               seed_track_id)
            seed_track_id = self._id_list[self._rng.integers(0, self._N)]
            self._play_track(seed_track_id)
        else:
            # Already playing — don't restart, just mark as current
            self._played.add(seed_track_id)
            self._history.append(seed_track_id)
            logger.info("Continuing from: %s", self._track_name(seed_track_id))

        self._last_hop_time = time.monotonic()

        try:
            while True:
                time.sleep(3)  # poll every 3 seconds

                # Check for map click override
                override = self._check_override()
                if override:
                    self._play_track(override)
                    self._last_hop_time = time.monotonic()
                    continue

                # Check if Spotify track was manually changed
                current = self._get_current_spotify_track()
                if (current and current in self._id_to_idx
                        and current != self._history[-1]):
                    logger.info(
                        "Spotify manual change detected: %s",
                        self._track_name(current),
                    )
                    self._played.add(current)
                    self._history.append(current)
                    self._last_hop_time = time.monotonic()
                    continue

                # Check remaining playback time
                remaining = self._get_remaining_ms()

                # Pick next track when ~10s remain (gives time to compute)
                if remaining > 10000 or remaining == 0:
                    continue

                # If we already queued the next track, just wait
                if self._queued_next:
                    if remaining > 1500:
                        continue
                    # Track ended — Spotify will auto-play the queued track.
                    # Just mark it as played in our state.
                    self._play_track(self._queued_next, via_queue=True)
                    self._queued_next = None
                    continue

                # Track ending — pick next
                now = time.monotonic()
                time_since_hop = now - self._last_hop_time

                current_emb = self._get_embedding(self._history[-1])
                if current_emb is None:
                    continue

                # Listen mode: use the live audio embedding from the engine
                if self._mode == "listen" and self._live_emb is not None:
                    current_emb = self._live_emb

                if time_since_hop >= self.hop_interval:
                    next_tid = self._pick_long_hop()
                    self._last_hop_time = now
                    hop_type = "LONG"
                else:
                    next_tid = self._pick_normal_hop(current_emb)
                    hop_type = "normal"

                if next_tid:
                    self._pending_hop_type = hop_type
                    # Queue via Spotify — lets current track finish naturally
                    uri = f"spotify:track:{next_tid}"
                    try:
                        self._sp.add_to_queue(uri)
                        self._queued_next = next_tid
                        prefix = "LONG HOP queued" if hop_type == "LONG" else "Queued"
                        logger.info(
                            "%s: %s", prefix, self._track_name(next_tid),
                        )
                    except Exception as e:
                        logger.warning("Queue failed, playing directly: %s", e)
                        self._play_track(next_tid)
                else:
                    logger.warning("Could not find next track — resetting played set")
                    self._played.clear()

        except KeyboardInterrupt:
            logger.info("DJ mode stopped. Played %d tracks.", len(self._played))


def find_seed_track(query: str, db: dict) -> str | None:
    """Fuzzy match a track name from the library."""
    query_lower = query.lower()
    for tid, row in db.items():
        title = (row["title"] or "").lower()
        artist = (row["artist"] or "").lower()
        if query_lower in title or query_lower in f"{title} {artist}":
            return tid
    return None


def main() -> None:
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
        help="Long hop distance multiplier (default: 5×)",
    )
    parser.add_argument(
        "--mode", type=str, default="exact", choices=["exact", "listen"],
        help="exact: hop from stored preview embedding (default). "
             "listen: hop from live audio embedding (requires main engine).",
    )
    parser.add_argument(
        "--port", type=int, default=57001,
        help="UDP port to receive engine broadcasts (listen mode)",
    )
    args = parser.parse_args()

    dj = DJ(
        hop_interval_minutes=args.hop_interval,
        hop_multiplier=args.hop_multiplier,
        mode=args.mode,
        udp_port=args.port,
    )

    seed = None
    if args.seed:
        # Try as track ID first, then fuzzy match
        if args.seed in dj._id_to_idx:
            seed = args.seed
        else:
            seed = find_seed_track(args.seed, dj._db)
            if seed:
                logger.info("Matched seed: %s", dj._track_name(seed))
            else:
                logger.warning("Could not match seed '%s'", args.seed)

    dj.run(seed_track_id=seed)


if __name__ == "__main__":
    main()
