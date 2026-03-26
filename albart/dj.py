"""ALBart DJ — automated music exploration through embedding space.

Plays tracks from your Spotify library, navigating through embedding space:
  - Normal hops: pick from the ~20 nearest unplayed tracks (weighted random,
    with artist penalty to avoid same-artist runs)
  - Long hops (every ~30 min): extrapolate the recent trajectory 5× further,
    landing in a related but distinct genre region
  - Override: change the track in Spotify or click a track in the MapView

Two modes:
  - exact:  hop from the stored preview embedding of the track just played.
            Controlled, repeatable — the preview is what the index was built from.
  - listen: hop from the Listener's latest live audio embedding.
            Reflects what the song actually sounds like right now, but may
            drift at the tail end of a song.

Usage:
    python -m albart.dj                         # exact mode (default)
    python -m albart.dj --mode listen           # anchor to live audio
    python -m albart.dj --seed "track name"
    python -m albart.dj --hop-interval 20       # long hop every 20 min
"""

from __future__ import annotations

import argparse
import logging
import pickle
import socket
import threading
import time

import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from albart.pipeline.database import DB_PATH, get_connection
from albart.pipeline.embedder import FAISS_NORM_INDEX_PATH, FAISS_NORM_IDS_PATH
from albart.utils import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("albart.dj")

EMBEDDINGS_PATH = DATA_DIR / "embeddings_norm.npy"
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
        udp_port: int = 57002,
        temperature: float = 0.5,
        mood: str | None = None,
    ) -> None:
        self.hop_interval = hop_interval_minutes * 60.0  # seconds
        self.hop_multiplier = hop_multiplier
        self._mode = mode  # "exact" or "listen"
        # Song temperature = K candidates (1-50). Set temperature = hop multiplier.
        self._song_k = max(1, min(50, int(temperature * NORMAL_HOP_K)
                                  if temperature <= 1.0 else int(temperature)))
        self._temperature = temperature  # keep for backward compat

        # ── Load embeddings + metadata ────────────────────────────────────
        logger.info("Loading embeddings and metadata...")
        import faiss
        self._index = faiss.read_index(str(FAISS_NORM_INDEX_PATH))
        ids_arr = np.load(str(FAISS_NORM_IDS_PATH), allow_pickle=True)
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

        # ── Map display broadcast (send embeddings directly to map) ──────
        self._map_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._map_port = 57001
        self._map_cached_tid: str | None = None
        self._map_cached_emb: np.ndarray | None = None

        # ── Live audio embedding listener (coupled mode) ─────────────────
        self._live_emb: np.ndarray | None = None
        self._live_top1: str | None = None
        # Always listen for live embeddings (needed for unknown track fallback)
        self._start_udp_listener(udp_port)
        if self._mode == "listen":
            logger.info("Listen mode: hopping from live audio embedding (UDP %d)",
                        udp_port)
        else:
            logger.info("Exact mode: hopping from stored preview embeddings "
                        "(live fallback on UDP %d for unknown tracks)", udp_port)

        # ── DJ state ─────────────────────────────────────────────────────
        self._played: set[str] = set()
        self._history: list[str] = []  # ordered list of played track IDs
        self._last_hop_time: float = time.monotonic()
        self._next_pick: str | None = None       # picked but not yet played
        self._monitored_track: str | None = None  # Spotify track ID we're watching
        self._pending_hop_type: str | None = None
        self._rng = np.random.default_rng()

        # ── Mood filtering ──────────────────────────────────────────────
        # mood_embs: (M, 512) CLAP text embeddings defining "in-bounds" space
        # If set, candidates must have cosine sim > threshold to at least one
        self._mood_embs: np.ndarray | None = None
        self._mood_embs_neg: np.ndarray | None = None
        self._mood_text: str | None = None
        self._mood_descriptors: list[str] = []
        self._mood_threshold: float = 0.35  # cosine sim; higher = stricter
        if mood:
            self._setup_mood(mood)

        logger.info("Song K=%d  Set multiplier=%.1f×",
                    self._song_k, self.hop_multiplier)

    def _setup_mood(self, mood_text: str) -> None:
        """Convert free-text mood description into CLAP embeddings for filtering.

        Sends the text to Claude to expand into ~20 genre descriptors,
        then embeds each via CLAP text inference.
        """
        import anthropic
        from albart.text_embedder import embed_texts

        self._mood_text = mood_text
        logger.info("Processing mood: %s", mood_text)

        # Step 1: Claude expands the mood into genre descriptors
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    "I'm setting up a music DJ that should play tracks matching "
                    "a specific mood. Given this description:\n\n"
                    f'"{mood_text}"\n\n'
                    "Generate exactly 20 short music genre/mood descriptors "
                    "(2-5 words each) that define what kind of music should play. "
                    "Include both positive descriptors (what TO play) and avoid "
                    "descriptors prefixed with 'NOT:' for what to avoid.\n\n"
                    "Return ONLY the list, one per line, no numbering."
                ),
            }],
        )
        lines = [
            ln.strip() for ln in response.content[0].text.strip().split("\n")
            if ln.strip()
        ]
        self._mood_descriptors = lines
        logger.info("Mood descriptors (%d):", len(lines))
        for ln in lines:
            logger.info("  %s", ln)

        # Step 2: Embed descriptors via CLAP text inference
        positive = [ln for ln in lines if not ln.upper().startswith("NOT:")]
        negative = [ln[4:].strip() for ln in lines if ln.upper().startswith("NOT:")]

        if positive:
            self._mood_embs = embed_texts(positive)
            logger.info("Mood filter: %d positive descriptors embedded",
                        len(positive))
        else:
            self._mood_embs = None

        if negative:
            self._mood_embs_neg = embed_texts(negative)
            logger.info("Mood filter: %d negative descriptors embedded",
                        len(negative))
        else:
            self._mood_embs_neg = None

        if self._mood_embs is None and self._mood_embs_neg is None:
            logger.warning("No mood descriptors — filter disabled")

    def _is_in_mood(self, tid: str) -> bool:
        """Check if a track is within the mood-defined region.

        A track passes if:
          1. It matches at least one positive descriptor (above threshold)
          2. It does NOT match any negative descriptor (above threshold)
        """
        if self._mood_embs is None and self._mood_embs_neg is None:
            return True
        emb = self._get_embedding(tid)
        if emb is None:
            return True
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

        # Must match at least one positive descriptor
        if self._mood_embs is not None:
            pos_sims = self._mood_embs @ emb_norm
            if float(np.max(pos_sims)) < self._mood_threshold:
                return False

        # Must NOT match any negative descriptor
        if self._mood_embs_neg is not None:
            neg_sims = self._mood_embs_neg @ emb_norm
            if float(np.max(neg_sims)) > self._mood_threshold:
                return False

        return True

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
        """Pick from nearest unplayed tracks.

        Temperature controls the pool size and weight sharpness.
        Mood filter rejects tracks outside the mood region.
        Artist penalty avoids same-artist runs.
        """
        candidates = self._find_nearest_unplayed(current_emb, k=self._song_k)
        if not candidates:
            logger.warning("No unplayed tracks found nearby!")
            return None

        # Mood filter: reject candidates outside the mood region
        if self._mood_embs is not None:
            candidates = [(t, d) for t, d in candidates if self._is_in_mood(t)]
            if not candidates:
                logger.warning("No in-mood tracks nearby — relaxing filter")
                candidates = self._find_nearest_unplayed(current_emb, k=k * 3)
                candidates = [(t, d) for t, d in candidates if self._is_in_mood(t)]
            if not candidates:
                return None

        recent_artists = self._recent_artists(3)

        tids = [c[0] for c in candidates]
        dists = np.array([c[1] for c in candidates])

        # With K=1, always pick nearest (deterministic)
        if self._song_k <= 1:
            return tids[0]
        sharpness = 1.0  # 1/distance weighting; K controls exploration width
        weights = (1.0 / np.maximum(dists, 1e-8)) ** sharpness

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

        # Find nearest unplayed to the target point, filtered by mood
        candidates = self._find_nearest_unplayed(target, k=NORMAL_HOP_K)
        if self._mood_embs is not None:
            candidates = [(t, d) for t, d in candidates if self._is_in_mood(t)]
            if not candidates:
                # Widen search if mood filter rejected everything
                candidates = self._find_nearest_unplayed(target, k=NORMAL_HOP_K * 3)
                candidates = [(t, d) for t, d in candidates if self._is_in_mood(t)]
        if not candidates:
            return self._pick_normal_hop(emb_curr)

        # Pick from candidates with artist penalty
        recent_artists = self._recent_artists(3)
        tids = [c[0] for c in candidates]
        dists = np.array([c[1] for c in candidates])
        weights = 1.0 / np.maximum(dists, 1e-8)
        for i, tid in enumerate(tids):
            r = self._db.get(tid)
            if r and r["artist"] and r["artist"].lower() in recent_artists:
                weights[i] *= 0.1
        weights /= weights.sum()
        chosen = self._rng.choice(len(tids), p=weights)

        logger.info(
            "LONG HOP: trajectory %s → %s, extrapolated %.2f × %.4f = %.4f",
            self._track_name(self._history[-2])[:30],
            self._track_name(self._history[-1])[:30],
            self.hop_multiplier, dist, jump_dist,
        )
        return tids[chosen]

    def _get_active_device(self) -> str | None:
        """Find an active Spotify device to play on."""
        try:
            devices = self._sp.devices()
            for d in devices.get("devices", []):
                if d.get("is_active"):
                    return d["id"]
            # No active device — try the first available
            for d in devices.get("devices", []):
                return d["id"]
        except Exception:
            pass
        return None

    def _play_track(self, tid: str, via_queue: bool = False) -> bool:
        """Start playing a track on Spotify. Returns True on success."""
        if not via_queue:
            uri = f"spotify:track:{tid}"
            try:
                self._sp.start_playback(uris=[uri])
            except Exception:
                # Might fail if no active device — find one and retry
                device = self._get_active_device()
                if device:
                    try:
                        self._sp.start_playback(uris=[uri], device_id=device)
                    except Exception as e:
                        logger.error("Playback failed for %s: %s", tid, e)
                        return False
                else:
                    logger.error("No Spotify device available for %s", tid)
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
        # Broadcast embedding to map display (works without main engine)
        self._broadcast_to_map(tid)
        return True

    def _broadcast_to_map(self, tid: str) -> None:
        """Send this track's embedding to the map display via UDP.

        Uses a cached noisy version (computed once per track) to avoid the
        exact-match void without jiggling the display.
        """
        if tid != self._map_cached_tid:
            emb = self._get_embedding(tid)
            if emb is None:
                logger.debug("No embedding for %s — skip broadcast", tid)
                return
            # Add noise once so the query isn't an exact self-match
            noisy = emb.astype(np.float64) + self._rng.normal(size=512) * 0.04
            self._map_cached_emb = (
                noisy / (np.linalg.norm(noisy) + 1e-8)
            ).astype(np.float32)
            self._map_cached_tid = tid
        try:
            data = pickle.dumps({
                "raw": self._map_cached_emb, "top1": tid, "d_min_raw": 0.05,
            })
            self._map_sock.sendto(data, ("127.0.0.1", self._map_port))
        except Exception as e:
            logger.warning("Map broadcast failed: %s", e)

    def _get_current_spotify_track(self) -> str | None:
        """Get the currently playing Spotify track ID (None if nothing)."""
        try:
            pb = self._sp.current_playback()
            if pb and pb.get("is_playing") and pb.get("item"):
                return pb["item"]["id"]
            else:
                pass  # normal between-track gap
        except Exception as e:
            logger.warning("Could not get Spotify playback: %s", e)
        return None

    def _get_remaining_ms(self) -> int:
        """Get remaining playback time in ms. Returns 0 if paused or unknown."""
        try:
            pb = self._sp.current_playback()
            if pb and pb.get("item"):
                if not pb.get("is_playing"):
                    return 0  # paused — treat as ended
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
        # Seed from argument, current Spotify track, or nearest via live embedding
        if seed_track_id is None:
            seed_track_id = self._get_current_spotify_track()
        if seed_track_id is not None and seed_track_id in self._id_to_idx:
            # Known track — continue without interrupting
            self._played.add(seed_track_id)
            self._history.append(seed_track_id)
            logger.info("Continuing from: %s", self._track_name(seed_track_id))
        elif seed_track_id is not None:
            # Playing an unknown track — let it finish, then take control.
            # Use a dummy history entry so the main loop knows we're waiting.
            logger.info("Current track not in library — will take control when it ends")
            self._history.append(seed_track_id)  # unknown ID as placeholder
            self._played.add(seed_track_id)
        else:
            # Nothing playing
            seed_track_id = self._id_list[self._rng.integers(0, self._N)]
            self._play_track(seed_track_id)

        self._last_hop_time = time.monotonic()

        try:
            while True:
                time.sleep(5)  # poll Spotify API every 5 seconds

                # Keep the map display updated with current track's embedding
                # (works as standalone visualization when no engine is running)
                if self._history:
                    last = self._history[-1]
                    if last in self._id_to_idx:
                        self._broadcast_to_map(last)
                    elif self._live_emb is not None:
                        # Unknown track — send the live embedding instead
                        try:
                            data = pickle.dumps({
                                "raw": self._live_emb,
                                "top1": last,
                                "d_min_raw": 0.05,
                            })
                            self._map_sock.sendto(
                                data, ("127.0.0.1", self._map_port)
                            )
                        except Exception:
                            pass

                # Check for map click override
                override = self._check_override()
                if override:
                    self._play_track(override)
                    self._last_hop_time = time.monotonic()
                    continue

                # Check if Spotify track changed (manual override or album advance).
                # If we have a pending pick, ignore the change — we'll override
                # it when the current track ends. Only react if the user clearly
                # took control (changed to a known library track while we had
                # no pending pick).
                current = self._get_current_spotify_track()
                if (current and current != self._history[-1]
                        and self._next_pick is None):
                    if current in self._id_to_idx:
                        logger.info(
                            "Manual change: %s", self._track_name(current),
                        )
                        self._played.add(current)
                        self._history.append(current)
                        self._last_hop_time = time.monotonic()
                        continue
                    else:
                        logger.info(
                            "Manual change to unknown track — "
                            "will use live embedding at end"
                        )
                        self._history.append(current)
                        self._last_hop_time = time.monotonic()
                        continue

                # Check remaining playback time
                remaining = self._get_remaining_ms()

                # If we already picked the next track, play it when song ends
                if self._next_pick is not None:
                    # Detect: monitored track changed (Spotify advanced) OR
                    # remaining is low — either way, take control now
                    track_changed = (
                        self._monitored_track is not None
                        and current != self._monitored_track
                    )
                    if track_changed or remaining <= 2000:
                        self._play_track(self._next_pick)
                        self._next_pick = None
                        self._monitored_track = None
                    continue

                # Nothing playing or too early to pick
                if remaining == 0 or remaining > 8000:
                    continue

                # Track ending — pick next
                now = time.monotonic()
                time_since_hop = now - self._last_hop_time

                current_emb = self._get_embedding(self._history[-1])

                # Use live audio embedding if: listen mode, OR current track
                # is unknown (not in library — live embedding is our only signal)
                if current_emb is None and self._live_emb is not None:
                    current_emb = self._live_emb
                    logger.info("Using live embedding (current track not in library)")
                elif current_emb is None and self._live_emb is None:
                    logger.debug("No embedding available (no stored, no live)")
                    continue
                elif self._mode == "listen" and self._live_emb is not None:
                    current_emb = self._live_emb

                if current_emb is None:
                    continue

                if time_since_hop >= self.hop_interval:
                    next_tid = self._pick_long_hop()
                    self._last_hop_time = now
                    hop_type = "LONG"
                else:
                    next_tid = self._pick_normal_hop(current_emb)
                    hop_type = "normal"

                if next_tid:
                    self._pending_hop_type = hop_type
                    self._next_pick = next_tid
                    self._monitored_track = current
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
        "--port", type=int, default=57002,
        help="UDP port to receive engine broadcasts (listen mode)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="Exploration temperature: 0 = deterministic (always nearest), "
             "0.1 = tight (nearest 3), 1.0 = wide (top 20). Default: 0.5",
    )
    parser.add_argument(
        "--mood", type=str, default=None,
        help='Mood description, e.g. "chill dinner party, jazz, '
             'downtempo electronic, no opera". Uses Claude API to expand '
             "into genre descriptors, then CLAP to filter candidates.",
    )
    args = parser.parse_args()

    dj = DJ(
        hop_interval_minutes=args.hop_interval,
        hop_multiplier=args.hop_multiplier,
        mode=args.mode,
        udp_port=args.port,
        temperature=args.temperature,
        mood=args.mood,
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
