"""DJ Engine — single-threaded state owner.

Replaces the 940-line ``DJ`` class.  Owns the ``DJState``, runs the
poll loop, dispatches commands to effects, and provides thread-safe
snapshots for the server.

Threading model:
  - Engine thread: runs ``run()``, the ONLY writer of ``_state``.
  - Server thread: reads via ``get_snapshot()`` (lock-protected).
  - UDP listener thread: writes to its own locked field; engine reads
    via ``UDPListener.get_latest()``.
  - Ingestion thread: background ThreadPoolExecutor for on-the-fly
    track ingestion (max 1 worker).
"""

from __future__ import annotations

import logging
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from albart.core.commands import (
    BroadcastToMapCommand,
    Command,
    ComputeLongHopTargetCommand,
    ComputeMoodMaskCommand,
    ComputeTransitTargetCommand,
    FindNeighborsCommand,
    IngestTrackCommand,
    PlayTrackCommand,
    ResumePlaybackCommand,
    UpdateOrbitPositionCommand,
)
from albart.core.navigation import (
    on_neighbors_found,
    on_orbit_transit_arrival,
    on_override,
    on_poll_tick,
    on_track_played,
)
from albart.core.orbit_logic import check_arrival, transit_done
from albart.core.state import DJState, PlaybackSnapshot
from albart.effects.broadcast import BroadcastClient
from albart.effects.database import DatabaseClient
from albart.effects.playback import MetadataClient, PlaybackClient
from albart.effects.udp_listener import UDPListener
from albart.utils import DATA_DIR

logger = logging.getLogger(__name__)

OVERRIDE_PATH = DATA_DIR / "dj_override.txt"


class Engine:
    """Single-threaded DJ engine.

    The server interacts via:
      - ``get_snapshot()`` — thread-safe read of immutable state
      - ``update_param(name, value)`` — thread-safe parameter updates
      - ``enqueue_action(action)`` — thread-safe action queue
    """

    def __init__(
        self,
        db: DatabaseClient,
        playback: PlaybackClient,
        broadcast: BroadcastClient,
        udp_listener: UDPListener,
        *,
        mode: str = "exact",
        song_k: int = 10,
        hop_multiplier: float = 5.0,
        hop_interval_minutes: float = 30.0,
        projector=None,  # UmapProjector | None
        norm_target: float = 0.12,
    ) -> None:
        self._db = db
        self._playback = playback
        self._broadcast = broadcast
        self._udp = udp_listener
        self._projector = projector
        self._norm_target = norm_target

        self._state = DJState(
            mode=mode,
            song_k=song_k,
            hop_multiplier=hop_multiplier,
            hop_interval_seconds=hop_interval_minutes * 60.0,
            total_tracks=db.get_total_tracks(),
        )
        self._lock = threading.Lock()
        self._rng = np.random.default_rng()

        # Derived data (not in state — cached here)
        self._mood_mask: Optional[frozenset[str]] = None
        self._track_artists: dict[str, str] = {}
        self._target_embeddings: dict[str, np.ndarray] = {}

        # Action queue for server → engine communication
        self._action_queue: list[Callable[[DJState], DJState]] = []
        self._action_lock = threading.Lock()

        # Background ingestion of unknown tracks
        self._ingestion_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="Ingest",
        )
        self._ingestion_futures: dict[str, Future] = {}
        self._clap_model = None  # lazy-loaded on first ingestion

    # ── Thread-safe interface for the server ──────────────────────────

    def get_snapshot(self) -> DJState:
        """Return a read-only snapshot of the current state."""
        with self._lock:
            return self._state

    def get_mood_in_count(self) -> int:
        """Number of tracks passing the mood filter."""
        if self._mood_mask is not None:
            return len(self._mood_mask)
        return self._state.total_tracks

    def update_param(self, name: str, value: Any) -> None:
        """Thread-safe parameter update from the server."""
        with self._lock:
            self._state = self._state.model_copy(update={name: value})

    def enqueue_action(self, action: Callable[[DJState], DJState]) -> None:
        """Enqueue a state-modifying action from the server thread."""
        with self._action_lock:
            self._action_queue.append(action)

    def request_stop(self) -> None:
        """Signal the engine to stop."""
        with self._lock:
            self._state = self._state.model_copy(update={"stop_requested": True})

    def skip(self) -> Optional[str]:
        """Immediately pick the next track and play it.  Called from server thread.

        Returns the track_id played, or None on failure.
        """
        with self._lock:
            state = self._state
        if not state.history:
            return None

        now = time.monotonic()
        last_tid = state.history[-1]

        # In orbit transit, skip should play the already-queued next_pick
        # if one exists (the engine eagerly picks in orbit mode). This avoids
        # double-consuming transit steps.
        if state.orbit is not None and state.orbit.phase.value == "transit" and state.next_pick:
            next_tid = state.next_pick
            state = state.model_copy(update={
                "next_pick": None,
                "monitored_track": None,
            })
            success = self._playback.play_track(next_tid)
            if not success:
                return None
            result = on_track_played(state, next_tid, now)
            state = result.state
            state = self._execute_commands(state, result.commands, None, now)
            state = self._check_orbit_arrival(state, next_tid, now)
            with self._lock:
                self._state = state
            track = self._db.get_track(next_tid)
            name = f"{track.title} — {track.artist}" if track else next_tid
            logger.info("▶  SKIP → %s  [played: %d / %d]", name, len(state.played), state.total_tracks)
            return next_tid

        # Pick a new track
        anchor_ids = frozenset(
            a.track_id for a in state.orbit.anchors
        ) if state.orbit else frozenset()
        mood_ids = self._mood_mask if state.mood.descriptors else None

        if state.orbit is not None and state.orbit.phase.value == "dwell":
            # Orbit dwell: 25D neighbor around anchor (exclude anchor tracks)
            anchor_tid = state.orbit.anchors[state.orbit.current_index].track_id
            target = self._db.get_embedding_25d(anchor_tid)
            if target is not None:
                exclude = state.played | anchor_ids
                candidates = self._db.find_neighbors_25d(
                    target, state.song_k, exclude, mood_ids,
                )
            else:
                candidates = []
        else:
            # Normal hop: 25D neighbor from current track
            target = self._db.get_embedding_25d(last_tid)
            if target is not None:
                candidates = self._db.find_neighbors_25d(
                    target, state.song_k, state.played,
                    self._mood_mask,
                )
            else:
                candidates = []

        if not candidates:
            return None

        track_artists = self._db.get_artists_for_tracks([c[0] for c in candidates])
        result = on_neighbors_found(
            state, candidates, track_artists, "normal", now, self._rng,
        )
        next_tid = result.state.next_pick
        if next_tid is None:
            return None

        # Play immediately
        success = self._playback.play_track(next_tid)
        if not success:
            return None

        result2 = on_track_played(result.state, next_tid, now)
        state = result2.state
        state = self._execute_commands(state, result2.commands, None, now)
        state = self._check_orbit_arrival(state, next_tid, now)

        with self._lock:
            self._state = state

        track = self._db.get_track(next_tid)
        name = f"{track.title} — {track.artist}" if track else next_tid
        logger.info("▶  SKIP → %s  [played: %d / %d]", name, len(state.played), state.total_tracks)
        return next_tid

    def new_set(self) -> Optional[str]:
        """Force a long hop — pick and play immediately.  Called from server thread.

        Returns the track_id played, or None on failure.
        """
        with self._lock:
            state = self._state
        if not state.history:
            return None

        now = time.monotonic()

        # For orbit mode with force_transit, start transit then pick
        if state.orbit is not None:
            from albart.core.orbit_logic import start_transit
            last_tid = state.history[-1]
            next_anchor_idx = (state.orbit.current_index + 1) % len(state.orbit.anchors)
            anchor_tid = state.orbit.anchors[next_anchor_idx].track_id
            initial_dist = self._db.compute_initial_transit_distance(last_tid, anchor_tid)
            orbit = start_transit(state.orbit, initial_distance=initial_dist, start_tid=last_tid)
            state = state.model_copy(update={
                "orbit": orbit,
                "pending_hop_type": "ORBIT TRANSIT",
            })
            target = self._db.compute_transit_target(last_tid, anchor_tid, 0.1)
            if target is not None:
                candidates = self._db.find_neighbors_25d(
                    target, state.song_k, state.played,
                    self._mood_mask,
                )
            else:
                candidates = []
        elif len(state.history) >= 2:
            # Long hop: extrapolate trajectory in 25D
            target = self._db.compute_long_hop_target(
                state.history[-2], state.history[-1],
                state.hop_multiplier, self._rng,
            )
            state = state.model_copy(update={
                "pending_hop_type": "NEW SET",
                "last_hop_time": now,
            })
            if target is not None:
                candidates = self._db.find_neighbors_25d(
                    target, 20, state.played,
                    self._mood_mask,
                )
            else:
                candidates = []
        else:
            # Not enough history for long hop — do a normal hop
            target = self._db.get_embedding_25d(state.history[-1])
            state = state.model_copy(update={"pending_hop_type": "NEW SET"})
            if target is not None:
                candidates = self._db.find_neighbors_25d(
                    target, state.song_k, state.played,
                    self._mood_mask,
                )
            else:
                candidates = []

        if not candidates:
            return None

        track_artists = self._db.get_artists_for_tracks([c[0] for c in candidates])
        result = on_neighbors_found(
            state, candidates, track_artists, "NEW SET", now, self._rng,
        )
        next_tid = result.state.next_pick
        if next_tid is None:
            return None

        success = self._playback.play_track(next_tid)
        if not success:
            return None

        result2 = on_track_played(result.state, next_tid, now)
        state = result2.state
        state = self._execute_commands(state, result2.commands, None, now)
        state = self._check_orbit_arrival(state, next_tid, now)

        with self._lock:
            self._state = state

        track = self._db.get_track(next_tid)
        name = f"{track.title} — {track.artist}" if track else next_tid
        logger.info("▶  NEW SET → %s  [played: %d / %d]", name, len(state.played), state.total_tracks)
        return next_tid

    @property
    def db(self) -> DatabaseClient:
        """Expose database client for server queries."""
        return self._db

    @property
    def playback(self) -> PlaybackClient:
        """Expose playback client for server-initiated actions."""
        return self._playback

    # ── Main loop ────────────────────────────────────────────────────

    def run(self, seed_track_id: Optional[str] = None) -> None:
        """Main engine loop.  Runs in the DJ thread."""
        self._udp.start()

        # Seed
        state = self._state
        state = self._seed(state, seed_track_id)
        with self._lock:
            self._state = state

        try:
            while True:
                with self._lock:
                    if self._state.stop_requested:
                        break

                time.sleep(5)

                # Re-read state after sleep — skip() and new_set() may
                # have updated it from the server thread during the wait.
                with self._lock:
                    state = self._state

                # Process queued actions from the server
                state = self._drain_actions(state)

                # Gather external inputs
                playback = self._playback.poll_playback()
                live_emb, _ = self._udp.get_latest()
                override = self._check_override(state)
                now = time.monotonic()

                # Call pure logic
                result = on_poll_tick(
                    state,
                    playback,
                    now,
                    override_tid=override,
                    has_live_emb=live_emb is not None,
                )

                # Execute commands
                state = result.state
                state = self._execute_commands(state, result.commands, live_emb, now)

                # Patch transit initial distance if pure logic started transit
                # with a placeholder (it can't call the database)
                state = self._patch_transit_distance(state)

                # In roomear mode, broadcast live embedding every tick
                if state.mode == "roomear" and live_emb is not None and state.history:
                    self._broadcast.broadcast_raw(live_emb, state.history[-1])

                # Check if current track is unknown → trigger ingestion
                state = self._maybe_ingest_unknown(state)

                # Check completed ingestion futures
                state = self._check_ingestion_completions(state)

                if result.log_message:
                    logger.info(result.log_message)

                # Publish state
                with self._lock:
                    self._state = state

        except KeyboardInterrupt:
            logger.info("Engine stopped. Played %d tracks.", len(state.played))

    # ── Seeding ──────────────────────────────────────────────────────

    def _seed(self, state: DJState, seed_track_id: Optional[str]) -> DJState:
        """Initialize the first track."""
        if seed_track_id is None:
            # Check what Spotify is currently playing
            pb = self._playback.poll_playback()
            if pb.current_track_id and pb.is_playing:
                seed_track_id = pb.current_track_id

        if seed_track_id is not None:
            emb = self._db.get_embedding_25d(seed_track_id)
            if emb is not None:
                # Known track — continue without interrupting
                logger.info("Continuing from: %s", seed_track_id)
                state = state.model_copy(update={
                    "played": state.played | {seed_track_id},
                    "history": (*state.history, seed_track_id),
                    "last_hop_time": time.monotonic(),
                })
                return state
            else:
                # Unknown track — ingest in background, let it finish playing
                logger.info("Current track not in library — ingesting in background")
                state = state.model_copy(update={
                    "history": (*state.history, seed_track_id),
                    "played": state.played | {seed_track_id},
                    "last_hop_time": time.monotonic(),
                })
                self._start_ingestion(state, seed_track_id)
                return state

        # Nothing playing — try to pick a random track, but don't fail
        # if no Spotify device is available (e.g. after sleep/wake).
        # The engine loop will keep polling and the user can start
        # manually from the UI.
        with self._db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT track_id FROM tracks WHERE umap_25d IS NOT NULL "
                    "ORDER BY random() LIMIT 1"
                )
                row = cur.fetchone()
                if row:
                    seed_track_id = row[0]
                    success = self._playback.play_track(seed_track_id)
                    if success:
                        state = state.model_copy(update={
                            "played": state.played | {seed_track_id},
                            "history": (*state.history, seed_track_id),
                            "last_hop_time": time.monotonic(),
                        })
                    else:
                        logger.info("No Spotify device available — waiting for user to start")
        return state

    # ── Command execution ────────────────────────────────────────────

    def _execute_commands(
        self,
        state: DJState,
        commands: list[Command],
        live_emb: Optional[np.ndarray],
        now: float,
    ) -> DJState:
        """Execute commands and feed results back into logic."""
        for cmd in commands:
            if isinstance(cmd, PlayTrackCommand):
                success = self._playback.play_track(cmd.track_id)
                if success:
                    result = on_track_played(state, cmd.track_id, now)
                    state = result.state

                    # Log
                    track = self._db.get_track(cmd.track_id)
                    name = f"{track.title} — {track.artist}" if track else cmd.track_id
                    if result.log_message:
                        logger.info(
                            "▶  %s  [played: %d / %d] ═══ %s",
                            name, len(state.played), state.total_tracks, result.log_message,
                        )
                    else:
                        logger.info(
                            "▶  %s  [played: %d / %d]",
                            name, len(state.played), state.total_tracks,
                        )

                    # Execute sub-commands (broadcast etc.)
                    state = self._execute_commands(state, result.commands, live_emb, now)

                    # Check orbit transit arrival
                    state = self._check_orbit_arrival(state, cmd.track_id, now)

            elif isinstance(cmd, FindNeighborsCommand):
                state = self._execute_find_neighbors(state, cmd, live_emb, now)

            elif isinstance(cmd, ComputeTransitTargetCommand):
                state = self._execute_transit_target(state, cmd, live_emb, now)

            elif isinstance(cmd, ComputeLongHopTargetCommand):
                state = self._execute_long_hop(state, cmd, live_emb, now)

            elif isinstance(cmd, BroadcastToMapCommand):
                last_tid = state.history[-1] if state.history else None
                if last_tid:
                    if state.mode == "roomear" and live_emb is not None:
                        # In roomear mode, always send live audio embedding
                        self._broadcast.broadcast_raw(live_emb, last_tid)
                    else:
                        known = self._db.get_embedding_25d(last_tid) is not None
                        if known:
                            self._broadcast.broadcast_track(cmd.track_id, self._db)
                        else:
                            # Unknown track: broadcast metadata immediately.
                            # Use live audio embedding if available, otherwise
                            # use the previous track's stored embedding so
                            # MapView holds position while showing the new label.
                            title, artist = self._get_track_display(last_tid)
                            if live_emb is not None:
                                self._broadcast.broadcast_raw(
                                    live_emb, last_tid,
                                    title=title, artist=artist,
                                )
                            else:
                                prev_tid = self._last_known_broadcast_tid(state)
                                if prev_tid:
                                    emb = self._db.get_embedding_512(prev_tid)
                                    if emb is not None:
                                        self._broadcast.broadcast_raw(
                                            emb, last_tid,
                                            title=title, artist=artist,
                                        )

            elif isinstance(cmd, ResumePlaybackCommand):
                self._playback.resume()
                logger.info("Resumed paused playback")

            elif isinstance(cmd, ComputeMoodMaskCommand):
                self._execute_mood_mask(cmd)

            elif isinstance(cmd, UpdateOrbitPositionCommand):
                pass  # Progress tracked via state.orbit.last_played_tid

            elif isinstance(cmd, IngestTrackCommand):
                state = self._start_ingestion(state, cmd.track_id)

        return state

    def _execute_find_neighbors(
        self,
        state: DJState,
        cmd: FindNeighborsCommand,
        live_emb: Optional[np.ndarray],
        now: float,
    ) -> DJState:
        """Execute a neighbor search and feed results to logic."""
        # Resolve the query embedding
        if cmd.use_target_embedding and cmd.use_target_embedding in self._target_embeddings:
            target = self._target_embeddings.pop(cmd.use_target_embedding)
        elif cmd.target_track_id:
            target = self._db.get_embedding_25d(cmd.target_track_id)
            if target is None:
                logger.debug("No 25D embedding for %s", cmd.target_track_id)
                # Roomear fallback: project live embedding to 25D
                if state.mode == "roomear" and live_emb is not None and self._projector is not None:
                    target = self._projector.project(live_emb)
                    logger.info("Using roomear live embedding (projected to 25D)")
        else:
            target = None

        # Roomear mode: if no stored embedding, use live audio
        if target is None and state.mode == "roomear" and live_emb is not None and self._projector is not None:
            target = self._projector.project(live_emb)
            logger.info("Using roomear live embedding for neighbor search")

        if target is None:
            logger.debug("No embedding available for neighbor search")
            return state

        # All navigation uses 25D
        mood_ids = self._mood_mask if cmd.mood_mask_active else None
        candidates = self._db.find_neighbors_25d(
            target, cmd.k, cmd.exclude_played, mood_ids,
        )

        # Get artist info for penalty — include recent history tracks
        # so the penalty can look back further than just the candidates
        cand_ids = [c[0] for c in candidates]
        recent_ids = list(state.history[-8:])
        track_artists = self._db.get_artists_for_tracks(cand_ids + recent_ids)

        # Feed back to logic
        result = on_neighbors_found(
            state, candidates, track_artists, cmd.hop_type, now, self._rng,
        )
        state = result.state
        if result.log_message:
            logger.info(result.log_message)
        return self._execute_commands(state, result.commands, live_emb, now)

    def _execute_transit_target(
        self,
        state: DJState,
        cmd: ComputeTransitTargetCommand,
        live_emb: Optional[np.ndarray],
        now: float,
    ) -> DJState:
        """Compute transit interpolation point and issue neighbor search.

        Interpolates from the CURRENT track (wherever we actually are)
        toward the target anchor by a fraction proportional to the step.
        This creates varied paths — each transit finds a different route
        through the space rather than walking a fixed line.
        """
        origin_tid = cmd.current_track_id
        total = state.orbit.transit_total if state.orbit else 10
        # Fraction of remaining distance to cover in this step.
        # With N steps remaining, move 1/N of the way to the anchor.
        # This converges on the anchor regardless of where we wander.
        fraction = 1.0 / max(cmd.transit_remaining, 1)
        target = self._db.compute_transit_target(
            origin_tid, cmd.target_anchor_track_id, fraction,
        )
        if target is None:
            return state

        # Log transit step details (in 25D)
        anchor_emb = self._db.get_embedding_25d(cmd.target_anchor_track_id)
        origin_emb = self._db.get_embedding_25d(origin_tid)
        if anchor_emb is not None and origin_emb is not None:
            dist_origin_to_anchor = float(np.linalg.norm(
                anchor_emb.astype(np.float64) - origin_emb.astype(np.float64)
            ))
            dist_target_to_anchor = float(np.linalg.norm(
                anchor_emb.astype(np.float64) - target.astype(np.float64)
            ))
            logger.info(
                "Transit step: remaining=%d frac=%.3f  "
                "origin→anchor=%.4f  target_point→anchor=%.4f",
                cmd.transit_remaining, fraction,
                dist_origin_to_anchor, dist_target_to_anchor,
            )

        # Exclude all anchor tracks from transit candidates
        anchor_ids = frozenset(
            a.track_id for a in state.orbit.anchors
        ) if state.orbit else frozenset()
        exclude = state.played | anchor_ids

        # Query neighbors in 25D, excluding anchors
        candidates = self._db.find_neighbors_25d(
            target, state.song_k, exclude,
            self._mood_mask if state.mood.descriptors else None,
        )
        if not candidates:
            return state

        cand_ids_t = [c[0] for c in candidates]
        recent_ids_t = list(state.history[-8:])
        track_artists = self._db.get_artists_for_tracks(cand_ids_t + recent_ids_t)
        result = on_neighbors_found(
            state, candidates, track_artists, "orbit_transit", now, self._rng,
        )
        state = result.state
        if result.log_message:
            logger.info(result.log_message)
        return self._execute_commands(state, result.commands, live_emb, now)

    def _execute_long_hop(
        self,
        state: DJState,
        cmd: ComputeLongHopTargetCommand,
        live_emb: Optional[np.ndarray],
        now: float,
    ) -> DJState:
        """Compute long-hop target and issue neighbor search."""
        target = self._db.compute_long_hop_target(
            cmd.prev_track_id, cmd.current_track_id,
            cmd.hop_multiplier, self._rng,
        )
        if target is None:
            return state

        self._target_embeddings[cmd.result_key] = target

        from albart.core.neighbor import build_neighbor_query
        query = build_neighbor_query(
            state,
            use_target_embedding=cmd.result_key,
            space="25d",
            k=20,  # wider search for long hops
            hop_type="NEW SET",
        )
        return self._execute_commands(state, [query], live_emb, now)

    def _execute_mood_mask(self, cmd: ComputeMoodMaskCommand) -> None:
        """Compute and cache the mood mask.

        Descriptors are cluster labels (from the interpret step).
        Positive labels are included, NOT:-prefixed labels are excluded.
        Matches labels to cluster IDs and computes the distance-based mask.
        Falls back to CLAP text embeddings if clusters.json is missing.
        """
        if not cmd.descriptors:
            self._mood_mask = None
            logger.info("Mood mask cleared")
            return

        from albart.core.mood import parse_mood_descriptors
        positive, negative = parse_mood_descriptors(list(cmd.descriptors))

        # Try cluster-based filtering
        clusters_path = DATA_DIR / "clusters.json"
        if clusters_path.exists():
            import json
            data = json.loads(clusters_path.read_text())
            labels = data["labels"]
            centroids = np.array(data["centroids"], dtype=np.float32)
            mean_radii = np.array(data["mean_radii"], dtype=np.float32)

            # Match descriptor strings to cluster IDs
            label_to_id = {label.lower(): i for i, label in enumerate(labels)}
            pos_ids = [label_to_id[d.lower()] for d in positive if d.lower() in label_to_id]
            neg_ids = [label_to_id[d.lower()] for d in negative if d.lower() in label_to_id]

            if pos_ids or neg_ids:
                pos_labels = [labels[i] for i in pos_ids]
                neg_labels = [labels[i] for i in neg_ids]
                logger.info(
                    "Cluster mood: +[%s] -[%s]",
                    ", ".join(pos_labels), ", ".join(neg_labels),
                )
                self._mood_mask = self._db.compute_cluster_mood_mask(
                    pos_ids, neg_ids, centroids, mean_radii,
                )
                return
            else:
                logger.warning("No descriptors matched cluster labels")

        # Fallback: legacy CLAP text embedding approach
        logger.info("Falling back to CLAP text mood filter")
        pos_embs = None
        neg_embs = None
        if positive or negative:
            from albart.text_embedder import embed_texts
            if positive:
                pos_embs = embed_texts(positive)
            if negative:
                neg_embs = embed_texts(negative)

        self._mood_mask = self._db.compute_mood_mask(pos_embs, neg_embs, cmd.threshold)

    def _patch_transit_distance(self, state: DJState) -> DJState:
        """If orbit just entered transit with placeholder distance, compute the real one."""
        orbit = state.orbit
        if orbit is None or orbit.phase.value != "transit":
            return state
        if orbit.transit_initial_dist != 1.0:
            return state  # already computed
        if not state.history:
            return state
        last_tid = state.history[-1]
        target_tid = orbit.anchors[orbit.current_index].track_id
        real_dist = self._db.compute_initial_transit_distance(last_tid, target_tid)
        if real_dist == 1.0:
            return state  # couldn't compute
        updates = {"transit_initial_dist": real_dist}
        if not orbit.transit_start_tid:
            updates["transit_start_tid"] = last_tid
        new_orbit = orbit.model_copy(update=updates)
        return state.model_copy(update={"orbit": new_orbit})

    def _check_orbit_arrival(
        self, state: DJState, track_id: str, now: float,
    ) -> DJState:
        """Check if a transit track has arrived at the target anchor."""
        orbit = state.orbit
        if orbit is None or orbit.phase.value != "transit":
            return state

        track_emb = self._db.get_embedding_25d(track_id)
        if track_emb is None:
            return state

        target_tid = orbit.anchors[orbit.current_index].track_id
        target_emb = self._db.get_embedding_25d(target_tid)
        if target_emb is None:
            return state

        dist = float(np.linalg.norm(
            target_emb.astype(np.float64) - track_emb.astype(np.float64)
        ))

        arrived = check_arrival(orbit, dist) or transit_done(orbit)
        if arrived and not orbit.arrived:
            # Mark as arrived but stay in transit phase — the actual
            # transition to dwell happens when the next pick cycle runs
            # (i.e., when this arriving track is about to end).
            # This prevents the UI from showing "Dwelling" while the
            # last transit track is still playing.
            new_orbit = orbit.model_copy(update={"arrived": True})
            state = state.model_copy(update={"orbit": new_orbit})
            logger.info(
                "Orbit transit arrived at [%d] (dist=%.4f, steps used=%d/%d) — "
                "will transition to dwell when track ends",
                orbit.current_index,
                dist,
                orbit.transit_total - orbit.transit_remaining,
                orbit.transit_total,
            )
        return state

    # ── Track ingestion ────────────────────────────────────────────

    def _get_clap_model(self) -> tuple:
        """Lazy-load CLAP model for ingestion (heavy, ~300MB)."""
        if self._clap_model is None:
            from albart.pipeline.embedder import load_model
            model, processor, device = load_model(allow_mps=True)
            self._clap_model = (model, processor, device)
            logger.info("CLAP model loaded for ingestion (device=%s)", device)
        return self._clap_model

    def _start_ingestion(self, state: DJState, track_id: str) -> DJState:
        """Launch background ingestion if not already running for this track."""
        if track_id in self._ingestion_futures:
            return state
        if track_id in state.ingesting_tracks:
            return state
        # Skip if already in database with embedding
        if self._db.get_embedding_25d(track_id) is not None:
            return state

        from albart.effects.ingest import ingest_track

        clap_model, clap_processor, clap_device = self._get_clap_model()
        # Pass the raw Spotify client for ingestion (needs sp.track() for
        # metadata + preview URL).  Non-Spotify backends would implement
        # their own ingestion pipeline.
        sp = getattr(self._playback, "sp", None)
        if sp is None:
            logger.warning("Ingestion not supported for this playback backend")
            return state
        future = self._ingestion_executor.submit(
            ingest_track,
            track_id,
            sp,
            self._db,
            self._projector,
            clap_model,
            clap_processor,
            clap_device,
            self._norm_target,
        )
        self._ingestion_futures[track_id] = future
        state = state.model_copy(update={
            "ingesting_tracks": state.ingesting_tracks | {track_id},
        })
        logger.info("Started background ingestion for %s", track_id)
        return state

    def _check_ingestion_completions(self, state: DJState) -> DJState:
        """Check for completed ingestion futures and update state."""
        completed = []
        for tid, future in self._ingestion_futures.items():
            if future.done():
                try:
                    success = future.result()
                    if success:
                        logger.info("Ingestion complete: %s", tid)
                        state = state.model_copy(update={
                            "total_tracks": self._db.get_total_tracks(),
                        })
                    else:
                        logger.warning("Ingestion returned False for %s", tid)
                except Exception as e:
                    logger.error("Ingestion failed for %s: %s", tid, e)
                completed.append(tid)
        for tid in completed:
            del self._ingestion_futures[tid]
        if completed:
            remaining = state.ingesting_tracks - set(completed)
            state = state.model_copy(update={"ingesting_tracks": remaining})
        return state

    def _maybe_ingest_unknown(self, state: DJState) -> DJState:
        """If the current Spotify track is unknown, trigger ingestion."""
        pb = state.playback
        if not pb.current_track_id or not pb.is_playing:
            return state
        tid = pb.current_track_id
        if tid in state.ingesting_tracks:
            return state
        if tid in self._ingestion_futures:
            return state
        if self._db.get_embedding_25d(tid) is not None:
            return state
        return self._start_ingestion(state, tid)

    # ── Helpers ──────────────────────────────────────────────────────

    def _last_known_broadcast_tid(self, state: DJState) -> Optional[str]:
        """Find the most recent track in history that has a stored embedding."""
        for tid in reversed(state.history):
            if self._db.get_embedding_512(tid) is not None:
                return tid
        return None

    def _get_track_display(self, track_id: str) -> tuple[str | None, str | None]:
        """Get display title/artist for a track, from cache/DB/Spotify API."""
        if not hasattr(self, "_display_cache"):
            self._display_cache: dict[str, tuple[str | None, str | None]] = {}
        if track_id in self._display_cache:
            return self._display_cache[track_id]
        # Try database first (fast, works for known tracks)
        track = self._db.get_track(track_id)
        if track:
            result = (track.title, track.artist)
            self._display_cache[track_id] = result
            return result
        # Try playback client's metadata lookup (if supported)
        if hasattr(self._playback, "get_track_metadata"):
            meta = self._playback.get_track_metadata(track_id)
            if meta is not None:
                self._display_cache[track_id] = meta
                return meta
        return None, None

    def _check_override(self, state: DJState) -> Optional[str]:
        """Check for map click override file."""
        if OVERRIDE_PATH.exists():
            try:
                tid = OVERRIDE_PATH.read_text().strip()
                OVERRIDE_PATH.unlink()
                if tid:
                    emb = self._db.get_embedding_512(tid)
                    if emb is not None:
                        return tid
            except Exception:
                pass
        return None

    def _drain_actions(self, state: DJState) -> DJState:
        """Process queued actions from the server."""
        with self._action_lock:
            actions = list(self._action_queue)
            self._action_queue.clear()
        for action in actions:
            state = action(state)
        with self._lock:
            self._state = state
        return state
