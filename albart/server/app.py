"""ALBart Control Center — FastAPI thin adapter over the Engine.

Every read goes through ``engine.get_snapshot()`` (immutable state).
Every mutation goes through ``engine.update_param()`` or ``engine.enqueue_action()``.
No direct field access, no race conditions by design.

Usage:
    python -m albart.dj_server          # start via the existing entry point
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Optional

# Configure logging before anything else — uvicorn's reload mode
# overrides the root logger, so we set our loggers explicitly.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Ensure our loggers aren't suppressed by uvicorn
for _name in ("albart", "albart.engine", "albart.server", "albart.effects"):
    logging.getLogger(_name).setLevel(logging.INFO)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from albart.core.mood import clear_mood, parse_mood_descriptors, update_mood
from albart.core.navigation import on_override
from albart.core.orbit_logic import (
    find_track,
    get_progress,
    normalize_search,
)
from albart.core.state import (
    DJState,
    MoodState,
    OrbitAnchorState,
    OrbitPhase,
    OrbitState,
)
from albart.effects.broadcast import BroadcastClient
from albart.effects.database import DatabaseClient, DatabaseConfig
from albart.effects.spotify import SpotifyClient
from albart.effects.udp_listener import UDPListener
from albart.engine import Engine
from albart.server.models import (
    DescriptorsUpdate,
    DeviceInfo,
    MoodThresholdUpdate,
    MoodUpdate,
    OrbitAnchorInfo,
    OrbitDescriptorsUpdate,
    OrbitProgress,
    OrbitRequest,
    ProcessStatus,
    SeekRequest,
    SetDistanceUpdate,
    SongKUpdate,
    StartRequest,
    StatusResponse,
    SystemStatus,
    TrackInfo,
    VolumeUpdate,
)
from albart.utils import DATA_DIR, load_config

logger = logging.getLogger("albart.server")


def _load_projector():
    """Load parametric UMAP projector if model exists."""
    config = load_config()
    model_path = config.get("umap_25d", {}).get("model_path", "data/umap_25d_model/model.pt")
    full_path = DATA_DIR.parent / model_path
    if full_path.exists():
        from albart.effects.umap_projector import UmapProjector
        return UmapProjector.load(full_path)
    return None


def _get_norm_target() -> float:
    config = load_config()
    return float(config.get("pipeline", {}).get("norm_target", 0.12))

app = FastAPI(title="ALBart DJ", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global engine state ──────────────────────────────────────────────

_engine: Optional[Engine] = None
_engine_thread: Optional[threading.Thread] = None
_engine_lock = threading.Lock()

_mapview_proc: Optional[subprocess.Popen] = None
_listener_proc: Optional[subprocess.Popen] = None


# ── Helpers ──────────────────────────────────────────────────────────

def _track_info(db: DatabaseClient, tid: str, set_starts: tuple = ()) -> TrackInfo:
    track = db.get_track(tid)
    # Look up set label: set_starts is ((track_id, label), ...)
    label = None
    for start_tid, start_label in set_starts:
        if start_tid == tid:
            label = start_label
            break
    return TrackInfo(
        track_id=tid,
        title=track.title if track else "?",
        artist=track.artist if track else "?",
        set_start=label,
    )


def _run_engine(engine: Engine, seed: Optional[str]) -> None:
    try:
        engine.run(seed_track_id=seed)
    except Exception as e:
        logger.error("Engine loop error: %s", e)


# ── Status ───────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status() -> StatusResponse:
    if _engine is None:
        return StatusResponse(
            playing=False, song_k=10, set_distance=5.0,
            mood_descriptors=[], mood_threshold=0.35, mood_in_count=0,
            played_count=0, total_tracks=_get_db().get_total_tracks(), history=[],
        )

    state = _engine.get_snapshot()
    db = _engine.db

    current = None
    if state.history:
        current = _track_info(db, state.history[-1], state.set_starts)

    history = [_track_info(db, tid, state.set_starts) for tid in reversed(state.history)]

    # Progress from cached snapshot
    elapsed = time.monotonic() - state.playback.snapshot_time
    progress_ms = min(
        state.playback.progress_ms + int(elapsed * 1000),
        state.playback.duration_ms,
    )

    # Orbit
    orbit_anchors: list[OrbitAnchorInfo] = []
    orbit_progress = None
    if state.orbit is not None:
        for i, a in enumerate(state.orbit.anchors):
            track = db.get_track(a.track_id)
            orbit_anchors.append(OrbitAnchorInfo(
                description=a.description,
                track_id=a.track_id,
                title=track.title if track else "?",
                artist=track.artist if track else "?",
                art_url=f"/api/art/{a.track_id}",
                active=(i == state.orbit.current_index),
            ))
        # Transit progress from 25D distance (organic, shows space structure)
        seg_progress = 0.0
        orbit = state.orbit
        if orbit.phase.value == "transit" and orbit.last_played_tid:
            last_emb = db.get_embedding_25d(orbit.last_played_tid)
            target_emb = db.get_embedding_25d(orbit.anchors[orbit.current_index].track_id)
            if last_emb is not None and target_emb is not None and orbit.transit_initial_dist > 1e-8:
                import numpy as np
                dist_remaining = float(np.linalg.norm(
                    target_emb.astype(np.float64) - last_emb.astype(np.float64)
                ))
                seg_progress = max(0.0, min(1.0,
                    1.0 - dist_remaining / orbit.transit_initial_dist
                ))
        prog = get_progress(state.orbit, time.monotonic(), segment_progress=seg_progress)
        orbit_progress = OrbitProgress(**prog)

    return StatusResponse(
        playing=_engine_thread is not None and _engine_thread.is_alive(),
        current_track=current,
        progress_ms=progress_ms,
        duration_ms=state.playback.duration_ms,
        song_k=state.song_k,
        set_distance=state.hop_multiplier,
        mode=state.mode,
        dj_active=state.dj_active,
        mood_text=state.mood.mood_text,
        mood_descriptors=list(state.mood.descriptors),
        mood_threshold=state.mood.threshold,
        mood_in_count=_engine.get_mood_in_count(),
        played_count=len(state.played),
        total_tracks=state.total_tracks,
        history=history,
        volume=state.playback.volume,
        orbit_active=state.orbit is not None,
        orbit_anchors=orbit_anchors,
        orbit_progress=orbit_progress,
    )


# ── Session control ──────────────────────────────────────────────────

@app.post("/api/start")
def start_session(req: StartRequest) -> dict:
    global _engine, _engine_thread

    with _engine_lock:
        if _engine is not None:
            _engine.request_stop()

        db = DatabaseClient(config=DatabaseConfig())
        spotify = SpotifyClient.create()
        broadcast = BroadcastClient()
        udp = UDPListener()

        _engine = Engine(
            db=db,
            playback=spotify,
            broadcast=broadcast,
            udp_listener=udp,
            mode=req.mode,
            song_k=max(1, min(50, req.song_k)),
            hop_multiplier=req.set_distance,
            hop_interval_minutes=req.hop_interval,
            projector=_load_projector(),
            norm_target=_get_norm_target(),
        )

        # Handle mood
        if req.mood:
            from albart.core.commands import ComputeMoodMaskCommand
            result = update_mood(
                _engine.get_snapshot(),
                mood_text=req.mood,
                descriptors=[],
            )
            _engine.update_param("mood", result.state.mood)

        # Resolve seed
        seed = None
        if req.seed:
            track = db.get_track(req.seed)
            if track:
                seed = req.seed
            else:
                results = db.search_tracks(req.seed, limit=1)
                seed = results[0].track_id if results else None

        _engine_thread = threading.Thread(
            target=_run_engine, args=(_engine, seed), daemon=True,
        )
        _engine_thread.start()

    return {"status": "started", "seed": seed}


@app.post("/api/stop")
def stop_session() -> dict:
    global _engine, _engine_thread
    if _engine is not None:
        _engine.request_stop()
        _engine.playback.pause()
    _engine = None
    _engine_thread = None
    return {"status": "stopped"}


# ── Parameter updates ────────────────────────────────────────────────

@app.put("/api/song_k")
def set_song_k(req: SongKUpdate) -> dict:
    if _engine is None:
        return {"error": "No active session"}
    val = max(1, min(50, req.song_k))
    _engine.update_param("song_k", val)
    return {"song_k": val}


@app.put("/api/mood_threshold")
def set_mood_threshold(req: MoodThresholdUpdate) -> dict:
    if _engine is None:
        return {"error": "No active session"}
    val = max(0.0, min(0.6, req.threshold))
    state = _engine.get_snapshot()
    result = update_mood(state, threshold=val)
    _engine.update_param("mood", result.state.mood)
    # Execute mood mask recomputation
    for cmd in result.commands:
        _engine._execute_mood_mask(cmd)
    return {"threshold": val}


@app.put("/api/set_distance")
def set_set_distance(req: SetDistanceUpdate) -> dict:
    if _engine is None:
        return {"error": "No active session"}
    val = max(1.0, min(20.0, req.set_distance))
    _engine.update_param("hop_multiplier", val)
    return {"set_distance": val}


@app.put("/api/mode")
def set_mode(req: dict) -> dict:
    """Switch DJ input mode: 'exact' or 'roomear'."""
    if _engine is None:
        return {"error": "No active session"}
    mode = req.get("mode", "exact")
    if mode not in ("exact", "roomear"):
        return {"error": f"Invalid mode: {mode}"}
    _engine.update_param("mode", mode)
    logger.info("Mode switched to: %s", mode)
    return {"mode": mode}


@app.get("/api/mode")
def get_mode() -> dict:
    """Get current DJ input mode and live embedding status."""
    if _engine is None:
        return {"mode": "exact", "live_emb": False, "dj_active": True}
    emb, top1 = _engine._udp.get_latest()
    state = _engine.get_snapshot()
    return {
        "mode": state.mode,
        "dj_active": state.dj_active,
        "live_emb": emb is not None,
        "live_top1": top1,
    }


@app.put("/api/dj_active")
def set_dj_active(req: dict) -> dict:
    """Toggle DJ Control (active) vs Spotify Control (passive)."""
    if _engine is None:
        return {"error": "No active session"}
    active = bool(req.get("dj_active", True))
    _engine.update_param("dj_active", active)
    label = "DJ Control" if active else "Spotify Control"
    logger.info("Switched to %s", label)
    return {"dj_active": active}


# ── Mood ─────────────────────────────────────────────────────────────

@app.post("/api/interpret")
def interpret_mood(req: MoodUpdate) -> dict:
    """Map mood text to cluster include/exclude using Claude.

    If clusters.json exists, Claude selects from the actual cluster labels.
    The returned descriptors are the selected labels (prefixed with NOT:
    for exclusions), which the user can review and edit before applying.
    """
    import anthropic
    import json

    clusters_path = DATA_DIR / "clusters.json"

    try:
        client = anthropic.Anthropic()

        if clusters_path.exists():
            data = json.loads(clusters_path.read_text())
            labels = data["labels"]

            labels_text = "\n".join(f"{i}: {label}" for i, label in enumerate(labels))

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system="You are a JSON API. Return only valid JSON, no commentary.",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Music mood: \"{req.mood}\"\n\n"
                        f"Clusters:\n{labels_text}\n\n"
                        "Which clusters to include and exclude? "
                        "Be selective — only include clusters that clearly fit. "
                        "Exclude clusters that clearly conflict. "
                        "Leave ambiguous clusters out of both lists.\n\n"
                        '{"include": [...], "exclude": [...]}'
                    ),
                }],
            )

            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)

            # Convert to descriptor lines for the UI (user can review/edit)
            lines = []
            for i in result.get("include", []):
                if i < len(labels):
                    lines.append(labels[i])
            for i in result.get("exclude", []):
                if i < len(labels):
                    lines.append(f"NOT: {labels[i]}")

            return {"descriptors": lines, "mood": req.mood}

        else:
            # Fallback: legacy CLAP-style descriptor generation
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": (
                        "I'm setting up a music DJ that should play tracks matching "
                        "a specific mood. Given this description:\n\n"
                        f'"{req.mood}"\n\n'
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
            return {"descriptors": lines, "mood": req.mood}

    except Exception as e:
        return {"error": str(e)}


@app.put("/api/mood")
def apply_mood(req: DescriptorsUpdate) -> dict:
    """Apply mood filter from edited descriptors."""
    if _engine is None:
        return {"error": "No active session — start the DJ first"}
    try:
        lines = [ln.strip() for ln in req.descriptors if ln.strip()]
        state = _engine.get_snapshot()
        result = update_mood(state, descriptors=lines, mood_text="(edited)")
        _engine.update_param("mood", result.state.mood)
        for cmd in result.commands:
            _engine._execute_mood_mask(cmd)
        return {"status": "mood applied", "descriptors": lines}
    except Exception as e:
        return {"error": str(e)}


# ── Playback control ─────────────────────────────────────────────────

@app.put("/api/seek")
def seek_track(req: SeekRequest) -> dict:
    if _engine is None:
        return {"error": "No active session"}
    _engine.playback.seek(req.position_ms)
    return {"position_ms": req.position_ms}


@app.post("/api/new_set")
def new_set() -> dict:
    """Force a long hop — pick and play immediately."""
    if _engine is None:
        return {"error": "No active session"}
    next_tid = _engine.new_set()
    if next_tid:
        track = _engine.db.get_track(next_tid)
        name = f"{track.title} — {track.artist}" if track else next_tid
        return {"status": "new set", "now_playing": name}
    return {"error": "Could not find a track for the long hop"}


@app.post("/api/skip")
def skip_track() -> dict:
    """Skip to the next track immediately."""
    if _engine is None:
        return {"error": "No active session"}
    next_tid = _engine.skip()
    if next_tid:
        track = _engine.db.get_track(next_tid)
        name = f"{track.title} — {track.artist}" if track else next_tid
        return {"status": "skipped", "now_playing": name}
    return {"error": "Could not find next track"}


@app.post("/api/play/{track_id}")
def play_now(track_id: str) -> dict:
    if _engine is None:
        return {"error": "No active session"}
    emb = _engine.db.get_embedding_512(track_id)
    if emb is None:
        return {"error": "Track not in library"}

    def _override(state: DJState) -> DJState:
        result = on_override(state, track_id, time.monotonic())
        return result.state

    _engine.enqueue_action(_override)
    _engine.playback.play_track(track_id)
    track = _engine.db.get_track(track_id)
    name = f"{track.title} — {track.artist}" if track else track_id
    return {"status": "playing", "track": name}


@app.post("/api/queue/{track_id}")
def queue_next(track_id: str) -> dict:
    if _engine is None:
        return {"error": "No active session"}
    emb = _engine.db.get_embedding_512(track_id)
    if emb is None:
        return {"error": "Track not in library"}

    def _queue(state: DJState) -> DJState:
        return state.model_copy(update={
            "next_pick": track_id,
            "pending_hop_type": None,
            "last_hop_time": time.monotonic(),
        })

    _engine.enqueue_action(_queue)
    track = _engine.db.get_track(track_id)
    name = f"{track.title} — {track.artist}" if track else track_id
    return {"status": "queued", "track": name}


# ── Search ───────────────────────────────────────────────────────────

_shared_db: Optional[DatabaseClient] = None

def _get_db() -> DatabaseClient:
    """Get a DatabaseClient — from the engine if running, otherwise standalone."""
    global _shared_db
    if _engine is not None:
        return _engine.db
    if _shared_db is None:
        _shared_db = DatabaseClient(config=DatabaseConfig())
    return _shared_db

@app.get("/api/search")
def search_tracks(q: str, limit: int = 10) -> list[TrackInfo]:
    results = _get_db().search_tracks(q, limit=limit)
    return [TrackInfo(track_id=r.track_id, title=r.title, artist=r.artist) for r in results]


# ── Devices & Volume ─────────────────────────────────────────────────

@app.get("/api/devices")
def get_devices() -> list[DeviceInfo]:
    if _engine is None:
        return []
    # Device listing is player-specific (Spotify has multiple devices)
    pb = _engine.playback
    if not hasattr(pb, "get_devices"):
        return []
    devices = pb.get_devices()
    return [
        DeviceInfo(
            id=d["id"], name=d["name"], type=d["type"],
            is_active=d["is_active"],
            volume=d.get("volume_percent", 0),
        )
        for d in devices
    ]


@app.put("/api/volume")
def set_volume(req: VolumeUpdate) -> dict:
    if _engine is None:
        return {"error": "No active session"}
    _engine.playback.set_volume(req.volume)
    return {"volume": req.volume}


@app.post("/api/device/{device_id}")
def set_device(device_id: str) -> dict:
    if _engine is None:
        return {"error": "No active session"}
    pb = _engine.playback
    if not hasattr(pb, "transfer_playback"):
        return {"error": "Device transfer not supported by this player"}
    pb.transfer_playback(device_id)
    return {"status": "transferred", "device": device_id}


# ── Album art ────────────────────────────────────────────────────────

@app.get("/api/art/{track_id}")
def get_art(track_id: str):
    art_path = DATA_DIR / "art_original" / f"{track_id}.jpg"
    if art_path.exists():
        return FileResponse(art_path, media_type="image/jpeg")
    return {"error": "Art not found"}


# ── Orbit ────────────────────────────────────────────────────────────

@app.post("/api/orbit/interpret")
def interpret_orbit(req: OrbitRequest) -> dict:
    """Send journey description to Claude for anchor suggestions."""
    import anthropic
    from collections import defaultdict

    if _engine is None:
        return {"error": "No active session — start the DJ first"}

    import random

    db = _engine.db
    all_tracks = db.get_all_tracks()

    # Sample up to 5000 tracks to keep the LLM context manageable
    # as the library grows. A random 5000 is always enough to find
    # good anchors for any journey description.
    MAX_TRACKS_FOR_LLM = 5000
    if len(all_tracks) > MAX_TRACKS_FOR_LLM:
        all_tracks = random.sample(all_tracks, MAX_TRACKS_FOR_LLM)

    by_artist: dict[str, list[str]] = defaultdict(list)
    for t in all_tracks:
        by_artist[t.get("artist", "Unknown")].append(t.get("title", "?"))

    lib_lines = []
    for artist in sorted(by_artist.keys()):
        lib_lines.append(f"{artist}: {', '.join(by_artist[artist])}")
    library_text = "\n".join(lib_lines)

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    "I have an AI DJ that navigates my music library through "
                    "high-dimensional audio embedding space. When I give it a "
                    "sequence of anchor tracks, it:\n\n"
                    "1. Dwells near each anchor for ~30 minutes, playing tracks "
                    "from the same genre neighborhood\n"
                    "2. Transits between anchors through intermediate tracks "
                    "that bridge the genre gap\n"
                    "3. Cycles back to the first anchor and repeats\n\n"
                    "The DJ handles all transitions and genre exploration "
                    "automatically. Your job is to pick the anchor waypoints "
                    "— the iconic destinations that define the journey.\n\n"
                    "The user described this journey:\n"
                    f'"{req.description}"\n\n'
                    "Pick 4-8 anchor tracks from my library. Each anchor "
                    "defines a genre neighborhood the DJ will explore, so:\n"
                    "- Choose tracks that are unmistakably representative of "
                    "their genre or era\n"
                    "- Space anchors across distinct musical territories — the "
                    "DJ fills the gaps\n"
                    "- Order them so the sequence tells a story — dramatic "
                    "jumps between very different genres are fine, the DJ is "
                    "skilled at finding surprising connections across any "
                    "genre gap\n"
                    "- The last anchor should connect naturally back to the "
                    "first (it's a cycle)\n"
                    "- Pick more anchors for complex multi-genre journeys, "
                    "fewer for focused ones\n\n"
                    "My library (one artist per line, with their tracks):\n\n"
                    f"{library_text}\n\n"
                    "Format each track as \"Artist \u2014 Title\", copying the "
                    "artist and title exactly from the library above. Reply "
                    "with only a JSON object:\n"
                    "{\"tracks\": [\"Artist \u2014 Title\", \"Artist \u2014 Title\", ...]}"
                ),
            }],
        )
        import json
        import re
        raw = response.content[0].text.strip()
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', raw, re.DOTALL)
        if fence_match:
            raw = fence_match.group(1).strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        try:
            data = json.loads(raw)
            lines = [t.strip() for t in data.get("tracks", []) if t.strip()]
        except json.JSONDecodeError:
            lines = []
            for ln in response.content[0].text.strip().split("\n"):
                ln = ln.strip().rstrip(".")
                ln = re.sub(r'^\d+[.)]\s*', '', ln)
                if ("\u2014" in ln or " -- " in ln or " - " in ln) and len(ln) > 5:
                    lines.append(ln)

        if not lines:
            return {"error": "No tracks found in response"}

        # Ask Claude about same-artist runs
        allow_same_artist = False
        anchors_text = ", ".join(lines[:5])
        try:
            artist_resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4,
                system="Answer only Yes or No.",
                messages=[{
                    "role": "user",
                    "content": (
                        f'A DJ orbit will dwell near these anchor tracks:\n'
                        f'{anchors_text}\n\n'
                        f'Journey: "{req.description}"\n\n'
                        "Should the DJ allow playing multiple tracks by the "
                        "same artist in a row when dwelling near an anchor? "
                        "Say Yes if the journey focuses on specific artists "
                        "or deep dives, No if it emphasizes variety."
                    ),
                }],
            )
            allow_same_artist = artist_resp.content[0].text.strip().lower().startswith("yes")
            logger.info("Orbit allow_same_artist: %s", allow_same_artist)
        except Exception:
            pass

        return {
            "descriptions": lines,
            "journey": req.description,
            "allow_same_artist": allow_same_artist,
        }
    except Exception as e:
        return {"error": str(e)}


@app.put("/api/orbit/apply")
def apply_orbit(req: OrbitDescriptorsUpdate) -> dict:
    """Build orbit from anchor descriptions and activate."""
    if _engine is None:
        return {"error": "No active session — start the DJ first"}

    try:
        db = _engine.db
        descriptions = [d.strip() for d in req.descriptions if d.strip()]
        if not descriptions:
            return {"error": "No anchor descriptions provided"}

        # Match descriptions to tracks using pure logic
        all_tracks = db.get_all_tracks()
        used_ids: set[str] = set()
        anchors: list[OrbitAnchorState] = []

        for desc in descriptions:
            tid = find_track(desc, all_tracks, used_ids)
            if tid is None:
                logger.warning("No match for anchor '%s'", desc)
                continue
            used_ids.add(tid)
            anchors.append(OrbitAnchorState(description=desc, track_id=tid))

        if not anchors:
            return {"error": "No anchor tracks matched in the library"}

        orbit = OrbitState(
            anchors=tuple(anchors),
            current_index=0,
            phase=OrbitPhase.DWELL,
            dwell_start_mono=time.monotonic(),
            allow_same_artist=req.allow_same_artist,
        )

        def _apply_orbit(state: DJState) -> DJState:
            return state.model_copy(update={
                "orbit": orbit,
                "pending_hop_type": "ORBIT DWELL",
                "orbit_picked": True,
            })

        _engine.enqueue_action(_apply_orbit)

        return {
            "status": "orbit activated",
            "anchors": [a.description for a in anchors],
        }
    except Exception as e:
        logger.error("Orbit apply error: %s", e)
        return {"error": str(e)}


@app.delete("/api/orbit")
def clear_orbit() -> dict:
    if _engine is None:
        return {"error": "No active session"}

    def _clear(state: DJState) -> DJState:
        return state.model_copy(update={"orbit": None})

    _engine.enqueue_action(_clear)
    return {"status": "orbit cleared"}


# ── Subprocess management ────────────────────────────────────────────

@app.get("/api/system")
def get_system_status() -> SystemStatus:
    return SystemStatus(
        mapview=ProcessStatus(
            running=_mapview_proc is not None and _mapview_proc.poll() is None,
            pid=_mapview_proc.pid if _mapview_proc and _mapview_proc.poll() is None else None,
        ),
        listener=ProcessStatus(
            running=_listener_proc is not None and _listener_proc.poll() is None,
            pid=_listener_proc.pid if _listener_proc and _listener_proc.poll() is None else None,
        ),
    )


@app.post("/api/mapview/start")
def start_mapview() -> dict:
    global _mapview_proc
    if _mapview_proc and _mapview_proc.poll() is None:
        return {"status": "already running", "pid": _mapview_proc.pid}
    _mapview_proc = subprocess.Popen(
        [sys.executable, "-m", "albart.mapview", "--view", "neighborhood"],
    )
    return {"status": "started", "pid": _mapview_proc.pid}


@app.post("/api/mapview/stop")
def stop_mapview() -> dict:
    global _mapview_proc
    if _mapview_proc and _mapview_proc.poll() is None:
        _mapview_proc.terminate()
        _mapview_proc.wait(timeout=5)
    _mapview_proc = None
    return {"status": "stopped"}


@app.post("/api/listener/start")
def start_listener() -> dict:
    global _listener_proc
    if _listener_proc and _listener_proc.poll() is None:
        return {"status": "already running", "pid": _listener_proc.pid}
    _listener_proc = subprocess.Popen(
        [sys.executable, "-m", "albart.listener"],
    )
    return {"status": "started", "pid": _listener_proc.pid}


@app.post("/api/listener/stop")
def stop_listener() -> dict:
    global _listener_proc
    if _listener_proc and _listener_proc.poll() is None:
        _listener_proc.terminate()
        _listener_proc.wait(timeout=5)
    _listener_proc = None
    return {"status": "stopped"}


# ── Auto-start ───────────────────────────────────────────────────────

def auto_start() -> None:
    """Auto-start a DJ session from whatever Spotify is playing."""
    global _engine, _engine_thread
    try:
        db = DatabaseClient(config=DatabaseConfig())
        spotify = SpotifyClient.create()
        broadcast = BroadcastClient()
        udp = UDPListener()

        _engine = Engine(
            db=db, playback=spotify, broadcast=broadcast,
            udp_listener=udp, mode="exact", song_k=10,
            projector=_load_projector(),
            norm_target=_get_norm_target(),
        )
        pb = spotify.poll_playback()
        seed = pb.current_track_id if pb.is_playing else None

        _engine_thread = threading.Thread(
            target=_run_engine, args=(_engine, seed), daemon=True,
        )
        _engine_thread.start()
        logger.info("Auto-started DJ session")
    except Exception as e:
        logger.warning("Auto-start failed: %s", e)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ALBart DJ Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (dev mode)")
    args = parser.parse_args()

    logger.info("Starting ALBart DJ Server on %s:%d", args.host, args.port)

    # Prevent macOS idle sleep
    _caffeinate = None
    if sys.platform == "darwin":
        try:
            _caffeinate = subprocess.Popen(
                ["caffeinate", "-i", "-w", str(os.getpid())],
            )
        except Exception:
            pass

    if args.reload:
        # In reload mode, uvicorn manages the process — auto_start runs
        # after each reload via the startup event below.
        uvicorn.run(
            "albart.server.app:app",
            host=args.host,
            port=args.port,
            log_level="warning",
            reload=True,
            reload_dirs=["albart"],
        )
    else:
        auto_start()
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


@app.on_event("startup")
def _on_startup():
    """Auto-start DJ session when running with --reload."""
    # Only auto-start if no engine is running (first load or after reload)
    if _engine is None:
        auto_start()
