"""ALBart Control Center — web API for DJ, MapView, and Listener.

Runs the DJ in a background thread and can launch/stop the MapView
and Listener as subprocesses.  All controlled from the React UI.

Usage:
    python -m albart.dj_server
    python -m albart.dj_server --port 8765
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.responses import FileResponse

from albart.dj import DJ, find_seed_track
from albart.utils import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("albart.dj_server")

app = FastAPI(title="ALBart DJ", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_dj: Optional[DJ] = None
_dj_thread: Optional[threading.Thread] = None
_dj_lock = threading.Lock()

# Subprocess management for MapView and Listener
_mapview_proc: Optional[subprocess.Popen] = None
_listener_proc: Optional[subprocess.Popen] = None


# ── Models ───────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    seed: Optional[str] = None
    song_k: int = 10
    set_distance: float = 5.0
    mood: Optional[str] = None
    hop_interval: float = 30.0
    mode: str = "exact"


class SongKUpdate(BaseModel):
    song_k: int


class SetDistanceUpdate(BaseModel):
    set_distance: float


class MoodThresholdUpdate(BaseModel):
    threshold: float


class VolumeUpdate(BaseModel):
    volume: int


class DeviceInfo(BaseModel):
    id: str
    name: str
    type: str
    is_active: bool
    volume: int


class MoodUpdate(BaseModel):
    mood: str


class TrackInfo(BaseModel):
    track_id: str
    title: str
    artist: str
    set_start: bool = False


class OrbitAnchorInfo(BaseModel):
    description: str
    track_id: str
    title: str
    artist: str
    art_url: str
    active: bool = False


class OrbitProgress(BaseModel):
    phase: str  # "dwell" or "transit"
    current_index: int
    prev_index: int
    segment_progress: float
    completed_segments: list[int] = []  # indices of from-anchor for completed segments
    dwell_elapsed: float = 0
    dwell_duration: float = 0
    transit_remaining: int = 0
    transit_total: int = 0


class StatusResponse(BaseModel):
    playing: bool
    current_track: Optional[TrackInfo] = None
    progress_ms: int = 0
    duration_ms: int = 0
    song_k: int
    set_distance: float
    mood_text: Optional[str] = None
    mood_descriptors: list[str]
    mood_threshold: float
    mood_in_count: int
    played_count: int
    total_tracks: int
    history: list[TrackInfo]
    volume: int = -1  # -1 = unknown; synced from Spotify
    orbit_active: bool = False
    orbit_anchors: list[OrbitAnchorInfo] = []
    orbit_progress: Optional[OrbitProgress] = None


class SeekRequest(BaseModel):
    position_ms: int


# ── Helpers ──────────────────────────────────────────────────────────────

def _track_info(dj: DJ, tid: str) -> TrackInfo:
    row = dj._db.get(tid)
    return TrackInfo(
        track_id=tid,
        title=row["title"] if row else "?",
        artist=row["artist"] if row else "?",
        set_start=tid in dj._set_starts,
    )


def _run_dj(dj: DJ, seed: Optional[str]) -> None:
    try:
        dj.run(seed_track_id=seed)
    except Exception as e:
        logger.error("DJ loop error: %s", e)


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status() -> StatusResponse:
    if _dj is None:
        return StatusResponse(
            playing=False, song_k=10, set_distance=5.0,
            mood_descriptors=[], mood_threshold=0.35, mood_in_count=0,
            played_count=0, total_tracks=0, history=[],
        )

    current = None
    if _dj._history:
        current = _track_info(_dj, _dj._history[-1])

    history = [_track_info(_dj, tid) for tid in reversed(_dj._history)]

    # Progress from cache (no API call — DJ loop updates this every 5s)
    elapsed = time.monotonic() - _dj._cached_progress_time
    progress_ms = _dj._cached_progress_ms + int(elapsed * 1000)
    duration_ms = _dj._cached_duration_ms
    if progress_ms > duration_ms:
        progress_ms = duration_ms

    orbit_anchors = []
    orbit_progress = None
    if _dj._orbit is not None:
        for i, a in enumerate(_dj._orbit.anchors):
            row = _dj._db.get(a.track_id)
            orbit_anchors.append(OrbitAnchorInfo(
                description=a.description,
                track_id=a.track_id,
                title=row["title"] if row else "?",
                artist=row["artist"] if row else "?",
                art_url=f"/api/art/{a.track_id}",
                active=(i == _dj._orbit.current_index),
            ))
        prog = _dj._orbit.get_progress()
        orbit_progress = OrbitProgress(**prog)

    return StatusResponse(
        playing=_dj_thread is not None and _dj_thread.is_alive(),
        current_track=current,
        progress_ms=progress_ms,
        duration_ms=duration_ms,
        song_k=_dj._song_k,
        set_distance=_dj.hop_multiplier,
        mood_text=_dj._mood_text,
        mood_descriptors=_dj._mood_descriptors,
        mood_threshold=_dj._mood_threshold,
        mood_in_count=int(_dj._mood_mask.sum()) if _dj._mood_mask is not None else _dj._N,
        played_count=len(_dj._played),
        total_tracks=_dj._N,
        history=history,
        volume=_dj._cached_volume,
        orbit_active=_dj._orbit is not None,
        orbit_anchors=orbit_anchors,
        orbit_progress=orbit_progress,
    )


@app.post("/api/start")
def start_session(req: StartRequest) -> dict:
    global _dj, _dj_thread

    with _dj_lock:
        # Stop previous session if running
        if _dj is not None:
            _dj._stop_requested = True

        _dj = DJ(
            hop_interval_minutes=req.hop_interval,
            hop_multiplier=req.set_distance,
            mode=req.mode,
            temperature=req.song_k,  # DJ maps >1 values directly to K
            mood=req.mood,
        )
        _dj._song_k = max(1, min(50, req.song_k))

        seed = None
        if req.seed:
            if req.seed in _dj._id_to_idx:
                seed = req.seed
            else:
                seed = find_seed_track(req.seed, _dj._db)

        _dj_thread = threading.Thread(
            target=_run_dj, args=(_dj, seed), daemon=True
        )
        _dj_thread.start()

    return {"status": "started", "seed": seed}


@app.post("/api/stop")
def stop_session() -> dict:
    global _dj, _dj_thread
    if _dj is not None:
        _dj._stop_requested = True
        try:
            _dj._sp.pause_playback()
        except Exception:
            pass
    _dj = None
    _dj_thread = None
    return {"status": "stopped"}


@app.put("/api/song_k")
def set_song_k(req: SongKUpdate) -> dict:
    if _dj is None:
        return {"error": "No active session"}
    _dj._song_k = max(1, min(50, req.song_k))
    logger.info("Song K set to %d", _dj._song_k)
    return {"song_k": _dj._song_k}


@app.put("/api/mood_threshold")
def set_mood_threshold(req: MoodThresholdUpdate) -> dict:
    if _dj is None:
        return {"error": "No active session"}
    _dj._mood_threshold = max(0.0, min(0.6, req.threshold))
    _dj._recompute_mood_mask()
    logger.info("Mood threshold set to %.2f", _dj._mood_threshold)
    return {"threshold": _dj._mood_threshold}


@app.put("/api/set_distance")
def set_set_distance(req: SetDistanceUpdate) -> dict:
    if _dj is None:
        return {"error": "No active session"}
    _dj.hop_multiplier = max(1.0, min(20.0, req.set_distance))
    logger.info("Set distance set to %.1f×", _dj.hop_multiplier)
    return {"set_distance": _dj.hop_multiplier}


@app.post("/api/interpret")
def interpret_mood(req: MoodUpdate) -> dict:
    """Call Claude to expand mood text into descriptors. No DJ needed.

    Also pre-warms the CLAP text embedder so Apply is fast.
    """
    import anthropic

    try:
        client = anthropic.Anthropic()
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

        # Pre-warm CLAP embedder (loads model if needed, stays cached 5 min)
        positive = [ln for ln in lines if not ln.upper().startswith("NOT:")]
        if positive:
            from albart.text_embedder import embed_texts
            embed_texts(positive[:1])  # warm up with one descriptor

        return {"descriptors": lines, "mood": req.mood}
    except Exception as e:
        return {"error": str(e)}


class DescriptorsUpdate(BaseModel):
    descriptors: list[str]


@app.put("/api/mood")
def apply_mood(req: DescriptorsUpdate) -> dict:
    """Apply mood filter from edited descriptors (no Claude call)."""
    if _dj is None:
        return {"error": "No active session — start the DJ first"}
    try:
        from albart.text_embedder import embed_texts

        lines = [ln.strip() for ln in req.descriptors if ln.strip()]
        _dj._mood_descriptors = lines
        _dj._mood_text = "(edited)"

        positive = [ln for ln in lines if not ln.upper().startswith("NOT:")]
        negative = [ln[4:].strip() for ln in lines if ln.upper().startswith("NOT:")]

        _dj._mood_embs = embed_texts(positive) if positive else None
        _dj._mood_embs_neg = embed_texts(negative) if negative else None
        _dj._recompute_mood_mask()

        logger.info("Mood applied: %d positive, %d negative descriptors",
                    len(positive), len(negative))
        return {"status": "mood applied", "descriptors": lines}
    except Exception as e:
        return {"error": str(e)}


@app.put("/api/seek")
def seek_track(req: SeekRequest) -> dict:
    if _dj is None:
        return {"error": "No active session"}
    try:
        _dj._sp.seek_track(max(0, req.position_ms))
        return {"position_ms": req.position_ms}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/new_set")
def new_set() -> dict:
    """Force a long hop — start a new set immediately."""
    if _dj is None:
        return {"error": "No active session"}
    if _dj._orbit is not None and _dj._history:
        next_tid = _dj._pick_orbit_hop(_dj._history[-1], force_transit=True)
    else:
        next_tid = _dj._pick_long_hop()
    if next_tid:
        _dj._pending_hop_type = "LONG"
        _dj._next_pick = None       # clear any queued pick
        _dj._monitored_track = None
        _dj._play_track(next_tid)
        import time
        _dj._last_hop_time = time.monotonic()
        return {"status": "new set", "now_playing": _dj._track_name(next_tid)}
    return {"error": "Could not find a track for the long hop"}


@app.post("/api/skip")
def skip_track() -> dict:
    if _dj is None:
        return {"error": "No active session"}
    if not _dj._history:
        return {"error": "No track history"}
    if _dj._orbit is not None:
        next_tid = _dj._pick_orbit_hop(_dj._history[-1])
    else:
        current_emb = _dj._get_embedding(_dj._history[-1])
        next_tid = _dj._pick_normal_hop(current_emb) if current_emb is not None else None
    if next_tid:
        _dj._next_pick = None       # clear any queued pick
        _dj._monitored_track = None
        _dj._play_track(next_tid)
        return {"status": "skipped", "now_playing": _dj._track_name(next_tid)}
    return {"error": "Could not find next track"}


@app.post("/api/play/{track_id}")
def play_now(track_id: str) -> dict:
    """Interrupt and play this track immediately (starts new set)."""
    if _dj is None:
        return {"error": "No active session"}
    if track_id not in _dj._id_to_idx:
        return {"error": "Track not in library"}
    _dj._play_track(track_id)
    import time
    _dj._last_hop_time = time.monotonic()  # reset set timer
    return {"status": "playing", "track": _dj._track_name(track_id)}


@app.post("/api/queue/{track_id}")
def queue_next(track_id: str) -> dict:
    """Queue this track as the next to play (starts new set after it)."""
    if _dj is None:
        return {"error": "No active session"}
    if track_id not in _dj._id_to_idx:
        return {"error": "Track not in library"}
    _dj._next_pick = track_id
    _dj._pending_hop_type = "normal"
    import time
    _dj._last_hop_time = time.monotonic()
    return {"status": "queued", "track": _dj._track_name(track_id)}


def _normalize_search(s: str) -> str:
    """Normalize for fuzzy matching: lowercase, collapse dashes/punctuation."""
    import re
    s = s.lower()
    s = s.replace("–", "-").replace("—", "-").replace("'", "'").replace("'", "'")
    s = re.sub(r'[.\-_/\\]+', ' ', s)  # dots, dashes, underscores → spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


@app.get("/api/search")
def search_tracks(q: str, limit: int = 10) -> list[TrackInfo]:
    if _dj is None:
        return []
    results = []
    q_norm = _normalize_search(q)
    for tid, row in _dj._db.items():
        if tid not in _dj._id_to_idx:
            continue
        title = _normalize_search(row["title"] or "")
        artist = _normalize_search(row["artist"] or "")
        if q_norm in title or q_norm in artist:
            results.append(TrackInfo(
                track_id=tid,
                title=row["title"] or "",
                artist=row["artist"] or "",
            ))
            if len(results) >= limit:
                break
    return results


@app.get("/api/devices")
def get_devices() -> list[DeviceInfo]:
    if _dj is None:
        return []
    try:
        devices = _dj._sp.devices()
        return [
            DeviceInfo(
                id=d["id"], name=d["name"], type=d["type"],
                is_active=d["is_active"],
                volume=d.get("volume_percent", 0),
            )
            for d in devices.get("devices", [])
        ]
    except Exception:
        return []


@app.put("/api/volume")
def set_volume(req: VolumeUpdate) -> dict:
    if _dj is None:
        return {"error": "No active session"}
    try:
        _dj._sp.volume(max(0, min(100, req.volume)))
        return {"volume": req.volume}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/device/{device_id}")
def set_device(device_id: str) -> dict:
    if _dj is None:
        return {"error": "No active session"}
    try:
        _dj._sp.transfer_playback(device_id, force_play=True)
        return {"status": "transferred", "device": device_id}
    except Exception as e:
        return {"error": str(e)}


# ── Album art ────────────────────────────────────────────────────────────

@app.get("/api/art/{track_id}")
def get_art(track_id: str):
    """Serve album art from the local data directory."""
    art_path = DATA_DIR / "art_original" / f"{track_id}.jpg"
    if art_path.exists():
        return FileResponse(art_path, media_type="image/jpeg")
    return {"error": "Art not found"}


# ── Orbit navigation ─────────────────────────────────────────────────────

class OrbitRequest(BaseModel):
    description: str


class OrbitDescriptorsUpdate(BaseModel):
    descriptions: list[str]
    allow_same_artist: bool = False


@app.post("/api/orbit/interpret")
def interpret_orbit(req: OrbitRequest) -> dict:
    """Send journey description + library sample to Claude.

    Claude suggests specific tracks from the library to use as anchors.
    Returns track suggestions that the user can edit before applying.
    """
    import anthropic

    if _dj is None:
        return {"error": "No active session — start the DJ first"}

    # Build a sample of the library for Claude (artist — title, grouped)
    from collections import defaultdict
    by_artist: dict[str, list[str]] = defaultdict(list)
    for tid in _dj._id_list:
        r = _dj._db.get(tid)
        if r:
            by_artist[r["artist"] or "Unknown"].append(r["title"] or "?")

    # Format: all tracks listed per artist
    lib_lines = []
    for artist in sorted(by_artist.keys()):
        titles = by_artist[artist]
        lib_lines.append(f"{artist}: {', '.join(titles)}")
    library_text = '\n'.join(lib_lines)

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    "An automated DJ navigates my music library by drifting "
                    "between anchor tracks based on musical similarity. It plays "
                    "many tracks between each anchor, so anchors should be spread "
                    "far apart. The DJ handles transitions — you just pick the "
                    "highlights.\n\n"
                    "The user described this journey:\n"
                    f'"{req.description}"\n\n'
                    "Pick 5-6 tracks from my library that are interesting "
                    "highlights along this journey. Each track should clearly "
                    "fit with the description — no in-between or transitional "
                    "picks. If the description mentions multiple genres, places, "
                    "or eras, pick several distinct highlights within each one "
                    "to show its range. The tracks form a cycle — the last "
                    "should connect back to the first.\n\n"
                    "My library:\n\n"
                    f"{library_text}\n\n"
                    "Rules:\n"
                    "- Each track MUST exist in the library above (copy exactly)\n"
                    "- Pick iconic, on-the-nose choices that clearly belong to "
                    "the described journey (e.g. if it says 'grunge', pick an "
                    "actual grunge track, not something adjacent)\n"
                    "- Do NOT pick transitional tracks between themes — the DJ "
                    "navigates between anchors on its own\n"
                    "- Spread picks across the full journey, with variety within "
                    "each theme (e.g. two different grunge acts, not two tracks "
                    "by the same artist)\n\n"
                    "Reply in JSON: {\"tracks\": [\"Artist \u2014 Title\", ...]}\n"
                    "Return only the JSON object, no markdown fences, no commentary."
                ),
            }],
        )
        import json
        import re as _re
        raw = response.content[0].text.strip()
        logger.info("Orbit interpret raw response: %s", raw[:500])
        # Strip markdown code fences if present
        fence_match = _re.search(r'```(?:json)?\s*\n?(.*?)```', raw, _re.DOTALL)
        if fence_match:
            raw = fence_match.group(1).strip()
        # Find JSON object in the response
        json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        try:
            data = json.loads(raw)
            lines = [t.strip() for t in data.get("tracks", []) if t.strip()]
        except json.JSONDecodeError as e:
            logger.error("JSON parse failed: %s — raw: %s", e, raw[:200])
            # Fallback: try line-based parsing
            lines = []
            for ln in response.content[0].text.strip().split("\n"):
                ln = ln.strip().rstrip(".")
                ln = _re.sub(r'^\d+[.)]\s*', '', ln)
                if ("\u2014" in ln or " -- " in ln or " - " in ln) and len(ln) > 5:
                    lines.append(ln)
        logger.info("Orbit interpret: %d tracks parsed", len(lines))

        if not lines:
            # Claude likely returned commentary instead of tracks
            raw_text = response.content[0].text.strip()
            return {"error": f"No tracks found in response: {raw_text[:200]}"}

        # Ask Claude whether same-artist runs make sense for this journey
        allow_same_artist = False
        try:
            artist_resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4,
                messages=[{
                    "role": "user",
                    "content": (
                        "A music DJ is about to play a session described as:\n"
                        f'"{req.description}"\n\n'
                        "Would it make sense for this session to play multiple "
                        "tracks in a row by the same artist? Answer with a "
                        "single word: Yes or No."
                    ),
                }],
            )
            answer = artist_resp.content[0].text.strip().lower()
            allow_same_artist = answer.startswith("yes")
            logger.info("Same-artist runs: %s (Claude said: %s)",
                        allow_same_artist, answer)
        except Exception as e:
            logger.warning("Same-artist check failed: %s", e)

        return {
            "descriptions": lines,
            "journey": req.description,
            "allow_same_artist": allow_same_artist,
        }
    except Exception as e:
        return {"error": str(e)}


@app.put("/api/orbit/apply")
def apply_orbit(req: OrbitDescriptorsUpdate) -> dict:
    """Embed anchor descriptions and activate orbit navigation."""
    if _dj is None:
        return {"error": "No active session — start the DJ first"}
    try:
        from albart.orbit import build_orbit

        descriptions = [d.strip() for d in req.descriptions if d.strip()]
        if not descriptions:
            return {"error": "No anchor descriptions provided"}

        orbit = build_orbit(
            descriptions,
            _dj._embeddings,
            _dj._umap_5d,
            _dj._id_list,
            _dj._db,
            allow_same_artist=req.allow_same_artist,
        )
        # Start dwelling at anchor 0 (not yet arrived via transit)
        orbit.current_index = 0
        orbit.start_dwell()
        orbit._arrived = False              # no transit completed yet
        orbit._completed_segments.clear()   # no segments traversed yet

        _dj._orbit = orbit
        logger.info("Orbit activated: %d anchors, dwelling at [0]: %s",
                     len(descriptions), orbit.target.description)

        # Queue a track one hop from the first anchor (using 512D, not 5D)
        anchor_emb = orbit.anchors[0].embedding_512
        start_tid = _dj._pick_normal_hop(anchor_emb)
        if start_tid:
            _dj._next_pick = start_tid
            _dj._pending_hop_type = "LONG"
            _dj._orbit_picked = True
            # Set monitored track so DJ detects when current track ends
            current = _dj._get_current_spotify_track()
            _dj._monitored_track = current

        return {
            "status": "orbit activated",
            "anchors": descriptions,
            "queued": _dj._track_name(start_tid) if start_tid else None,
        }
    except Exception as e:
        logger.error("Orbit apply error: %s", e)
        return {"error": str(e)}


@app.delete("/api/orbit")
def clear_orbit() -> dict:
    """Deactivate orbit navigation, revert to normal DJ behavior."""
    if _dj is None:
        return {"error": "No active session"}
    _dj._orbit = None
    logger.info("Orbit cleared")
    return {"status": "orbit cleared"}


# ── Subprocess management (MapView + Listener) ───────────────────────────

class ProcessStatus(BaseModel):
    running: bool
    pid: Optional[int] = None


class SystemStatus(BaseModel):
    mapview: ProcessStatus
    listener: ProcessStatus


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
    logger.info("MapView started (pid %d)", _mapview_proc.pid)
    return {"status": "started", "pid": _mapview_proc.pid}


@app.post("/api/mapview/stop")
def stop_mapview() -> dict:
    global _mapview_proc
    if _mapview_proc and _mapview_proc.poll() is None:
        _mapview_proc.terminate()
        _mapview_proc.wait(timeout=5)
        logger.info("MapView stopped")
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
    logger.info("Listener started (pid %d)", _listener_proc.pid)
    return {"status": "started", "pid": _listener_proc.pid}


@app.post("/api/listener/stop")
def stop_listener() -> dict:
    global _listener_proc
    if _listener_proc and _listener_proc.poll() is None:
        _listener_proc.terminate()
        _listener_proc.wait(timeout=5)
        logger.info("Listener stopped")
    _listener_proc = None
    return {"status": "stopped"}


def _auto_start() -> None:
    """Auto-start a DJ session from whatever Spotify is currently playing."""
    global _dj, _dj_thread
    try:
        _dj = DJ(mode="exact", temperature=10)
        _dj._song_k = 10
        current = _dj._get_current_spotify_track()
        seed = current if current and current in _dj._id_to_idx else None
        _dj_thread = threading.Thread(
            target=_run_dj, args=(_dj, seed), daemon=True
        )
        _dj_thread.start()
        logger.info("Auto-started DJ session")
    except Exception as e:
        logger.warning("Auto-start failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="ALBart DJ Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    logger.info("Starting ALBart DJ Server on %s:%d", args.host, args.port)

    # Prevent macOS idle sleep so the DJ keeps running with screens off
    _caffeinate = None
    if sys.platform == "darwin":
        try:
            _caffeinate = subprocess.Popen(
                ["caffeinate", "-i", "-w", str(os.getpid())],
            )
            logger.info("caffeinate: preventing idle sleep (pid %d)", _caffeinate.pid)
        except Exception as e:
            logger.warning("Could not start caffeinate: %s", e)

    _auto_start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
