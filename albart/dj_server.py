"""ALBart DJ Server — web API wrapping the DJ for the React UI.

Runs the DJ in a background thread and exposes control via FastAPI.

Usage:
    python -m albart.dj_server
    python -m albart.dj_server --port 8765
"""

from __future__ import annotations

import argparse
import logging
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from albart.dj import DJ, find_seed_track

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


class StatusResponse(BaseModel):
    playing: bool
    current_track: Optional[TrackInfo] = None
    song_k: int
    set_distance: float
    mood_text: Optional[str] = None
    mood_descriptors: list[str]
    mood_threshold: float
    mood_in_count: int
    played_count: int
    total_tracks: int
    history: list[TrackInfo]


# ── Helpers ──────────────────────────────────────────────────────────────

def _track_info(dj: DJ, tid: str) -> TrackInfo:
    row = dj._db.get(tid)
    return TrackInfo(
        track_id=tid,
        title=row["title"] if row else "?",
        artist=row["artist"] if row else "?",
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

    history = [_track_info(_dj, tid) for tid in reversed(_dj._history[-20:])]

    return StatusResponse(
        playing=_dj_thread is not None and _dj_thread.is_alive(),
        current_track=current,
        song_k=_dj._song_k,
        set_distance=_dj.hop_multiplier,
        mood_text=_dj._mood_text,
        mood_descriptors=_dj._mood_descriptors,
        mood_threshold=_dj._mood_threshold,
        mood_in_count=int(_dj._mood_mask.sum()) if _dj._mood_mask is not None else _dj._N,
        played_count=len(_dj._played),
        total_tracks=_dj._N,
        history=history,
    )


@app.post("/api/start")
def start_session(req: StartRequest) -> dict:
    global _dj, _dj_thread

    with _dj_lock:
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
        try:
            _dj._sp.pause_playback()
        except Exception:
            pass  # device might be unavailable
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


@app.post("/api/skip")
def skip_track() -> dict:
    if _dj is None:
        return {"error": "No active session"}
    current_emb = None
    if _dj._history:
        current_emb = _dj._get_embedding(_dj._history[-1])
    if current_emb is not None:
        next_tid = _dj._pick_normal_hop(current_emb)
        if next_tid:
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
    _auto_start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
