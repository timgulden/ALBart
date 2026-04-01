"""Pydantic request/response models for the DJ server API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ── Request models ───────────────────────────────────────────────────

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


class MoodUpdate(BaseModel):
    mood: str


class DescriptorsUpdate(BaseModel):
    descriptors: list[str]


class SeekRequest(BaseModel):
    position_ms: int


class OrbitRequest(BaseModel):
    description: str


class OrbitDescriptorsUpdate(BaseModel):
    descriptions: list[str]
    allow_same_artist: bool = False


# ── Response models ──────────────────────────────────────────────────

class TrackInfo(BaseModel):
    track_id: str
    title: str
    artist: str
    set_start: Optional[str] = None  # "NEW SET", "DWELL", "TRANSIT", or None


class DeviceInfo(BaseModel):
    id: str
    name: str
    type: str
    is_active: bool
    volume: int


class OrbitAnchorInfo(BaseModel):
    description: str
    track_id: str
    title: str
    artist: str
    art_url: str
    active: bool = False


class OrbitProgress(BaseModel):
    phase: str
    current_index: int
    prev_index: int
    segment_progress: float
    completed_segments: list[int] = []
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
    mode: str = "exact"
    dj_active: bool = True
    mood_text: Optional[str] = None
    mood_descriptors: list[str]
    mood_threshold: float
    mood_in_count: int
    played_count: int
    total_tracks: int
    history: list[TrackInfo]
    volume: int = -1
    orbit_active: bool = False
    orbit_anchors: list[OrbitAnchorInfo] = []
    orbit_progress: Optional[OrbitProgress] = None


class ProcessStatus(BaseModel):
    running: bool
    pid: Optional[int] = None


class SystemStatus(BaseModel):
    mapview: ProcessStatus
    listener: ProcessStatus
