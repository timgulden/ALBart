"""Spotify playback effects — isolated side-effect wrapper around spotipy.

Implements the PlaybackClient and MetadataClient protocols defined in
``albart.effects.playback``.  The engine and server interact with Spotify
exclusively through these interfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from albart.core.state import PlaybackSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpotifyClient:
    """Frozen wrapper around spotipy.Spotify.

    The underlying ``sp`` object is mutable (token refresh), but we
    treat this dataclass as a configuration holder — create once, use
    everywhere.
    """
    sp: spotipy.Spotify = field(repr=False)

    @staticmethod
    def create(scope: str = "user-read-playback-state,user-modify-playback-state") -> SpotifyClient:
        """Create and verify a SpotifyClient from environment variables."""
        sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(scope=scope),
            retries=0,
        )
        user = sp.current_user()
        logger.info("Spotify: logged in as %s", user["display_name"])
        return SpotifyClient(sp=sp)

    def poll_playback(self) -> PlaybackSnapshot:
        """Single API call to get current playback state."""
        import time
        try:
            pb = self.sp.current_playback()
            if pb and pb.get("item"):
                volume = -1
                if pb.get("device"):
                    volume = pb["device"].get("volume_percent", -1)
                return PlaybackSnapshot(
                    progress_ms=pb.get("progress_ms", 0),
                    duration_ms=pb["item"].get("duration_ms", 0),
                    snapshot_time=time.monotonic(),
                    is_playing=pb.get("is_playing", False),
                    volume=volume,
                    current_track_id=pb["item"].get("id"),
                )
            # Nothing playing
            return PlaybackSnapshot(
                snapshot_time=time.monotonic(),
                is_playing=pb.get("is_playing", False) if pb else False,
                volume=pb["device"].get("volume_percent", -1) if pb and pb.get("device") else -1,
            )
        except Exception as e:
            logger.warning("Spotify poll error: %s", e)
            import time
            return PlaybackSnapshot(snapshot_time=time.monotonic())

    def play_track(self, track_id: str) -> bool:
        """Start playing a track on the currently active Spotify device.

        Does NOT attempt to select or transfer devices — the user
        manages device selection directly from Spotify.
        """
        uri = f"spotify:track:{track_id}"
        try:
            self.sp.start_playback(uris=[uri])
            return True
        except Exception as e:
            logger.warning("Playback failed for %s: %s", track_id, e)
            return False

    def resume(self) -> bool:
        try:
            self.sp.start_playback()
            return True
        except Exception as e:
            logger.warning("Could not resume: %s", e)
            return False

    def pause(self) -> bool:
        try:
            self.sp.pause_playback()
            return True
        except Exception as e:
            logger.warning("Could not pause: %s", e)
            return False

    def seek(self, position_ms: int) -> bool:
        try:
            self.sp.seek_track(max(0, position_ms))
            return True
        except Exception as e:
            logger.warning("Seek failed: %s", e)
            return False

    def set_volume(self, volume: int) -> bool:
        try:
            self.sp.volume(max(0, min(100, volume)))
            return True
        except Exception as e:
            logger.warning("Volume set failed: %s", e)
            return False

    def get_track_metadata(self, track_id: str) -> Optional[tuple[str, str]]:
        """Look up (title, artist) for a Spotify track ID.

        Implements MetadataClient protocol.
        """
        try:
            item = self.sp.track(track_id)
            title = item.get("name")
            artists = item.get("artists", [])
            artist = ", ".join(a["name"] for a in artists) if artists else None
            return (title, artist)
        except Exception:
            return None

