"""Spotify playback effects — isolated side-effect wrapper around spotipy.

All Spotify API calls go through this client.  The engine calls these
methods; pure logic never touches the Spotify API.
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
        """Start playing a track.  Returns True on success.

        Falls back to the first available device if no active device
        is found (e.g. after sleep/wake).
        """
        uri = f"spotify:track:{track_id}"
        try:
            self.sp.start_playback(uris=[uri])
            return True
        except Exception:
            pass

        # No active playback — find any available device
        device = self._get_active_device()
        if device:
            try:
                self.sp.start_playback(uris=[uri], device_id=device)
                return True
            except Exception as e:
                logger.warning("Playback failed on device %s: %s", device, e)

        # Last resort: try to wake the first device by transferring to it
        try:
            devices = self.sp.devices()
            for d in devices.get("devices", []):
                try:
                    self.sp.transfer_playback(d["id"], force_play=False)
                    import time
                    time.sleep(1)  # give device a moment to wake
                    self.sp.start_playback(uris=[uri], device_id=d["id"])
                    logger.info("Woke device '%s' and started playback", d["name"])
                    return True
                except Exception:
                    continue
        except Exception:
            pass

        logger.warning("No Spotify device available for %s", track_id)
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

    def get_devices(self) -> List[dict]:
        try:
            result = self.sp.devices()
            return result.get("devices", [])
        except Exception:
            return []

    def set_volume(self, volume: int) -> bool:
        try:
            self.sp.volume(max(0, min(100, volume)))
            return True
        except Exception as e:
            logger.warning("Volume set failed: %s", e)
            return False

    def transfer_playback(self, device_id: str, force_play: bool = True) -> bool:
        try:
            self.sp.transfer_playback(device_id, force_play=force_play)
            return True
        except Exception as e:
            logger.warning("Transfer failed: %s", e)
            return False

    def _get_active_device(self) -> Optional[str]:
        """Find an active Spotify device."""
        try:
            devices = self.sp.devices()
            for d in devices.get("devices", []):
                if d.get("is_active"):
                    return d["id"]
            for d in devices.get("devices", []):
                return d["id"]
        except Exception:
            pass
        return None
