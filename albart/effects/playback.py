"""Playback client protocol — the interface between the engine and any music player.

The engine and server interact with music playback exclusively through this
protocol.  ``SpotifyClient`` is the default implementation; alternative
implementations (e.g. MPD, local file player) can be substituted without
changing the engine or pure logic.

To implement a new playback backend:

    1. Create a class that satisfies ``PlaybackClient`` (the 6 methods below).
    2. Optionally implement ``MetadataClient`` if your system can look up
       track metadata by ID (used for on-the-fly ingestion display).
    3. Pass your client to ``Engine.__init__(playback=...)`` instead of
       ``SpotifyClient``.

The protocol is deliberately minimal — it covers only what the engine's
poll loop and the server's control endpoints need.  Player-specific features
(device management, transfer, etc.) can be exposed via additional methods
on your concrete class and wired into the server separately.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

from albart.core.state import PlaybackSnapshot


class PlaybackClient(Protocol):
    """Minimal interface the engine requires from any music player.

    All methods should be safe to call from any thread (the engine thread
    calls poll_playback/play_track/resume; the server thread calls
    pause/seek/set_volume).
    """

    def poll_playback(self) -> PlaybackSnapshot:
        """Poll the player for current playback state.

        Called every ~5 seconds by the engine's main loop.
        Should be fast and non-blocking.

        Returns:
            PlaybackSnapshot with current_track_id, progress_ms,
            duration_ms, is_playing, volume, and snapshot_time
            (monotonic seconds).
        """
        ...

    def play_track(self, track_id: str) -> bool:
        """Start playing a specific track.

        Args:
            track_id: opaque string identifier for the track.
                      For Spotify, this is the Spotify track ID.
                      For local files, this could be a file path,
                      database key, or MusicBrainz ID.

        Returns:
            True if playback started successfully.
        """
        ...

    def resume(self) -> bool:
        """Resume paused playback.  Returns True on success."""
        ...

    def pause(self) -> bool:
        """Pause playback.  Returns True on success."""
        ...

    def seek(self, position_ms: int) -> bool:
        """Seek to a position in the current track.  Returns True on success."""
        ...

    def set_volume(self, volume: int) -> bool:
        """Set playback volume (0-100).  Returns True on success."""
        ...


class MetadataClient(Protocol):
    """Optional interface for looking up track metadata by ID.

    Used by the engine to display title/artist for unknown tracks
    (e.g. while ingestion is in progress).  If your playback backend
    can resolve track IDs to metadata, implement this.

    If not implemented, the engine falls back to the database — tracks
    will show metadata only after ingestion completes.
    """

    def get_track_metadata(self, track_id: str) -> Optional[Tuple[str, str]]:
        """Look up (title, artist) for a track ID.

        Returns:
            (title, artist) tuple, or None if not found.
        """
        ...
