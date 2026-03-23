"""Abstract display backend interface."""

import numpy as np


class DisplayBackend:
    """
    Abstract base for display backends. All display output must go through
    show_frame — never import display_sim or display_hub75 outside runtime/run.py.
    """

    def show_frame(self, rgb_array: np.ndarray) -> None:
        """
        Render a single frame to the display.

        Args:
            rgb_array: (32, 32, 3) uint8 numpy array in RGB order.
        """
        raise NotImplementedError

    def set_current_track(self, title: str, artist: str) -> None:
        """Notify the backend of the currently displayed track (sim only)."""
        pass

    def close(self) -> None:
        """Release display resources."""
        pass
